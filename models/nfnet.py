# Modified from original paper


from functools import partial
from typing import OrderedDict
import torch
from torch import nn
from models.layers import ScaledWSConv2d, SEModule, DropConnect
from utils import make_divisible

class DownsampleAvg(nn.Module):
    """ AvgPool Downsampling as in 'D' ResNet variants without norm layer."""
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, conv_layer=ScaledWSConv2d):
        super().__init__()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            self.pool = nn.AvgPool2d(2, avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = nn.Identity()

        self.conv = conv_layer(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        return self.conv(self.pool(x))
        

class NFblock(nn.Module):
    """NF-RegNet block."""

    def __init__(self, in_channels, out_channels=None, stride=1, alpha=1.0, beta=1.0, expansion=2.25,
                se_r=0.5, group_size=8, act_layer=None, conv_layer=None, skipinit=False, attn_gain=2.0,
                stochdepth_rate=None):
        super().__init__()
        out_channels = out_channels or in_channels
        width = int(in_channels * expansion)
        if group_size is None:
            groups= 1 
        else:
            width = make_divisible(width, group_size)
            groups = width // group_size
            # width = int(group_size * groups)
        self.alpha = alpha  
        self.beta = beta
        if in_channels != out_channels or stride != 1:
            self.downsample = DownsampleAvg(in_channels, out_channels, stride=stride, 
                                            conv_layer=conv_layer)
        else:
            self.downsample = None

        self.act1 = act_layer()
        self.conv1 = conv_layer(in_channels, width, 1)
        self.act2 = act_layer(inplace=True)   
        self.conv2 = conv_layer(width, width, 3, stride=stride, groups=groups)
        
        se_width = max(1, int(width * se_r))
        self.attn = SEModule(width, se_width, act_layer=act_layer)
        self.attn_gain = attn_gain
        self.act3 = act_layer()
        self.conv3 = conv_layer(width, out_channels, 1, gain_init=1.0 if skipinit else 0.)
        self.stoch_depth = DropConnect(stochdepth_rate) if stochdepth_rate > 0 else nn.Identity()
        self.skipinit_gain = nn.Parameter(torch.tensor(0.)) if skipinit else None

    def forward(self, x):
        out = self.act1(x) * self.beta

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.conv2(self.act2(out))
        if self.attn is not None:
            out = self.attn_gain * self.attn(out)

        out = self.conv3(self.act3(out))
        out = self.stoch_depth(out)

        if self.skipinit_gain is not None:
            out = out * self.skipinit_gain
        
        out = self.alpha * out + shortcut
        return out

def create_stem(in_channels, out_channels, stem_type='', conv_layer=None, 
                act_layer=None, preact_feature=True):
    stem_stride = 2
    stem_feature = dict(num_chs=out_channels, reduction=2, module='stem.conv')
    stem = OrderedDict()
    if 'deep' in stem_type:
        stem_chs = (out_channels // 2, out_channels // 2, out_channels)
        strides = (2, 1, 1)
        stem_feature = dict(num_chs=out_channels // 2, reduction=2, module='stem.conv2')
        last_idx = len(stem_chs) - 1
        for i, (c, s) in enumerate(zip(stem_chs, strides)):
            stem[f'conv{i + 1}'] = conv_layer(in_channels, c, kernel_size=3, stride=s)
            if i != last_idx:
                stem[f'act{i + 2}'] = act_layer(inplace=True)
            in_channels = c
    elif '3x3' in stem_type:
        stem['conv'] = conv_layer(in_channels, out_channels, kernel_size=3, stride=2)
    else:
        stem['conv'] = conv_layer(in_channels, out_channels, kernel=7, stride=2)

    if 'pool' in stem_type:
        stem['pool'] = nn.MaxPool2d(3, stride=2, padding=1)
        stem_stride = 4

    return nn.Sequential(stem), stem_stride, stem_feature

_nonlin_gamma = dict(
    identity=1.0,
    relu=1.7139588594436646,
    silu=1.7881293296813965,
)

class NormFreeNet(nn.Module):
    """ Normalization-Free Network"""

    def __init__(self, arg, stem_chs=48,  in_channels=3, global_pool='avg', 
                output_stride=32, drop_rate=0.2, depths=(1, 3, 6, 6), 
                channels=(48, 104, 208, 440), width_factor=0.75, stochdepth_rate=0.1):
        super().__init__()
        self.num_classes = arg.num_classes
        self.drop_rate = drop_rate
        conv_layer = ScaledWSConv2d
        act_layer = nn.SiLU
        conv_layer = partial(conv_layer, gamma=_nonlin_gamma['silu'])

        stem_chs = int(stem_chs * width_factor)
        stem_chs = make_divisible(stem_chs)
        self.stem, stem_stride, stem_feat = create_stem(in_channels, stem_chs, '3x3', 
                                                        conv_layer=conv_layer, act_layer=act_layer)
        
        prev_chs = stem_chs
        net_stride = stem_stride
        dilation = 1
        expected_var = 1.0
        stages = []
        stochdepth_rates = [x.tolist() for x in torch.linspace(0, stochdepth_rate, sum(depths)).split(depths)]
        for stage_idx, stage_depth in enumerate(depths):
            stride = 1 if stage_idx == 0 and stem_stride > 2 else 2
            if net_stride >= output_stride and stride > 1:
                dilation *= stride
                stride = 1
            net_stride *= stride

            blocks = []
            for block_idx in range(depths[stage_idx]):
                first_block = block_idx == 0 and stage_idx == 0
                out_channels = int(channels[stage_idx] * width_factor)
                out_channels = make_divisible(out_channels, 8)
                blocks +=[NFblock(
                    in_channels=prev_chs, out_channels=out_channels,
                    alpha=0.2,
                    beta = 1. / expected_var ** 0.5,
                    stride=stride if block_idx == 0 else 1,
                    group_size=8,
                    act_layer=act_layer,
                    conv_layer=conv_layer,
                    expansion=1 if first_block else 2.25,
                    stochdepth_rate=stochdepth_rates[stage_idx][block_idx]
                )]
                if block_idx == 0:
                    expected_var = 1
                expected_var += 0.2 ** 2
                prev_chs = out_channels
            stages += [nn.Sequential(*blocks)]
        self.stages = nn.Sequential(*stages)
        num_features = 1280 * channels[-1] // 440
        self.num_features = int(width_factor * num_features)
        self.final_conv = conv_layer(prev_chs, self.num_features, 1)
        self.final_act = act_layer(inplace=True)

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        if drop_rate > 0:
            self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(self.num_features, self.num_classes, bias=True)
        
        self._init_value()

    def _init_value(self):
        for n, m in self.named_modules():
            if 'fc' in n and isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0., .01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.final_conv(x)
        x = self.final_act(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        if self.drop_rate > 0:
            x = self.drop(x)
        x = self.fc(x)
        return x

