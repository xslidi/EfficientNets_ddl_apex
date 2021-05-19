"""
Modified from https://github.com/rwightman/pytorch-image-models/blob/5fcdd4b822fc6cf2166e1eac2914ed34e3b2a731/timm/models/resnet.py

ResNet family model zoo
"""

import math

import torch
import torch.nn as nn
from models.layers import SEModule

def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64, dilation=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, se_r=0):
        super(BasicBlock, self).__init__()

        if cardinality != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports cardinality=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        outplanes = planes * self.expansion

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, bias=False) 
        self.bn1 = norm_layer(planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(planes, outplanes, kernel_size=3, bias=False)
        self.bn2 = norm_layer(outplanes)

        self.se = SEModule(outplanes, int(outplanes * se_r)) if se_r else None

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def zero_init_residual(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.se is not None:
            x = self.se(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.act2(x)   

        return x     

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=1, base_width=64, dilation=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, se_r=0):
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        outplanes = planes * self.expansion

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=dilation, bias=False)
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = SEModule(outplanes, int(outplanes * se_r)) if se_r else None

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation       

    def zero_init_residual(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.act3(x)

        return x

def downsample_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    dilation = dilation if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, dilation)

    return nn.Sequential(*[
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, p, dilation, bias=False),
        norm_layer(out_channels)
    ])

def downsample_avg(in_channels, out_channels, kernel_size, stride=1, dilation=1, norm_layer=None):
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        pool = nn.AvgPool2d(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, 1, bias=False),
        norm_layer(out_channels)
    ])

def make_blocks(block_fn, channels, block_repeats, inplanes, output_stride=32, down_kernel_size=1, avg_down=False, **kwargs):

    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks) in enumerate(zip(channels, block_repeats)):
        stage_name = f'layer{stage_idx + 1}'
        stride = 1 if stage_idx == 0 else 2
        if net_stride > output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride
        
        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes, out_channels=planes * block_fn.expansion, kernel_size=down_kernel_size,
                stride=stride, dilation=dilation, norm_layer=nn.BatchNorm2d)

            downsample = downsample_avg(**down_kwargs) if avg_down else downsample_conv(**down_kwargs)

        block_kwargs = dict(dilation=dilation, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            blocks.append(block_fn(
                inplanes, planes, stride, downsample, **block_kwargs
            ))
            inplanes = planes * block_fn.expansion 

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info

class ResNet(nn.Module):

    """Resnet / ResNeXt / SE-ResNeXt / SE-Net
    
    """

    def __init__(self, block, layers, num_classes=1000, in_channels=3, cardinality=1, 
                base_width=64, stem_width=64, stem_type='', output_stride=32, down_kernel_size=1, avg_down=False, 
                act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, global_pool='avg', block_args=None, zero_init_residual=True):
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        super(ResNet, self).__init__()

        # stem
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else 64
        if deep_stem:
            stem_chs = (stem_width, stem_width)
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_channels, stem_chs[0], 3, stride=2, padding=1, bias=False),
                norm_layer(stem_chs[0]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                norm_layer(stem_chs[1]),
                act_layer(inplace=True),
                nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)])
        else:
            self.conv1 = nn.Conv2d(in_channels, inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = norm_layer(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        # Stem Pooling
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels = [64, 128, 256, 512]
        stage_modules, stage_feature_info = make_blocks(block, channels, layers, inplanes, cardinality=cardinality, base_width=base_width, output_stride=output_stride, avg_down=avg_down, down_kernel_size=down_kernel_size, act_layer=act_layer, norm_layer=norm_layer, **block_args)
        for stage in stage_modules:
            self.add_module(*stage)
        self.feature_info.extend(stage_feature_info)

        # Head (Pooling and Classifier)
        self.num_features = 512 * block.expansion
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        num_pooled_features = self.num_features
        self.fc = nn.Linear(num_pooled_features, num_classes)

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if hasattr(m, 'zero_init_residual'):
                    m.zero_init_residual()

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)









            