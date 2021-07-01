import math

import torch
import torch.nn as nn

from models.layers import conv2d, Flatten, SEModule, DropConnect, Conv_Bn_Act


class MBConv(nn.Module):
    def __init__(self, in_, out_, expand,
                 kernel_size, stride, skip,
                 se_ratio, dc_ratio=0.2, bn_eps=1e-3, bn_momentum=0.01):
        super().__init__()
        mid_ = in_ * expand
        self.expand_conv = Conv_Bn_Act(in_, mid_, kernel_size=1, eps=bn_eps, momentum=bn_momentum, bias=False) if expand != 1 else nn.Identity()

        self.depth_wise_conv = Conv_Bn_Act(mid_, mid_,
                                           kernel_size=kernel_size, stride=stride,
                                           eps=bn_eps, momentum=bn_momentum, groups=mid_, bias=False)

        self.se = SEModule(mid_, int(in_ * se_ratio)) if se_ratio > 0 else nn.Identity()

        self.project_conv = nn.Sequential(
            conv2d(mid_, out_, k=1, stride=1, bias=False),
            nn.BatchNorm2d(out_, bn_eps, bn_momentum)
        )

        # if _block_args.id_skip:
        # and all(s == 1 for s in self._block_args.strides)
        # and self._block_args.input_filters == self._block_args.output_filters:
        self.skip = skip and (stride == 1) and (in_ == out_)

        # DropConnect
        self.dropconnect = DropConnect(dc_ratio) if dc_ratio > 0 else nn.Identity()
        # Original TF Repo not using drop_rate
        # https://github.com/tensorflow/tpu/blob/05f7b15cdf0ae36bac84beb4aef0a09983ce8f66/models/official/efficientnet/efficientnet_model.py#L408
        # self.dropconnect = nn.Identity()

    def forward(self, inputs):
        expand = self.expand_conv(inputs)
        x = self.depth_wise_conv(expand)
        x = self.se(x)
        x = self.project_conv(x)
        if self.skip:
            x = self.dropconnect(x)
            x = x + inputs
        return x


class MBBlock(nn.Module):
    def __init__(self, in_, out_, expand, kernel, stride, num_repeat, skip, se_ratio, drop_connect_ratio=0.2, dc_step=0, bn_eps=1e-3, bn_momentum=0.01):
        super().__init__()
        layers = [MBConv(in_, out_, expand, kernel, stride, skip, se_ratio, drop_connect_ratio, bn_eps, bn_momentum)]
        for i in range(1, num_repeat):
            layers.append(MBConv(out_, out_, expand, kernel, 1, skip, se_ratio, drop_connect_ratio + i * dc_step, bn_eps, bn_momentum))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class EfficientNet(nn.Module):
    def __init__(self, width_coeff, depth_coeff,
                 depth_div=8, min_depth=None,
                 dropout_rate=0.2, drop_connect_rate=0.2,
                 num_classes=1000, se_r=0.25, bn_eps=1e-3, bn_momentum=0.01):
        super().__init__()
        min_depth = min_depth or depth_div
        channals = [32, 16, 24, 40, 80, 112, 192, 320]
        expand = [1, 6, 6, 6, 6, 6, 6]
        k = [3, 3, 5, 3, 5, 5, 3]
        s = [1, 2, 2, 2, 1, 2, 1]
        repeat = [1, 2, 2, 3, 3, 4, 1]
        
        def renew_ch(x):
            if not width_coeff:
                return x

            x *= width_coeff
            new_x = max(min_depth, int(x + depth_div / 2) // depth_div * depth_div)
            if new_x < 0.9 * x:
                new_x += depth_div
            return int(new_x)

        def renew_repeat(x):
            return int(math.ceil(x * depth_coeff))

        def blocks_builter(channals, expand, k, s, repeat):
            rn_channals = [renew_ch(ch) for ch in channals]
            rn_repeat = [renew_repeat(rp) for rp in repeat]
            total_num = sum(rn_repeat)
            dc_step = drop_connect_rate / total_num
            blocks = []
            save_ratio = 0.0

            for i in range(len(rn_repeat)):                
                blocks.append(MBBlock(rn_channals[i], rn_channals[i+1], expand[i], k[i], s[i], rn_repeat[i], True, se_r, save_ratio, dc_step, bn_eps, bn_momentum))
                save_ratio = save_ratio + rn_repeat[i] * dc_step

            return blocks

        self.stem = Conv_Bn_Act(3, renew_ch(32), kernel_size=3, eps=bn_eps, momentum=bn_momentum, stride=2, bias=False)
        
        self.blocks = nn.Sequential(*blocks_builter(channals, expand, k, s, repeat))
        # self.blocks = nn.Sequential(
        #     #       input channel  output    expand  k  s                   skip  se
        #     MBBlock(renew_ch(32), renew_ch(16), 1, 3, 1, renew_repeat(1), True, 0.25, drop_connect_rate),
        #     MBBlock(renew_ch(16), renew_ch(24), 6, 3, 2, renew_repeat(2), True, 0.25, drop_connect_rate),
        #     MBBlock(renew_ch(24), renew_ch(40), 6, 5, 2, renew_repeat(2), True, 0.25, drop_connect_rate),
        #     MBBlock(renew_ch(40), renew_ch(80), 6, 3, 2, renew_repeat(3), True, 0.25, drop_connect_rate),
        #     MBBlock(renew_ch(80), renew_ch(112), 6, 5, 1, renew_repeat(3), True, 0.25, drop_connect_rate),
        #     MBBlock(renew_ch(112), renew_ch(192), 6, 5, 2, renew_repeat(4), True, 0.25, drop_connect_rate),
        #     MBBlock(renew_ch(192), renew_ch(320), 6, 3, 1, renew_repeat(1), True, 0.25, drop_connect_rate)
        # )

        self.head = nn.Sequential(
            Conv_Bn_Act(renew_ch(320), renew_ch(1280), kernel_size=1, eps=bn_eps, momentum=bn_momentum, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(dropout_rate, True) if dropout_rate > 0 else nn.Identity(),
            Flatten(),
            nn.Linear(renew_ch(1280), num_classes)
        )

        self.init_weights()
    
    def init_weights(self):
        for n, m in self.named_modules():
            self._init_weight_goog(m,n)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out")
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         init_range = 1.0 / math.sqrt(m.weight.shape[0])
        #         nn.init.uniform_(m.weight, -init_range, init_range)
        #         m.bias.data.zero_()
                
    def _init_weight_goog(self, m, n='', fix_group_fanout=True):
        """ Weight initialization as per Tensorflow official implementations.
        Args:
            m (nn.Module): module to init
            n (str): module name
            fix_group_fanout (bool): enable correct (matching Tensorflow TPU impl) fanout calculation w/ group convs
        Handles layers in EfficientNet, EfficientNet-CondConv, MixNet, MnasNet, MobileNetV3, etc:
        * https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_model.py
        * https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
        """
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            if fix_group_fanout:
                fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            fan_out = m.weight.size(0)  # fan-out
            fan_in = 0
            if 'routing_fn' in n:
                fan_in = m.weight.size(1)
            init_range = 1.0 / math.sqrt(fan_in + fan_out)
            m.weight.data.uniform_(-init_range, init_range)
            m.bias.data.zero_()

    def forward(self, inputs):
        stem = self.stem(inputs)
        x = self.blocks(stem)
        head = self.head(x)
        return head


if __name__ == "__main__":
    print("Efficient B0 Summary")
    net = EfficientNet(1, 1)
    print(net)
    # for k, v in net.named_parameters():
    #     print(k)
    # print("----------------------------------------")
    # for k, ema_v in net.state_dict().items():
    #     print(k)
    # from torchsummary import summary
    # summary(net.cuda(), (3, 224, 224))
