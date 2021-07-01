import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding


# Calculate symmetric padding for a convolution
def get_padding(kernel_size, stride=1, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

def conv2d(w_in, w_out, k, stride=1, groups=1, bias=False):
    """Helper for building a conv2d layer."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    s, p, g, b = stride, ((stride - 1) + (k - 1)) // 2, groups, bias
    return nn.Conv2d(w_in, w_out, k, stride=s, padding=p, groups=g, bias=b)
    
class Conv_Bn_Act(nn.Module):
    def __init__(self, in_, out_, kernel_size,
                stride=1, groups=1, bias=True,
                eps=1e-5, momentum=0.1, act_layer=nn.SiLU, mode=''):
        super(Conv_Bn_Act, self).__init__()

        if mode == 'tf':
            self.conv = SamePadConv2d(in_, out_, kernel_size, stride, groups=groups, bias=bias)  
        else: 
            self.conv = conv2d(in_, out_, kernel_size, stride, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_, eps, momentum)
        self.act = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SamePadConv2d(nn.Conv2d):
    """
    Conv with TF padding='same'
    https://github.com/pytorch/pytorch/issues/3867#issuecomment-349279036
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_mode="zeros"):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias, padding_mode)

    def get_pad_odd(self, in_, weight, stride, dilation):
        effective_filter_size_rows = (weight - 1) * dilation + 1
        out_rows = (in_ + stride - 1) // stride
        padding_needed = max(0, (out_rows - 1) * stride + effective_filter_size_rows - in_)
        padding_rows = max(0, (out_rows - 1) * stride + (weight - 1) * dilation + 1 - in_)
        rows_odd = (padding_rows % 2 != 0)
        return padding_rows, rows_odd

    def forward(self, x):
        padding_rows, rows_odd = self.get_pad_odd(x.shape[2], self.weight.shape[2], self.stride[0], self.dilation[0])
        padding_cols, cols_odd = self.get_pad_odd(x.shape[3], self.weight.shape[3], self.stride[1], self.dilation[1])

        if rows_odd or cols_odd:
            x = F.pad(x, [0, int(cols_odd), 0, int(rows_odd)])

        return F.conv2d(x, self.weight, self.bias, self.stride,
                        padding=(padding_rows // 2, padding_cols // 2),
                        dilation=self.dilation, groups=self.groups)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class SEModule(nn.Module):
    def __init__(self, in_, squeeze_ch, act_layer=nn.SiLU):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_, squeeze_ch, kernel_size=1, stride=1, padding=0, bias=True),
            act_layer(inplace=True),
            nn.Conv2d(squeeze_ch, in_, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)

class DropConnect(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = 1.0 - ratio

    def forward(self, x):
        if not self.training:
            return x

        random_tensor = self.ratio
        random_tensor += torch.rand([x.shape[0], 1, 1, 1], dtype=torch.float, device=x.device)
        # random_tensor.requires_grad_(False)
        return x / self.ratio * random_tensor.floor()

class ScaledWSConv2d(nn.Conv2d):
    """2D Conv layer with Scaled Weight Standardization."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, dilation=1,
                    groups=1, gamma=1.0, bias=True, gain_init=1.0, eps=1e-5):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, 
        dilation=dilation, groups=groups, bias=bias)
        self.gain = nn.Parameter(torch.full((self.out_channels, 1, 1, 1), gain_init))
        self.eps = eps
        self.scale = gamma * self.weight[0].numel() ** -0.5



    def get_weight(self):
        std, mean = torch.std_mean(self.weight, dim=[1,2,3], keepdim=True, unbiased=False)
        weight = self.scale * (self.weight - mean) / (std + self.eps)
        return self.gain * weight

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, 
                self.dilation, self.groups)

class ECAModule(nn.Module):
    def __init__(self, channels=None, kernel_size=3, gamma=2, beta=1):
        super(ECAModule, self).__init__()
        if channels is not None:
            t = int(abs(math.log(channels, 2) + beta) / gamma)
            kernel_size = max(t if t % 2 else t + 1, 3)
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        y = x.mean((2, 3)).view(x.shape[0], 1, -1)
        y = self.conv(y)
        y = self.gate(y).view(y.shape[0], -1, 1, 1)
        return x * y.expand_as(x)


def get_attn(attn_type, channels, **kwards):
    if attn_type == 'se':
        module_cls = SEModule
    elif attn_type == 'eca':
        module_cls = ECAModule
    else:
        return None

    if module_cls is not None:
        return module_cls(channels, **kwards)
