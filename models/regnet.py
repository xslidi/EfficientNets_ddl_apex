# Modified from https://github.com/facebookresearch/pycls/blob/master/pycls/models/anynet.py

import numpy as np
import torch
from torch.nn import Module
from torch import nn
from models.layers import SamePadConv2d, conv_bn_act, SEModule, conv2d

class Anyhead(Module):
    """AnyNet head: AvgPool, 1x1."""

    def __init__(self, w_in, num_classes):
        super(Anyhead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(w_in, num_classes, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BottleneckTransform(Module):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, w_in, w_out, stride, params):
        super(BottleneckTransform, self).__init__()
        w_b = int(round(w_out * params['bot_mul']))
        w_se = int(round(w_in * params['se_r']))
        groups = w_b // params['group_w']
        self.a = conv_bn_act(w_in, w_b, 1, eps=1e-5, momentum=0.1, act_layer=nn.ReLU, mode=None, bias=False)
        self.b = conv_bn_act(w_b, w_b, 3, stride=stride, groups=groups, eps=1e-5, momentum=0.1, act_layer=nn.ReLU, mode=None, bias=False)
        self.se = SEModule(w_b, w_se, act_layer=nn.ReLU) if w_se else None
        self.c = conv2d(w_b, w_out, 1)
        self.c_bn = nn.BatchNorm2d(w_out)

    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        x = self.se(x)
        x = self.c(x)
        x = self.c_bn(x)

        return x

class ResBottleneckBlock(Module):
    """Residual bottleneck block: x + f(x), f = bottleneck transform."""

    def __init__(self, w_in, w_out, stride, params):
        super(ResBottleneckBlock, self).__init__()
        self.proj, self.bn = None, None
        if (w_in != w_out) or stride != 1:
            self.proj = conv2d(w_in, w_out, 1, stride=stride)
            self.bn = nn.BatchNorm2d(w_out) 
        
        self.f = BottleneckTransform(w_in, w_out, stride, params)
        self.af = nn.ReLU(inplace=True)

    def forward(self, x):
        x_p = self.bn(self.proj(x)) if self.proj else x
        return self.af(self.f(x)) + x_p

class SimpleStem(Module):
    """Simple stem for ImageNet: 3x3, BN, AF."""

    def __init__(self, w_in, w_out):
        super(SimpleStem, self).__init__()
        self.conv = conv2d(w_in, w_out, 3, stride=2)
        self.bn = nn.BatchNorm2d(w_out)
        self.af = nn.ReLU(inplace=True)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)

        return x


class Anystage(Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, d, params):
        super(Anystage, self).__init__()
        for i in range(d):
            block = ResBottleneckBlock(w_in, w_out, stride, params)
            self.add_module("b{}".format(i + 1), block)
            stride, w_in = 1, w_out

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x

class AnyNet(Module):
    """AnyNet model."""

    @staticmethod
    def get_params():
        return {
            "stem_w": cfg.ANYNET.STEM_W,
            "depths": cfg.ANYNET.DEPTHS,
            "widths": cfg.ANYNET.WIDTHS,
            "strides": cfg.ANYNET.STRIDES,
            "bot_muls": cfg.ANYNET.BOT_MULS if cfg.ANYNET.BOT_MULS else nones,
            "group_ws": cfg.ANYNET.GROUP_WS if cfg.ANYNET.GROUP_WS else nones,
            "se_r": cfg.ANYNET.SE_R if cfg.ANYNET.SE_ON else 0,
            "num_classes": cfg.MODEL.NUM_CLASSES,
        }



    def __init__(self, params=None):
        super(AnyNet, self).__init__()
        self.stem = SimpleStem(3, params['stem_w'])
        prev_w = params['stem_w']
        keys = ["depths", "widths", "strides", "bot_muls", "group_ws"]
        for i, (d, w, s, b, g) in enumerate(zip(*[params[k] for k in keys])): 
            param = {"bot_mul": b, "group_w": g, "se_r": params["se_r"]}
            stage = Anystage(prev_w, w, s, d, param)
            self.add_module("s{}".format(i + 1), stage)
            prev_w = w

        self.head = Anyhead(prev_w, params['num_classes'])
        # self.apply(init_weights)
        
    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per stage widths and depths from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    # Generate continuous per-block ws
    ws_cont = np.arange(d) * w_a + w_0
    # Generate quantized per-block ws
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws_all = w_0 * np.power(w_m, ks)    
    ws_all = np.round(np.divide(ws_all, q)).astype(int) * q
    # Generate per stage ws and ds (assumes ws_all are sorted)
    ws, ds = np.unique(ws_all, return_counts=True)
    # Compute number of actual stages and total possible stages
    num_stages, total_stages = len(ws), ks.max() + 1
    # Convert numpy arrays to lists and return
    ws, ds, ws_all, ws_cont = (x.tolist() for x in (ws, ds, ws_all, ws_cont))
    return ws, ds, num_stages, total_stages, ws_all, ws_cont

def adjust_block_compatibility(ws, bs, gs):
    """Adjusts the compatibility of widths, bottlenecks, and groups."""
    assert len(ws) == len(bs) == len(gs)
    assert all(w > 0 and b > 0 and g > 0 for w, b, g in zip(ws, bs, gs))
    vs = [int(max(1, w * b)) for w, b in zip(ws, bs)]
    # make sure widths not smaller than groups
    gs = [int(min(g, v)) for g, v in zip(gs, vs)]
    ms = [np.lcm(g, b) if b > 1 else g for g, b in zip(gs, bs)]
    # make suer that widths in bottlenecks are common multiple of bs and gs
    vs = [max(m, int(round(v / m) * m)) for v, m in zip(vs, ms)]
    ws = [int(v / b) for v, b in zip(vs, bs)]
    assert all(w * b % g == 0 for w, b, g in zip(ws, bs, gs))
    return ws, bs, gs


class RegNet(AnyNet):
    """RegNet model."""

    def get_params(self, arg):
        """Convert RegNet to AnyNet parameter format."""
        # Generates per stage ws, ds, gs, bs, and ss from RegNet parameters
        w_a, w_0, w_m, d = arg.REGNET_WA, arg.REGNET_W0, arg.REGNET_WM, arg.REGNET_DEPTH
        ws, ds = generate_regnet(w_a, w_0, w_m, d)[0:2]
        ss = [arg.REGNET_STRIDE for _ in ws]
        bs = [arg.REGNET_BOT_MUL for _ in ws]
        gs = [arg.REGNET_GROUP_W for _ in ws]
        ws, bs, gs = adjust_block_compatibility(ws, bs, gs)
        # Get AnyNet arguments defining the RegNet
        return {
            "stem_w": arg.REGNET_STEM_W,
            "depths": ds,
            "widths": ws,
            "strides": ss,
            "bot_muls": bs,
            "group_ws": gs,
            "se_r": arg.se_r,
            "num_classes": arg.num_classes,
        }

    def __init__(self, arg):
        params = self.get_params(arg)
        super(RegNet, self).__init__(params)


