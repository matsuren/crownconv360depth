import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import init
from torch.nn.modules.utils import _pair, _triple

padding_mode = 'replicate'  # ['replicate', 'zero']


class CrownConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        """

        """
        super(CrownConv2d, self).__init__()
        if stride != 1 and stride != 2:
            raise ValueError("stride must be 1 or 2")

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = _pair(stride)
        # weight
        weight_trainable = torch.Tensor(out_channels, in_channels, 7)
        nn.init.kaiming_uniform_(weight_trainable, mode='fan_in', nonlinearity='relu')
        self.weight = Parameter(weight_trainable)  # adding zero

        #         self._w1_index = [
        #             [-1, 5, 4],
        #             [0, 6, 3],
        #             [1, 2, -1]
        #         ]
        #         self._w2_index = [
        #             [-1, 0, 5],
        #             [1, 6, 4],
        #             [2, 3, -1]
        #         ]

        _w1_index = [
            5, 4,
            0, 6, 3,
            1, 2
        ]
        _w2_index = [
            0, 5,
            1, 6, 4,
            2, 3
        ]

        self.register_buffer('w1_index', torch.tensor(_w1_index))
        self.register_buffer('w2_index', torch.tensor(_w2_index))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            fan_in = in_channels * 9
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size=(hexagonal)'
             ', stride={stride}')
        return s.format(**self.__dict__)

    def get_weight(self):
        weight1 = F.pad(torch.index_select(self.weight, -1, self.w1_index), (1, 1))
        weight2 = F.pad(torch.index_select(self.weight, -1, self.w2_index), (1, 1))
        out_ch, in_ch = weight1.shape[:2]
        return weight1.view(out_ch, in_ch, 3, 3), weight2.view(out_ch, in_ch, 3, 3)

    def forward(self, feat_tuple):
        """
        (b*5, c, h, w) Tensor
        """
        feat, feat_row = feat_tuple
        weight1, weight2 = self.get_weight()

        if padding_mode == "zero":
            feat = F.conv2d(feat, weight1, self.bias, self.stride, 1)
            feat_row = F.conv2d(feat_row, weight2, self.bias, self.stride, 1)
        elif padding_mode == "replicate":
            pad = [1, 1, 1, 1]
            feat = F.conv2d(F.pad(feat, pad, mode='replicate'), weight1, self.bias, self.stride, 0)
            feat_row = F.conv2d(F.pad(feat_row, pad, mode='replicate'), weight2, self.bias, self.stride, 0)
        else:
            raise ValueError('wrong padding_mode')
        return feat, feat_row


class CrownConv2dBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bias=False, relu=True):
        super(CrownConv2dBNRelu, self).__init__()
        self.conv = CrownConv2d(in_channels, out_channels, stride, bias)
        self.bn = CrownBatchNorm2d(out_channels)
        self.relu = nn.ReLU() if relu else None
        self.stride = stride

    def forward(self, feat_tuple):
        feat_tuple = self.conv(feat_tuple)
        feat, feat_row = self.bn(feat_tuple)

        if self.relu is not None:
            feat = self.relu(feat)
            feat_row = self.relu(feat_row)

        return feat, feat_row


class CrownBatchNorm2d(nn.BatchNorm2d):
    def forward(self, x):
        feat, feat_row = x

        b, c, h, w = feat.shape
        # batch => len(x)*batch
        out_cat = super(CrownBatchNorm2d, self).forward(torch.cat((feat, feat_row.transpose(-1, -2)), dim=0))
        out = [out_cat[b * i:b * (i + 1)] for i in range(2)]  # => list of b x c x h x w

        return out[0], out[1].transpose(-1, -2)


class CrownConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        """

        """
        super(CrownConv3d, self).__init__()
        if stride != 1 and stride != 2:
            raise ValueError("stride must be 1 or 2")

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = _triple(stride)
        # weight
        weight_trainable = torch.Tensor(out_channels, in_channels, 7 * 3)
        nn.init.kaiming_uniform_(weight_trainable, mode='fan_in', nonlinearity='relu')
        self.weight = Parameter(weight_trainable)  # adding zero

        # _2dw1_index = [
        #     -1, 5, 4,0, 6, 3,1, 2, -1
        # ]
        # _2dw2_index = [
        #    -1, 0, 5, 1, 6, 4, 2, 3, -1
        # ]
        # weight = []
        # for i in range(3):
        #     for it in _2dw2_index:
        #         if it != -1:
        #             weight.append(it+i*7)
        #         else:
        #             weight.append(21)
        # print(weight)

        _w1_index = [
            21, 5, 4, 0, 6, 3, 1, 2, 21,
            21, 12, 11, 7, 13, 10, 8, 9, 21,
            21, 19, 18, 14, 20, 17, 15, 16, 21]
        _w2_index = [
            21, 0, 5, 1, 6, 4, 2, 3, 21,
            21, 7, 12, 8, 13, 11, 9, 10, 21,
            21, 14, 19, 15, 20, 18, 16, 17, 21]

        self.register_buffer('w1_index', torch.tensor(_w1_index))
        self.register_buffer('w2_index', torch.tensor(_w2_index))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            fan_in = in_channels * 9
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size=(hexagonal)'
             ', stride={stride}')
        return s.format(**self.__dict__)

    def get_weight(self):
        weight_wt_pad = F.pad(self.weight, (0, 1))
        weight1 = torch.index_select(weight_wt_pad, -1, self.w1_index)
        weight2 = torch.index_select(weight_wt_pad, -1, self.w2_index)
        out_ch, in_ch = weight1.shape[:2]
        return weight1.view(out_ch, in_ch, 3, 3, 3), weight2.view(out_ch, in_ch, 3, 3, 3)

    def forward(self, feat_tuple):
        """
        feat_tuple = (feat, feat_row)
        feat: Tensor
            (b*5, c, d, h, w)
        feat_row: Tensor
            (b*5, c, d, h_row, w_row)
        """
        feat, feat_row = feat_tuple
        weight1, weight2 = self.get_weight()

        if padding_mode == "zero":
            feat = F.conv3d(feat, weight1, self.bias, self.stride, 1)
            feat_row = F.conv3d(feat_row, weight2, self.bias, self.stride, 1)
        elif padding_mode == "replicate":
            pad = [1, 1, 1, 1, 1, 1]
            feat = F.conv3d(F.pad(feat, pad, mode='replicate'), weight1, self.bias, self.stride, 0)
            feat_row = F.conv3d(F.pad(feat_row, pad, mode='replicate'), weight2, self.bias, self.stride, 0)
        else:
            raise ValueError('wrong padding_mode')
        return feat, feat_row


class CrownConv3dBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bias=False, relu=True):
        super(CrownConv3dBNRelu, self).__init__()
        self.conv = CrownConv3d(in_channels, out_channels, stride, bias)
        self.bn = CrownBatchNorm3d(out_channels)
        self.relu = nn.ReLU() if relu else None

    def forward(self, feat_tuple):
        feat_tuple = self.conv(feat_tuple)
        feat, feat_row = self.bn(feat_tuple)

        if self.relu is not None:
            feat = self.relu(feat)
            feat_row = self.relu(feat_row)

        return feat, feat_row


class CrownBatchNorm3d(nn.BatchNorm3d):
    def forward(self, x):
        feat, feat_row = x

        b, c, d, h, w = feat.shape
        # batch => len(x)*batch
        out_cat = super(CrownBatchNorm3d, self).forward(torch.cat((feat, feat_row.transpose(-1, -2)), dim=0))
        out = [out_cat[b * i:b * (i + 1)] for i in range(2)]  # => list of b x c x d x h x w
        return out[0], out[1].transpose(-1, -2)
