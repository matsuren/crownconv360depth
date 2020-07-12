import torch.nn as nn

from models.crown_nn import CrownConv2d, CrownConv2dBNRelu, CrownBatchNorm2d
from utils.feature_integration import vertex_feat_to_unfold_feat, unfold_feat_to_vertex_feat, col_row_integration


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.convbn1 = CrownConv2dBNRelu(in_planes, planes, stride)
        self.convbn2 = CrownConv2dBNRelu(planes, planes, 1, relu=False)
        self.relu = nn.ReLU()

        self.downsample = None
        self.downsample_bn = None
        if stride == 2:
            if in_planes == planes:
                self.downsample = nn.AvgPool2d(3, 2, 1, count_include_pad=False)
            else:
                self.downsample = nn.Sequential(
                    nn.AvgPool2d(3, 2, 1, count_include_pad=False),
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0),
                )
                self.downsample_bn = CrownBatchNorm2d(planes)

    def forward(self, feat_tuple):
        identity_feat, identity_feat_row = feat_tuple

        feat_tuple = self.convbn1(feat_tuple)
        feat, feat_row = self.convbn2(feat_tuple)

        if self.downsample is not None:
            identity_feat = self.downsample(identity_feat)
            identity_feat_row = self.downsample(identity_feat_row)
            if self.downsample_bn is not None:
                identity_feat, identity_feat_row = self.downsample_bn((identity_feat, identity_feat_row))

        feat += identity_feat
        feat_row += identity_feat_row

        feat = self.relu(feat)
        feat_row = self.relu(feat_row)

        return feat, feat_row


class FeatureExtraction(nn.Module):
    def __init__(self, planes, num_layer1=3, num_layer2=5):
        super(FeatureExtraction, self).__init__()
        ch = planes
        self.conv1 = CrownConv2dBNRelu(3, ch, stride=2)
        self.conv2 = CrownConv2dBNRelu(ch, ch)

        self.layer1 = self._make_layer(BasicBlock, ch, ch, num_layer1)
        self.layer2 = self._make_layer(BasicBlock, ch, ch, num_layer2, downsample=True)
        #                 self.layer3 = self._make_layer(BasicBlock, ch, ch, 3)
        self.lastconv = CrownConv2d(ch, ch)

    def _make_layer(self, block, in_planes, planes, blocks, downsample=False):
        stride = 2 if downsample else 1
        layers = [block(in_planes, planes, stride)]
        for i in range(1, blocks):
            layers.append(block(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        feat_tuple = vertex_feat_to_unfold_feat(x, return_list=False)
        feat_tuple = self.conv1(feat_tuple)
        #         print(feat_tuple[0].shape)
        #         feat_tuple = col_row_integration(*feat_tuple, return_list=False)
        feat_tuple = self.conv2(feat_tuple)
        #         print(feat_tuple[0].shape)

        feat_tuple = col_row_integration(*feat_tuple, return_list=False)
        feat_tuple = self.layer1(feat_tuple)
        #         print(feat_tuple[0].shape)

        feat_tuple = col_row_integration(*feat_tuple, return_list=False)
        feat_tuple = self.layer2(feat_tuple)
        #         feat_tuple = col_row_integration(*feat_tuple, return_list=False)
        #         out = self.layer3(out)
        feat_tuple = col_row_integration(*feat_tuple, return_list=False)
        feat_tuple = self.lastconv(feat_tuple)
        vertex_feat = unfold_feat_to_vertex_feat(*feat_tuple, add_weight=False)
        return vertex_feat
