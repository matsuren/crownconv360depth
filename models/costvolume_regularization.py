import torch.nn as nn

from models.crown_nn import CrownConv3d, CrownConv3dBNRelu
from utils.feature_integration import vertex_feat_to_unfold_feat, unfold_feat_to_vertex_feat, col_row_integration


class CostRegularization(nn.Module):
    def __init__(self, in_planes, num_layer=5):
        super(CostRegularization, self).__init__()

        self.num_layer = num_layer
        ch = in_planes // 2
        self.dres0 = nn.Sequential(CrownConv3dBNRelu(2 * ch, 2 * ch),
                                   CrownConv3dBNRelu(2 * ch, ch, relu=False))

        # if self.num_layer >= 2:
        self.dres1 = nn.Sequential(CrownConv3dBNRelu(ch, ch),
                                   CrownConv3dBNRelu(ch, ch, relu=False))
        # if self.num_layer >= 3:
        self.dres2 = nn.Sequential(CrownConv3dBNRelu(ch, ch),
                                   CrownConv3dBNRelu(ch, ch, relu=False))
        # if self.num_layer >= 4:
        self.dres3 = nn.Sequential(CrownConv3dBNRelu(ch, ch),
                                   CrownConv3dBNRelu(ch, ch, relu=False))
        # if self.num_layer >= 5:
        self.dres4 = nn.Sequential(CrownConv3dBNRelu(ch, ch),
                                   CrownConv3dBNRelu(ch, ch, relu=False))

        self.classify = nn.Sequential(CrownConv3dBNRelu(ch, ch // 2),
                                      CrownConv3d(ch // 2, 1))

    def forward(self, costs):
        feat_tuple = vertex_feat_to_unfold_feat(costs, return_list=False)
        feat_tuple = self.dres0(feat_tuple)
        feat_tuple = col_row_integration(*feat_tuple, return_list=False)
        while True:
            if self.num_layer == 1:
                break
            feat_tuple = add_tuple(self.dres1(feat_tuple), feat_tuple)
            feat_tuple = col_row_integration(*feat_tuple, return_list=False)
            if self.num_layer == 2:
                break
            feat_tuple = add_tuple(self.dres2(feat_tuple), feat_tuple)
            feat_tuple = col_row_integration(*feat_tuple, return_list=False)
            if self.num_layer == 3:
                break
            feat_tuple = add_tuple(self.dres3(feat_tuple), feat_tuple)
            feat_tuple = col_row_integration(*feat_tuple, return_list=False)
            if self.num_layer == 4:
                break
            feat_tuple = add_tuple(self.dres4(feat_tuple), feat_tuple)
            feat_tuple = col_row_integration(*feat_tuple, return_list=False)
            break

        cost, cost_row = self.classify(feat_tuple)

        cost = cost.squeeze()  # => b x ndips x h x w
        cost_row = cost_row.squeeze()  # => b x ndips x h_row x w_row
        vertex_out = unfold_feat_to_vertex_feat(cost, cost_row, add_weight=False)  # => b x ndips x vertex_num
        vertex_out = vertex_out.unsqueeze(-2)  # => b x ndips x 1 x vertex_num

        return vertex_out


def add_tuple(feat_tuple1, feat_tuple2):
    a = feat_tuple1[0] + feat_tuple2[0]
    b = feat_tuple1[1] + feat_tuple2[1]
    return a, b
