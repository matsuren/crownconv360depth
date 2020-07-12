import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .costvolume_regularization import CostRegularization
from .feature_extraction import FeatureExtraction
from .icospherical_sweeping import IcoSphericalSweeping


class IcoSweepNet(nn.Module):
    def __init__(self, root_dir, ndisp, level, fov=220):
        super(IcoSweepNet, self).__init__()
        self.cam_list = ['cam1', 'cam2', 'cam3', 'cam4']
        self.ndisp = ndisp
        self.min_depth = 0.55
        # sweeping sphere inverse distance
        self.inv_depths = np.linspace(0, 1 / self.min_depth, self.ndisp) + np.finfo(np.float32).eps
        self.depths = 1. / self.inv_depths

        # module
        self.level = level
        self.feat_ch = 32
        self.idepth_level = level - 2
        self.feature_extraction = FeatureExtraction(self.feat_ch)
        self.cost_regularization = CostRegularization(self.feat_ch * len(self.cam_list))
        self.sweep = IcoSphericalSweeping(root_dir, self.idepth_level, fov=fov)

    def forward(self, batch):
        # randomly permuted concatenate
        cam_idxs = list(range(len(self.cam_list)))
        if self.training:
            random.shuffle(cam_idxs)

        batch_size = batch['cam1'].shape[0]
        device = batch['cam1'].device
        dtype = batch['cam1'].dtype

        # feature extraction
        vertex_features = []
        for cam_idx in cam_idxs:
            key = self.cam_list[cam_idx]
            x = self.feature_extraction(batch[key])
            vertex_features.append(x)

        # initialize cost
        vertex_num = 2 + 10 * 4 ** self.idepth_level
        feat_ch = self.feat_ch
        cost_size = (batch_size, feat_ch * len(self.cam_list), self.ndisp, vertex_num)
        costs = torch.zeros(cost_size, device=device, dtype=dtype)

        # construct cost volume
        for cost_cnt, vertex_feat in enumerate(vertex_features):
            for d_idx, depth in enumerate(self.depths):
                vertex_ids, weight, is_inside = self.sweep.get_grid(cam_idxs[cost_cnt], depth)
                warp_feat = triangle_interpolation(vertex_feat, vertex_ids, weight)
                costs[:, feat_ch * cost_cnt:feat_ch * (cost_cnt + 1), d_idx, is_inside] = warp_feat

        # cost regularization
        vertex_out = self.cost_regularization(costs)
        vertex_pred = DisparityRegression(self.ndisp)(vertex_out)

        return vertex_pred


def triangle_interpolation(vertex_feat, vertex_ids, weight):
    assert weight[0].min() > 0
    device = vertex_feat.device
    weight_0 = weight[0].to(device)
    weight_1 = weight[1].to(device)
    weight_2 = weight[2].to(device)

    feat0 = vertex_feat[:, :, vertex_ids[0]] * weight_0
    feat1 = vertex_feat[:, :, vertex_ids[1]] * weight_1
    feat2 = vertex_feat[:, :, vertex_ids[2]] * weight_2

    final_feat = feat0 + feat1 + feat2

    return final_feat


class DisparityRegression(nn.Module):
    """ Soft argmax disparity regression proposed in [1]

    Parameters
    ----------
    ndisp : int
        Number of disparity,
    min_disp : int
        Minimum index of disparity. Usually disparity starts from zero.

    References
    ----------
    [1] A. Kendall et al., “End-to-end learning of geometry and context for deep stereo regression”
    """

    def __init__(self, ndisp, min_disp=0):
        super(DisparityRegression, self).__init__()
        self.disp = torch.Tensor(np.reshape(np.array(range(min_disp, ndisp + min_disp)), [1, ndisp, 1, 1]))

    def forward(self, x):
        x = F.softmax(x, dim=1)
        self.disp = self.disp.to(x.device)
        self.disp.requires_grad_(False)
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1)
        return out
