import math
from functools import lru_cache

import numpy as np
import torch
from numpy.linalg import norm

from utils.geometry_helper import get_unfold_imgcoord, get_unfold_imgcoord_row, get_icosahedron


@lru_cache()
def get_count_for_integration(level):
    vertex_num = 2 + 10 * 4 ** level
    imgcoord = get_unfold_imgcoord(level, drop_NE=False)
    imgcoord_row = get_unfold_imgcoord_row(level, drop_NE=False)
    vertex_count = np.zeros(vertex_num, dtype=np.int)
    for key in range(5):
        vertex_count[imgcoord[key].ravel()] += 1
        vertex_count[imgcoord_row[key].ravel()] += 1
    return vertex_count


@lru_cache()
def calculate_weight(level=0):
    """ calculate weight alpha
    """
    img_coord = get_unfold_imgcoord(level, False)
    vertices, _ = get_icosahedron(level)
    key = 0
    vi_idx = img_coord[key][1:, 1:]
    vn1_idx = img_coord[key][1:, :-1]
    vn6_idx = img_coord[key][:-1, 1:]
    vi = vertices[vi_idx]
    vn1 = vertices[vn1_idx]
    vn6 = vertices[vn6_idx]

    # vector from vi to north pole
    north_pole = np.array([0, -1, 0])
    to_north_pole = north_pole - vi

    # unit vector from vi to neighbor
    vn1vi = vn1 - vi
    vn1vi /= norm(vn1vi, axis=-1, keepdims=True)
    vn6vi = vn6 - vi
    vn6vi /= norm(vn6vi, axis=-1, keepdims=True)

    # face normal
    face_n = np.cross(vn1vi, vn6vi)
    face_n /= norm(face_n, axis=-1, keepdims=True)

    # to north pole on tangent plane
    proj_vec = np.sum(to_north_pole * face_n, axis=-1, keepdims=True) * face_n
    np_tangent_plane = to_north_pole - proj_vec
    np_tangent_plane /= norm(np_tangent_plane, axis=-1, keepdims=True)

    # calculate cost
    psi = np.arccos(np.sum(vn1vi * np_tangent_plane, axis=-1))
    tmp_vals = np.sum(vn6vi * np_tangent_plane, axis=-1)
    tmp_vals[0, 0] = np.clip(tmp_vals[0, 0], -1, 1)  # make sure value is between -1 and 1
    phi = np.arccos(tmp_vals)
    base_weight = phi / (psi + phi)

    weight = np.concatenate((np.flip(base_weight[-1:]), base_weight), axis=0)
    weight = np.concatenate((np.flip(weight[:, -1:]), weight), axis=1)

    # top_weight = base_weight[-1][::-1]
    # left_weight = base_weight[:, -1][::-1]
    # weight = np.zeros(img_coord[0].shape)
    # weight[0, 0] = 0.5 # index 0 weight
    # weight[1:, 1:] = base_weight
    # weight[0, 1:] = top_weight
    # weight[1:, 0] = left_weight

    # to row weight
    h, w = weight.shape
    weight_row = 1 - np.concatenate((weight[:w], weight[w - 1:, 1:]), axis=1)

    return weight, weight_row


def batch_to_list(torch_arr, num=5):
    total_batch = torch_arr.size(0)
    b = total_batch // num
    list_arr = [torch_arr[i * b:(i + 1) * b] for i in range(num)]
    return list_arr


def unfold_feat_to_vertex_feat(feat, feat_row, add_weight=False):
    if not isinstance(feat, list):
        feat = batch_to_list(feat)
    if not isinstance(feat_row, list):
        feat_row = batch_to_list(feat_row)

    h, w = feat[0].shape[-2:]
    device = feat[0].device
    level = int(math.log2(w - 1))
    vertex_num = 2 + 10 * 4 ** level
    imgcoord = get_unfold_imgcoord(level, drop_NE=False)
    imgcoord_row = get_unfold_imgcoord_row(level, drop_NE=False)

    if add_weight:
        weight, weight_row = calculate_weight(level)
        weight_torch = torch.from_numpy(weight).to(device)
        weight_row_torch = torch.from_numpy(weight_row).to(device)

        feat = [weight_torch * feat[i] for i in range(5)]
        feat_row = [weight_row_torch * feat_row[i] for i in range(5)]

    #
    if feat[0].ndim == 4:
        b, c = feat[0].shape[:2]
        vertex_feat = torch.zeros((b, c, vertex_num), device=device)
        for key in range(5):
            vertex_feat[..., imgcoord[key].ravel()] += feat[key].reshape(b, c, -1)
            vertex_feat[..., imgcoord_row[key].ravel()] += feat_row[key].reshape(b, c, -1)
    elif feat[0].ndim == 5:
        b, c, d = feat[0].shape[:3]
        vertex_feat = torch.zeros((b, c, d, vertex_num), device=device)
        for key in range(5):
            vertex_feat[..., imgcoord[key].ravel()] += feat[key].reshape(b, c, d, -1)
            vertex_feat[..., imgcoord_row[key].ravel()] += feat_row[key].reshape(b, c, d, -1)
    else:
        raise ValueError("feat[0].ndim must be 4 or 5")

    # get count
    count = get_count_for_integration(level)
    factor = torch.from_numpy(count).to(device)
    vertex_feat /= factor

    return vertex_feat


def vertex_feat_to_unfold_feat(vertex_feat, return_list=True):
    if vertex_feat.size(-2) != 1:
        # size must be b x c x 1 x vertex_num or
        #              b x c x d x 1 x vertex_num
        vertex_feat.unsqueeze_(-2)
    vertex_num = vertex_feat.shape[-1]
    level = int(math.log((vertex_num - 2) // 10, 4))
    imgcoord = get_unfold_imgcoord(level, drop_NE=False)
    imgcoord_row = get_unfold_imgcoord_row(level, drop_NE=False)

    device = vertex_feat.device

    feat = [index_select(vertex_feat, torch.from_numpy(imgcoord[i]).to(device)) for i in range(5)]
    feat_row = [index_select(vertex_feat, torch.from_numpy(imgcoord_row[i]).to(device)) for i in range(5)]

    if return_list:
        return feat, feat_row
    else:
        return torch.cat(feat), torch.cat(feat_row)


def col_row_integration(feat, feat_row, return_list=False):
    vertex_feat = unfold_feat_to_vertex_feat(feat, feat_row, add_weight=False)
    integ_feat, integ_feat_row = vertex_feat_to_unfold_feat(vertex_feat, return_list)

    return integ_feat, integ_feat_row


def index_select(ico_img, imgcoord):
    """
    ico_img:
        3d tensor (b x c x vertex_num) or
        4d tensor (b x c x 1 x vertex_num) or
        5d tensor (b x c x d x 1 x vertex_num)

    imgcoord: numpy array
    """
    if ico_img.ndim is 4 or ico_img.ndim is 3:
        return _index_select_img(ico_img, imgcoord)
    elif ico_img.ndim is 5:
        return _index_select_volume(ico_img, imgcoord)
    else:
        raise ValueError("Wrong shape of tensor")


def _index_select_img(ico_img, imgcoord):
    """
    ico_img: tensor (b x c x 1 x vertex_num)
    imgcoord: numpy array
    """
    b, c = ico_img.shape[:2]
    device = ico_img.device
    h, w = imgcoord.shape[-2:]

    img = torch.index_select(ico_img, -1, imgcoord.reshape(-1))
    return img.view(b, c, h, w)


def _index_select_volume(ico_img, imgcoord):
    """
    ico_img: tensor (b x c x d x 1 x vertex_num)
    imgcoord: numpy array
    """
    b, c, d = ico_img.shape[:3]
    device = ico_img.device
    h, w = imgcoord.shape[-2:]

    img = torch.index_select(ico_img, -1, imgcoord.reshape(-1))
    return img.view(b, c, d, h, w)
