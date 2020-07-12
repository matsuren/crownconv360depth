from functools import lru_cache

import cv2
import numpy as np
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from utils.geometry_helper import get_icosahedron, get_unfold_imgcoord, weight_for_triangle_interpolation


class FisheyeToIcoDataset(Dataset):
    """Fisheye to unfolded icosahedron dataset.

    """

    def __init__(self, root_dataset, ocam_dict, pose_dict, level, reduction=2, transform=None,
                 border_value=[256 // 2, 256 // 2, 256 // 2]):
        self.root_dataset = root_dataset
        self.ocam_dict = ocam_dict
        self.pose_dict = pose_dict
        self.level = level
        self.transform = transform

        # fisheye to ico
        self.vertices, self.faces = get_icosahedron(level)
        self.fisheye_mapxy_pad = {}
        for key in self.ocam_dict:
            self.fisheye_mapxy_pad[key] = mapxy_fisheye_to_ico(self.ocam_dict[key], self.pose_dict[key], self.level)

        # idepth to ico
        self.idepth_level = level - reduction
        self.idepth_mapxy_pad = self._mapxy_idepth_to_ico(self.idepth_level)
        self.idepth_coord = get_unfold_imgcoord(self.idepth_level, drop_NE=False)

        # out of fov value
        self.border_value = border_value

    def __len__(self):
        if self.root_dataset is None:
            return 0
        else:
            return len(self.root_dataset)

    def _mapxy_idepth_to_ico(self, level):
        v, f = get_icosahedron(level)
        mapy, mapx = uv2img_idx(xyz2uv(v), h=320, w=640)

        # reshape to prevent shrt_max error in remap
        # https://answers.opencv.org/question/203798/shrt_max-in-cv2remap/
        len_v = len(v) ** 0.5
        tmp_h = int(len_v) + 1
        pad_w = tmp_h * tmp_h - mapx.shape[0]
        mapx = np.pad(mapx, (0, pad_w), 'constant', constant_values=-1)
        mapy = np.pad(mapy, (0, pad_w), 'constant', constant_values=-1)
        mapx = mapx.reshape(tmp_h, tmp_h).astype(np.float32)
        mapy = mapy.reshape(tmp_h, tmp_h).astype(np.float32)
        return mapx, mapy, pad_w

    def _remap_idepth_to_ico(self, idepth, mapx, mapy, pad_w):
        ico_idepth = cv2.remap(idepth, mapx, mapy, cv2.INTER_NEAREST).flatten()[:-pad_w]
        return ico_idepth

    def convert_to_unfold(self, data_dict, border_value=0):
        """Convert fisheye images and idepth to unfolded icosahedron.
        Image's key must start with 'cam' (e.g., cam1, cam2,,,)
        Inverse depth mush start with 'idepth'.
        """
        results = {}
        for key, img in data_dict.items():
            if key.startswith('cam'):
                mapx, mapy, pad_w = self.fisheye_mapxy_pad[key]
                #  mapx, mapy, pad_w = mapxy_fisheye_to_ico(self.ocam_dict[key], self.level)
                ico_img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderValue=border_value)
                ico_img = ico_img.reshape(-1, 3)[:-pad_w]
                results[key] = ico_img
            elif key.startswith('idepth'):
                ico_idepth = self._remap_idepth_to_ico(img, *self.idepth_mapxy_pad)
                results[key] = ico_idepth
        return results

    def __getitem__(self, idx):
        sample = self.root_dataset[idx]

        # unfolded images
        sample = self.convert_to_unfold(sample, border_value=self.border_value)

        if self.transform:
            sample = self.transform(sample)

        return sample


def mapxy_fisheye_to_ico(ocam, T_wc=np.eye(4), level=0):
    """Map to project a fisheye image to icosahedron surface

    Parameters
    ----------
    ocam : OCamCamera (https://github.com/matsuren/ocamcalib_undistort)
        OcamCalib camera class object
    level : int
        icosahedron resolution level

    Examples
    --------
    >>> mapx, mapy, pad_w = mapxy_fisheye_to_ico(ocam, level)
    >>> ico_img = cv2.remap(img, mapx, mapy,cv2.INTER_LINEAR)
    >>> ico_img = ico_img.reshape(-1, 3)[:-pad_w]
    """
    # get vertices
    v, f = get_icosahedron(level)
    # rotate vertices
    T_cw = np.linalg.inv(T_wc)
    rot_v = T_cw[:3,:3].dot(v.T)
    mapx, mapy = ocam.world2cam(rot_v)

    # reshape to prevent shrt_max error in remap
    # https://answers.opencv.org/question/203798/shrt_max-in-cv2remap/
    # from scipy.ndimage.interpolation import map_coordinates is slow
    # so I chose to use opencv remap
    len_v = len(v) ** 0.5
    tmp_h = int(len_v) + 1
    pad_w = tmp_h * tmp_h - mapx.shape[0]
    mapx = np.pad(mapx, (0, pad_w), 'constant', constant_values=-1)
    mapy = np.pad(mapy, (0, pad_w), 'constant', constant_values=-1)
    mapx = mapx.reshape(tmp_h, tmp_h)
    mapy = mapy.reshape(tmp_h, tmp_h)

    return mapx, mapy, pad_w


def genuv(h, w):
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = (u + 0.5) * 2 * np.pi / w - np.pi
    v = (v + 0.5) * np.pi / h - np.pi / 2
    uv = np.stack([u, v], axis=-1)
    return uv


def uv2xyz(uv):
    sin_u = np.sin(uv[..., 0])
    cos_u = np.cos(uv[..., 0])
    sin_v = np.sin(uv[..., 1])
    cos_v = np.cos(uv[..., 1])
    return np.stack([
        cos_v * sin_u,
        sin_v,
        cos_v * cos_u,
    ], axis=-1)


def xyz2uv(xyz):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    u = np.arctan2(x, z)
    c = np.sqrt(x * x + z * z)
    v = np.arctan2(y, c)
    return np.stack([u, v], axis=-1)


def uv2img_idx(uv, h, w):
    delta_w = 2 * np.pi / w
    delta_h = np.pi / h
    x = uv[..., 0] / delta_w + w / 2 - 0.5
    y = uv[..., 1] / delta_h + h / 2 - 0.5
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    return np.stack([y, x], axis=0)


@lru_cache()
def ico_to_erp_idx_weight(level, h=300, w=600):
    xyz = uv2xyz(genuv(h, w))
    pts_c = xyz.reshape(-1, 3)

    v, f = get_icosahedron(level)
    kdtree = cKDTree(v)
    _, tmp_indices = kdtree.query(pts_c, k=4)
    (v0_idx, v1_idx, v2_idx), (weight_0, weight_1, weight_2) = weight_for_triangle_interpolation(v, tmp_indices, pts_c)
    return (v0_idx, v1_idx, v2_idx), (weight_0, weight_1, weight_2)


def ico_to_erp(ico_img, level, h=300, w=600):
    ico_img = ico_img.squeeze()
    vertex_ids, weights = ico_to_erp_idx_weight(level, h, w)
    (v0_idx, v1_idx, v2_idx), (weight_0, weight_1, weight_2) = vertex_ids, weights
    erp_img = ico_img[v0_idx] * weight_0 + ico_img[v1_idx] * weight_1 + ico_img[v2_idx] * weight_2
    erp_img = erp_img.reshape(h, w)
    return erp_img
