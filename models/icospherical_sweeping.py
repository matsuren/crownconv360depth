from functools import lru_cache
from os.path import join

import numpy as np
import torch
from numpy.linalg import norm
from scipy.spatial import cKDTree

from dataloader.omnistereo_dataset import load_poses
from utils.geometry_helper import get_icosahedron, weight_for_triangle_interpolation


@lru_cache()
def get_KDTree(level):
    v, f = get_icosahedron(level)
    return cKDTree(v)


#     return NearestNeighbors(n_neighbors=3, algorithm='auto').fit(v)


class IcoSphericalSweeping(object):
    def __init__(self, root_dir, level, fov=220):
        """ IcoSphericalSweeping module.

        """
        self.level = level
        self.fov = fov

        # load poses T cam <- world
        self.poses_cw = load_poses(join(root_dir, 'poses.txt'))
        self._Tcw_bytes = [T.tobytes() for T in self.poses_cw]

        # estimate rig center
        center = []
        for T in self.poses_cw:
            camera_position = -T[:3, :3].T.dot(T[:3, 3])
            center.append(camera_position)
        center = np.array(center).mean(axis=0)
        self._center = center
        self._center_bytes = center.tobytes()

    def get_grid(self, idx, depth):
        """ Get grid for torch.nn.functional.grid_sample.
        """
        vertex_ids, weight, is_inside = icospherical_sweep_grid(self._Tcw_bytes[idx], self._center_bytes, depth,
                                                                self.level, self.fov, only_translation=True)
        return vertex_ids, weight, is_inside


def to_torch_grid(indices, d, h, w):
    """
    indices: numpy int array
        array of [z, y, x]
    """
    idx_z, idx_y, idx_x = indices[:, 0], indices[:, 1], indices[:, 2]
    idx_z = 2 * idx_z / (d - 1) - 1
    idx_y = 2 * idx_y / (h - 1) - 1
    idx_x = 2 * idx_x / (w - 1) - 1
    grid = np.stack([idx_x, idx_y, idx_z], axis=-1)
    return torch.from_numpy(grid).float()


@lru_cache(maxsize=None)
def icospherical_sweep_grid(Tcw_bytes, center_bytes, depth, level, fov=220, only_translation=False):
    # numpy array is not hashable so give array as array.tobytes()
    Tcw = np.frombuffer(Tcw_bytes).reshape(4, 4)
    center = np.frombuffer(center_bytes)

    # ------------------------------------
    # find projected points on sphere
    v, f = get_icosahedron(level)
    pts_w = depth * v + center

    # pt_c = T_cw*pt_w
    if only_translation:
        # dataloader deal with rotation
        Twc = np.linalg.inv(Tcw)
        pts_c = pts_w.T - Twc[:3, 3:4]
        cam_dir = Twc[:3, :3].dot([0, 0, 1])
    else:
        pts_c = Tcw[:3, :3].dot(pts_w.T) + Tcw[:3, 3:4]
        cam_dir = [0, 0, 1]
    pts_c = pts_c.T
    pts_c /= norm(pts_c, axis=1, keepdims=True)

    # ------------------------------------
    # extract inside fov
    fov_rad = np.deg2rad(fov / 2)
    cos_vals = pts_c.dot(cam_dir)
    is_inside = np.cos(fov_rad) < cos_vals
    # nearest neighbor search
    kdtree = get_KDTree(level)
    tmp_distances, tmp_indices = kdtree.query(pts_c[is_inside], k=4)
    #     tmp_distances, tmp_indices = kdtree.kneighbors(pts_c[is_inside])

    # ------------------------------------
    # calculate weight
    vertex_ids, (weight_0, weight_1, weight_2) = weight_for_triangle_interpolation(v, tmp_indices, pts_c[is_inside])
    # (v0_idx, v1_idx, v2_idx) = vertex_ids

    weight_0 = torch.from_numpy(weight_0).float()
    weight_1 = torch.from_numpy(weight_1).float()
    weight_2 = torch.from_numpy(weight_2).float()

    return vertex_ids, (weight_0, weight_1, weight_2), is_inside
