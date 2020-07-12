from os.path import join

import cv2
import numpy as np
from ocamcamera import OcamCamera
from scipy.spatial.transform import Rotation as Rot
from torch.utils.data import Dataset


class OmniStereoDataset(Dataset):
    """Omnidirectional Stereo Dataset.
    http://cvlab.hanyang.ac.kr/project/omnistereo/
    """

    def __init__(self, root_dir, filename_txt, transform=None, fov=220):
        self.root_dir = root_dir
        self.transform = transform

        # load filenames
        with open(filename_txt) as f:
            data = f.read()
        self.filenames = data.strip().split('\n')

        # folder name
        self.cam_list = ['cam1', 'cam2', 'cam3', 'cam4']
        self.depth_folder = 'depth_train_640'

        # load ocam calibration data and generate valid image
        self.ocams = {}
        self.valids = {}
        for cam in self.cam_list:
            ocam_file = join(root_dir, f'o{cam}.txt')
            self.ocams[cam] = OcamCamera(ocam_file, fov, show_flag=False)
            self.valids[cam] = self.ocams[cam].valid_area()

        # load poses T cam <- world
        poses_cw = load_poses(join(root_dir, 'poses.txt'))
        # poses world <- T cam
        self.pose_dict = {}
        for i in range(4):
            cam = f'cam{i + 1}'
            self.pose_dict[cam] = np.linalg.inv(poses_cw[i])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        sample = {}

        filename = self.filenames[idx]
        # load images
        for i, cam in enumerate(self.cam_list):
            img_path = join(self.root_dir, cam, filename)
            # sample[cam] = load_image(img_path, valid=self.valids[cam])
            sample[cam] = load_image(img_path)
        # load inverse depth
        depth_path = join(self.root_dir, self.depth_folder, filename)
        sample['idepth'] = load_invdepth(depth_path)

        if self.transform:
            sample = self.transform(sample)

        return sample


def load_invdepth(filename, min_depth=55):
    '''
    min_depth in [cm]
    '''
    invd_value = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
    invdepth = (invd_value / 100.0) / (min_depth * 655) + np.finfo(np.float32).eps
    invdepth *= 100  # unit conversion from cm to m
    return invdepth


def load_image(filename, gray=False, valid=None, out_of_fov_value=None):
    img = cv2.imread(filename)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if not valid is None:
        if out_of_fov_value is None:
            out_of_fov_value = 0
        img[valid == 0] = out_of_fov_value
    return img


def load_poses(pose_file):
    """Calculate pose T cam <- world \in SE(3)"""
    Tcw = []
    with open(pose_file) as f:
        data = f.readlines()

    for it in data:
        it = list(map(float, it.split()))
        T = np.eye(4)  # T world <- cam
        angle = it[:3]
        R = Rot.from_rotvec(angle).as_matrix()
        T[:3, :3] = R
        T[:3, 3] = it[3:]
        T[:3, 3] /= 100  # from cm to m
        Tcw.append(np.linalg.inv(T))
    return Tcw
