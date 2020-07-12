import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

from utils.geometry_helper import get_icosahedron, get_unfold_imgcoord
from utils.projection_helper import img2ERP, erp2sphere


# --------------------
# Dataset
class OmniMNISTDataset(Dataset):
    """Unfolded Icosahedron dataset.


    Examples
    --------
    >>> root_dataset = datasets.MNIST(root='raw_data', train=True, download=True)
    >>> dataset = OmniMNISTDataset(root_dataset, erp_shape=(60, 120), level=4, debug=True)
    >>> sample = dataset[0]

    >>> # show equirectangular image
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(sample['erp_img'])

    >>> # show image on icosahedron
    >>> from meshplot import plot
    >>> plot(dataset.vertices, dataset.faces, sample['ico_img'])

    >>> # show unfolded images
    >>> fig, ax = plt.subplots(1, 5)
    >>> _ = [ax[i].imshow(sample[(i+3)%5]) for i in range(5)]
    """

    def __init__(self, dataset, erp_shape, level, rotate=False, transform=None, debug=False, drop_NE=True):
        self.dataset = dataset
        self.transform = transform
        self.erp_shape = erp_shape
        self.level = level
        self.rotate = rotate

        self.vertices, self.faces = get_icosahedron(level)

        self.img_coord = get_unfold_imgcoord(level, drop_NE=drop_NE)

        self.debug = debug

    def __len__(self):
        return len(self.dataset)

    def get_erp_image(self, idx, v_rot=0, h_rot=0, erp_shape=None):
        if erp_shape is None:
            erp_shape = self.erp_shape
        img = np.array(self.dataset[idx][0])
        erp_img = img2ERP(img, v_rot=v_rot, h_rot=h_rot, outshape=erp_shape)
        return erp_img

    @property
    def classes(self):
        return self.dataset.classes

    def __getitem__(self, idx):
        sample = {}

        sample['label'] = self.dataset[idx][1]

        if self.rotate:
            h_rot = np.random.uniform(-180, 180)
            v_rot = np.random.uniform(-90, 90)
        else:
            v_rot = 0
            h_rot = 0

        erp_img = self.get_erp_image(idx, v_rot=v_rot, h_rot=h_rot)
        ico_img = erp2sphere(erp_img, self.vertices)

        # unfolded images
        for i in range(5):
            sample[i] = ico_img[self.img_coord[i]]

        sample['ico_img'] = ico_img

        # debug
        if self.debug:
            sample['erp_img'] = erp_img

        if self.transform:
            sample = self.transform(sample)

        return sample


# --------------------
# Custom transform
class ToTensor(object):
    def __init__(self):
        self.ToTensor = transforms.ToTensor()
        self.ico_img = 'ico_img'

    def __call__(self, sample):
        if self.ico_img in sample:
            sample[self.ico_img] = self.ToTensor(sample[self.ico_img][np.newaxis])
        for i in range(5):
            sample[i] = self.ToTensor(sample[i])
        return sample


class Normalize(object):
    def __init__(self, mean, std):
        self.normalizer = transforms.Normalize(mean, std)
        self.ico_img = 'ico_img'

    def __call__(self, sample):
        for i in range(5):
            sample[i] = self.normalizer(sample[i])
        sample[self.ico_img] = self.normalizer(sample[self.ico_img])
        return sample
