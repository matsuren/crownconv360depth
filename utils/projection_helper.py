# original https://github.com/ChiWeiHsiao/SphereNet-pytorch/blob/master/spherenet/dataset.py
# modified by https://github.com/matsuren
from functools import lru_cache

import numpy as np
from scipy.ndimage.interpolation import map_coordinates


@lru_cache(maxsize=360)
def genuv(h, w, v_rot=0):
    assert -np.pi / 2 <= v_rot and v_rot <= np.pi / 2
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = (u + 0.5) * 2 * np.pi / w - np.pi
    v = (v + 0.5) * np.pi / h - np.pi / 2
    uv = np.stack([u, v], axis=-1)

    if v_rot != 0:
        # rotation
        xyz = uv2xyz(uv.astype(np.float64))
        # Rx = np.array([
        #     [1, 0, 0],
        #     [0, np.cos(v_rot), np.sin(v_rot)],
        #     [0, -np.sin(v_rot), np.cos(v_rot)],
        # ])
        xyz_rot = xyz.copy()
        xyz_rot[..., 0] = xyz[..., 0]
        xyz_rot[..., 1] = np.cos(v_rot) * xyz[..., 1] + np.sin(v_rot) * xyz[..., 2]
        xyz_rot[..., 2] = -np.sin(v_rot) * xyz[..., 1] + np.cos(v_rot) * xyz[..., 2]
        uv = xyz2uv(xyz_rot)

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


def uv2proj_img_idx(uv, h, w, u_fov, v_fov):
    assert 0 < u_fov and u_fov < np.pi
    assert 0 < v_fov and v_fov < np.pi

    u = uv[..., 0]
    v = uv[..., 1]

    x = np.tan(u)
    y = np.tan(v) / np.cos(u)
    x = x * w / (2 * np.tan(u_fov / 2)) + w / 2
    y = y * h / (2 * np.tan(v_fov / 2)) + h / 2

    invalid = (u < -u_fov / 2) | (u > u_fov / 2) | \
              (v < -v_fov / 2) | (v > v_fov / 2)
    x[invalid] = -100
    y[invalid] = -100

    return np.stack([y, x], axis=0)


def remap(img, img_idx, cval=[0, 0, 0], method="linear"):
    # interpolation method
    if method == "linear":
        order = 1
    else:
        # nearest
        order = 0

    # remap image
    if img.ndim == 2:
        # grayscale
        x = map_coordinates(img, img_idx, order=order, cval=cval[0])
    elif img.ndim == 3:
        # color
        x = np.zeros([*img_idx.shape[1:], img.shape[2]], dtype=img.dtype)
        for i in range(img.shape[2]):
            x[..., i] = map_coordinates(img[..., i], img_idx, order=order, cval=cval[i])
    else:
        assert False, 'img.ndim should be 2 (grayscale) or 3 (color)'

    return x


def img2ERP(img, h_rot=0, v_rot=0, outshape=(60, 60), fov=120, cval=[0, 0, 0]):
    h, w = img.shape[:2]

    fov_rad = fov * np.pi / 180
    h_rot_rad = h_rot * np.pi / 180
    v_rot_rad = v_rot * np.pi / 180

    # Vertical rotate if applicable
    uv = genuv(*outshape, v_rot_rad)
    img_idx = uv2proj_img_idx(uv, h, w, fov_rad, fov_rad)

    # transform
    x = remap(img, img_idx, cval=cval)

    # Horizontal rotate
    delta = 2 * np.pi / (outshape[1])
    v_rot_idx = int(np.round(h_rot_rad / delta))
    x = np.roll(x, v_rot_idx, axis=1)
    return x


def uv2img_idx(uv, erp_img):
    h, w = erp_img.shape[:2]
    delta_w = 2 * np.pi / w
    delta_h = np.pi / h
    x = uv[..., 0] / delta_w + w / 2 - 0.5
    y = uv[..., 1] / delta_h + h / 2 - 0.5
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    return np.stack([y, x], axis=0)


def erp2sphere(erp_img, V, method="linear"):
    """

    Parameters
    ----------
    erp_img: equirectangular projection image
    V: array of spherical coordinates of shape (n_vertex, 3)
    method: interpolation method. "linear" or "nearest"
    """
    uv = xyz2uv(V)
    img_idx = uv2img_idx(uv, erp_img)
    x = remap(erp_img, img_idx, method=method)
    return x


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision import datasets

    outshape = (60, 60)
    print("getting mnist data")
    trainset = datasets.MNIST(root='raw_data', train=True, download=True)

    results = []
    for idx in range(3):
        h_rot = np.random.uniform(-180, 180)
        v_rot = np.random.uniform(-90, 90)
        print(f'Rotate horizontal:{h_rot:.1f} deg, vertical {v_rot:.1f} deg')
        img = np.array(trainset[idx][0])
        label_str = trainset.classes[trainset[idx][1]]
        x = img2ERP(img, v_rot=v_rot, h_rot=h_rot, outshape=outshape)
        results.append((img, x, label_str))

    # show
    fig, ax = plt.subplots(3, 2, figsize=(5, 8))
    for i, (img, x, label_str) in enumerate(results):
        ax[i][0].set_title(label_str)
        ax[i][0].imshow(img)
        ax[i][1].imshow(x)
    for it in ax.flatten():
        it.set_yticklabels([])
        it.set_xticklabels([])
    plt.show()
