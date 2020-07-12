import cv2
import numpy as np
import torch


class InvDepthConverter(object):
    def __init__(self, ndisp, invd_0, invd_max):
        self._ndisp = ndisp
        self._invd_0 = invd_0
        self._invd_max = invd_max

    def invdepth_to_index(self, idepth):
        invd_idx = (self._ndisp - 1) * (idepth - self._invd_0) / (self._invd_max - self._invd_0)
        return invd_idx

    def index_to_invdepth(self, invd_idx):
        idepth = self._invd_0 + invd_idx * (self._invd_max - self._invd_0) / (self._ndisp - 1)
        return idepth


def apply_colormap(gray, colormap=cv2.COLORMAP_VIRIDIS):
    array = gray.squeeze() / gray.max()
    array = (array * 255).astype(np.uint8)
    colored_array = cv2.applyColorMap(array, colormap)[:, :, ::-1]
    colored_array = colored_array.astype(np.float32).transpose(2, 0, 1) / 255
    return colored_array


def evaluation_metrics(preds, gts, ndisp):
    """Evaluation metrics

    Parameters
    ----------
    preds : torch.tensor
        Prediction invdepth index torch tensor [B x vertex_num]
    gts : torch.tensor
        Groundtruth invdepth index torch tensor [B x vertex_num]
    ndisp : int
        Number of disparity

    Returns
    -------
        total_errors : numpy array
            error values of ["a1", "a3", "a5", "mae", "rms"]
        error_names : list of string
            ["a1", "a3", "a5", "mae", "rms"]
    """
    if isinstance(preds, list):
        preds = torch.cat(preds)
    if isinstance(gts, list):
        gts = torch.cat(gts)
    assert preds.size() == gts.size()
    assert preds.ndim == 3

    pred_array = preds.detach().cpu().numpy()
    gt_array = gts.detach().cpu().numpy()
    b, h, w = pred_array.shape

    # calculate total mean error
    total_errors = []
    for one_pred, one_gt in zip(pred_array, gt_array):
        error = np.abs(one_pred - one_gt).flatten()
        # Eq.(3) error
        error = (error / ndisp) * 100

        a1 = (error > 1).mean() * 100  # %
        a3 = (error > 3).mean() * 100  # %
        a5 = (error > 5).mean() * 100  # %

        mae = error.mean()
        rms = np.sqrt((error ** 2).mean())

        total_errors.append([a1, a3, a5, mae, rms])

    total_errors = np.array(total_errors).mean(axis=0)
    error_names = ["a1", "a3", "a5", "mae", "rms"]
    return total_errors, error_names
