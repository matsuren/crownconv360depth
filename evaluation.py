import igl  # You need to import igl first to avoid some errors
import argparse
import json
import os
from collections import OrderedDict
from logging import getLogger, StreamHandler, FileHandler, INFO
from os.path import join

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader import FisheyeToIcoDataset, ToTensor, Normalize
from dataloader import OmniStereoDataset
from models import IcoSweepNet
from utils import evaluation_metrics, InvDepthConverter

# global var
args = None
writer = None
# setting logger
logger = getLogger(__name__)
logger.setLevel(INFO)
# create handler
handler_stream = StreamHandler()
handler_stream.setLevel(INFO)
logger.addHandler(handler_stream)

ToPIL = lambda x: transforms.ToPILImage()(x.cpu())
DN = lambda x: 0.5 + (0.5 * x)  # DeNormalizer

parser = argparse.ArgumentParser(description='Training for icosweepnet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('root_dir', metavar='DATA_DIR', help='path to dataset')
parser.add_argument('pretrained', metavar='PRETRAINED_PATH', help='path to pre-trained model')
parser.add_argument('-v', '--val-list', default='./dataloader/omnihouse_val.txt',
                    type=str, help='Text file includes filenames for validation')
parser.add_argument('-b', '--batch-size', default=2, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='J', help='number of data loading workers')
parser.add_argument('--arch', default='val_icosweep', type=str, help='architecture name for log folder')
parser.add_argument('--log-interval', type=int, default=1, metavar='L', help='tensorboard log interval')
parser.add_argument('--fov', type=float, default=220, help='field of view of the camera in degree')
parser.add_argument('--save_depth', action='store_true', help='save depth prediction')


def run(epoch, mode, model, data_loader, converter, device, optimizer=None, show_metrics=False, depth_folder=None):
    global args, writer, logger
    ndisp = model.module.ndisp

    if mode == 'train':
        print('Training mode')
        model.train()
    else:
        print(f"{mode} mode")
        model.eval()

    if show_metrics:
        preds = []
        gts = []
        gt_idepths = []
        if depth_folder is not None:
            os.mkdir(depth_folder)

    losses = []
    pbar = tqdm(data_loader)

    # save_gt = True
    # if save_gt:
    #     for idx, batch in enumerate(pbar):
    #         gt_idepth = batch['idepth']
    #         gt_idepths.append(gt_idepth.cpu())
    #     gt_idepths = torch.cat(gt_idepths)
    #     if depth_folder is not None:
    #         for i in range(len(gt_idepths)):
    #             fname = join(depth_folder, f'gt_{i + 1:05}.npy')
    #             np.save(fname, gt_idepths[i].numpy())
    #     return -1

    for idx, batch in enumerate(pbar):
        # to cuda
        for key in batch:
            batch[key] = batch[key].to(device)
        gt_idepth = batch['idepth']
        gt_invd_idx = converter.invdepth_to_index(gt_idepth)

        if mode == 'train':
            # Forward
            pred = model(batch)
            # Loss function
            loss = F.smooth_l1_loss(pred, gt_invd_idx, reduction='mean')
            losses.append(loss.item())
            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                pred = model(batch)
                # Loss function
                loss = F.smooth_l1_loss(pred, gt_invd_idx, reduction='mean')
                losses.append(loss.item())

        if show_metrics:
            # save for evaluation
            preds.append(pred.cpu())
            gts.append(gt_invd_idx.cpu())

        # update progress bar
        display = OrderedDict(mode=f'{mode}', epoch=f"{epoch:>2}", loss=f"{losses[-1]:.4f}")
        pbar.set_postfix(display)

    # End of one epoch
    ave_loss = sum(losses) / len(losses)
    logger.info(f"Epoch:{epoch}, Loss average:{ave_loss:.4f}")
    if show_metrics:
        preds = torch.cat(preds)
        gts = torch.cat(gts)
        errors, error_names = evaluation_metrics(preds, gts, ndisp)
        logger.info("Evaluation metrics: ")
        logger.info("{:>8}, {:>8}, {:>8}, {:>8}, {:>8}".format(*error_names))
        logger.info("{:8.4f}, {:8.4f}, {:8.4f}, {:8.4f}, {:8.4f}".format(*errors))

        if depth_folder is not None:
            for i in range(len(preds)):
                fname = join(depth_folder, f'{i + 1:05}.npy')
                np.save(fname, preds[i].numpy())

    return ave_loss


def main():
    global args, writer, logger
    args = parser.parse_args()
    logger.info('Arguments:')
    logger.info(json.dumps(vars(args), indent=1))
    # Dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    if device.type != 'cpu':
        cudnn.benchmark = True
    logger.info(f"device:{device}")

    # Load pretrained model
    logger.info("=> loading checkpoints")
    checkpoint = torch.load(args.pretrained)
    ndisp = checkpoint['ndisp']
    # min_depth = checkpoint['min_depth']
    level = checkpoint['level']
    logger.info(f'ndisp:{ndisp}')
    logger.info(f'level:{level}')

    logger.info('=> setting model')
    model = IcoSweepNet(args.root_dir, ndisp, level, fov=args.fov)
    model = model.to(device)
    invd_0 = model.inv_depths[0]
    invd_max = model.inv_depths[-1]
    converter = InvDepthConverter(model.ndisp, invd_0, invd_max)
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch'] - 1
    logger.info("=> Pretrained model epoch {}".format(epoch))

    logger.info('=> setting data loader')
    transform = transforms.Compose([ToTensor(), Normalize()])
    reduction = model.level - model.idepth_level
    root_val_dataset = OmniStereoDataset(args.root_dir, args.val_list, fov=args.fov)
    val_dataset = FisheyeToIcoDataset(root_val_dataset, root_val_dataset.ocams, root_val_dataset.pose_dict,
                                      level=model.level, reduction=reduction, transform=transform)
    logger.info(f'{len(val_dataset)} samples were found.')
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=args.workers)

    model = nn.DataParallel(model)

    # setup writer
    log_folder = os.path.dirname(args.pretrained)
    logger.info(f'=> save in checkpoint folder: {log_folder}')
    base = os.path.splitext(os.path.basename(args.pretrained))[0]
    root_dirname = args.root_dir.strip('/').split('/')[-1]
    # setup logger file handler
    handler = FileHandler(join(log_folder, f'eval_{root_dirname}_{base}.log'))
    handler.setLevel(INFO)
    logger.addHandler(handler)

    logger.info('Start evaluation')

    # ----------------------------
    # evaluation
    mode = 'eval'
    depth_folder = join(log_folder, f'depth_{root_dirname}_{base}') if args.save_depth else None
    ave_loss = run(epoch, mode, model, val_loader, converter, device, optimizer=None, show_metrics=True,
                   depth_folder=depth_folder)
    logger.info('Finish training.')


if __name__ == '__main__':
    main()
