import igl  # You need to import igl first to avoid some errors
import argparse
import json
import os
import shutil
from collections import OrderedDict
from datetime import datetime
from logging import getLogger, StreamHandler, FileHandler, INFO
from os.path import join

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloader import FisheyeToIcoDataset, ToTensor, Normalize
from dataloader import OmniStereoDataset, ColorJitter, RandomShift
from dataloader.icosahedron_dataset import ico_to_erp
from models import IcoSweepNet
from utils import apply_colormap
from utils import evaluation_metrics, InvDepthConverter
from utils.feature_integration import vertex_feat_to_unfold_feat

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
parser.add_argument('-t', '--train-list', default='./dataloader/omnithings_train.txt',
                    type=str, help='Text file includes filenames for training')
parser.add_argument('-v', '--val-list', default='./dataloader/omnithings_val.txt',
                    type=str, help='Text file includes filenames for validation')
parser.add_argument('--level', type=int, default=7, metavar='N', help='icosahedron resolution level')
parser.add_argument('--depth_level', type=int, default=5, metavar='N', help='icosahedron resolution level')

parser.add_argument('--ndisp', type=int, default=16, metavar='N', help='number of disparity')
parser.add_argument('--epochs', default=30, type=int, metavar='N', help='total epochs')
parser.add_argument('--pretrained', default=None, metavar='PATH',
                    help='path to pre-trained model')

parser.add_argument('-b', '--batch-size', default=2, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='J', help='number of data loading workers')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--arch', default='icosweep', type=str, help='architecture name for log folder')
parser.add_argument('--log-interval', type=int, default=1, metavar='L', help='tensorboard log interval')
parser.add_argument('--fov', type=float, default=220, help='field of view of the camera in degree')
parser.add_argument('--aug_real', action='store_true', help='data augmentation for real images. (color jitter etc.')


def run(epoch, mode, model, data_loader, converter, device, optimizer=None, show_metrics=False):
    global args, writer
    cam_list = model.module.cam_list
    ndisp = model.module.ndisp
    idepth_level = model.module.idepth_level

    if mode == 'train':
        logger.info('Training mode')
        model.train()
    else:
        logger.info(f"{mode} mode")
        model.eval()

    if show_metrics:
        preds = []
        gts = []

    losses = []
    pbar = tqdm(data_loader)
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

        # tensorboard log
        niter = epoch * len(pbar) + idx
        if idx % args.log_interval == 0:
            writer.add_scalar(f'{mode}/loss', loss.item(), niter)
        if idx % 100 * args.log_interval == 0:
            batch_idx = 0
            for key in cam_list:
                feat, _ = vertex_feat_to_unfold_feat(batch[key])
                vis = make_grid([DN(feat[(i + 3) % 5][batch_idx]) for i in range(5)])
                writer.add_image(f'{mode}/{key}', vis, niter)
            feat, _ = vertex_feat_to_unfold_feat(gt_invd_idx)
            vis_gt = make_grid([feat[(i + 3) % 5][batch_idx] / ndisp for i in range(5)])
            feat, _ = vertex_feat_to_unfold_feat(pred)
            vis_pred = make_grid([feat[(i + 3) % 5][batch_idx] / ndisp for i in range(5)])
            writer.add_image(f'{mode}/pred', vis_pred, niter)
            writer.add_image(f'{mode}/gt', vis_gt, niter)
            # erp images by interpolation
            erp_pred = converter.index_to_invdepth(pred[0].detach().cpu()).squeeze().numpy()
            erp_pred = apply_colormap(ico_to_erp(erp_pred, idepth_level))
            writer.add_image(f'{mode}/erp_pred', erp_pred, niter)
            erp_gt = converter.index_to_invdepth(gt_invd_idx[0].detach().cpu()).squeeze().numpy()
            erp_gt = apply_colormap(ico_to_erp(erp_gt, idepth_level))
            writer.add_image(f'{mode}/erp_gt', erp_gt, niter)

    # End of one epoch
    ave_loss = sum(losses) / len(losses)
    writer.add_scalar(f'{mode}/loss_ave', ave_loss, epoch)
    logger.info(f"Epoch:{epoch}, Loss average:{ave_loss:.4f}")
    if show_metrics:
        errors, error_names = evaluation_metrics(preds, gts, ndisp)
        for name, val in zip(error_names, errors):
            writer.add_scalar(f'{mode}_metrics/{name}', val, epoch)
        logger.info("Evaluation metrics: ")
        logger.info("{:>8}, {:>8}, {:>8}, {:>8}, {:>8}".format(*error_names))
        logger.info("{:8.4f}, {:8.4f}, {:8.4f}, {:8.4f}, {:8.4f}".format(*errors))

    return ave_loss


def main():
    global args, writer

    args = parser.parse_args()
    logger.info('Arguments:')
    logger.info(json.dumps(vars(args), indent=1))
    # Dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    if device.type != 'cpu':
        cudnn.benchmark = True
    logger.info(f"device:{device}")

    logger.info('=> setting data loader')
    reduction = args.level - args.depth_level
    fisheye_transform = transforms.Compose([ColorJitter(), RandomShift()]) if args.aug_real else None

    # Dataset
    root_train_dataset = OmniStereoDataset(args.root_dir, args.train_list, fisheye_transform, fov=args.fov)
    ocam_dict = root_train_dataset.ocams
    # camera poses world <- T cam
    pose_dict = root_train_dataset.pose_dict

    transform = transforms.Compose([ToTensor(), Normalize()])
    trainset = FisheyeToIcoDataset(root_train_dataset, ocam_dict, pose_dict, level=args.level, reduction=reduction,
                                   transform=transform)
    logger.info(f'trainset:{len(trainset)} samples were found.')
    train_loader = DataLoader(trainset, args.batch_size, shuffle=True, num_workers=args.workers)

    root_val_dataset = OmniStereoDataset(args.root_dir, args.val_list, fov=args.fov)
    val_dataset = FisheyeToIcoDataset(root_val_dataset, root_val_dataset.ocams, root_val_dataset.pose_dict,
                                      level=args.level, reduction=reduction, transform=transform)
    logger.info(f'{len(val_dataset)} samples were found.')
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=args.workers)

    logger.info('=> setting model')
    model = IcoSweepNet(args.root_dir, args.ndisp, args.level, fov=args.fov)
    total_params = 0
    for param in model.parameters():
        total_params += np.prod(param.shape)
    logger.info(f"Total model parameters: {total_params:,}.")
    model = model.to(device)

    invd_0 = model.inv_depths[0]
    invd_max = model.inv_depths[-1]
    converter = InvDepthConverter(model.ndisp, invd_0, invd_max)

    # setup solver scheduler
    logger.info('=> setting optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger.info('=> setting scheduler')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(2 * args.epochs / 3), gamma=0.1)

    start_epoch = 0
    # Load pretrained model
    if args.pretrained:
        checkpoint = torch.load(args.pretrained)
        param_check = {
            'ndisp': model.ndisp,
            'min_depth': model.min_depth,
            'level': model.level,
        }
        for key, val in param_check.items():
            if not checkpoint[key] == val:
                logger.error(f'Error! Key:{key} is not the same as the checkpoints')

        logger.info("=> using pre-trained weights")
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("=> Resume training from epoch {}".format(start_epoch))
    #
    model = nn.DataParallel(model)

    # setup writer
    timestamp = datetime.now().strftime("%m%d-%H%M")
    log_folder = join('checkpoints', f'{args.arch}_{timestamp}')
    logger.info(f'=> create log folder: {log_folder}')
    os.makedirs(log_folder, exist_ok=True)
    with open(join(log_folder, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=1)
    writer = SummaryWriter(log_dir=log_folder)
    writer.add_text('args', json.dumps(vars(args), indent=1).replace('\n', '  \n'))
    logger.info('=> copy models folder to log folder')
    shutil.copytree('./models', join(log_folder, 'models'))
    # setup logger file handler
    handler = FileHandler(join(log_folder, 'train.log'))
    handler.setLevel(INFO)
    logger.addHandler(handler)

    logger.info('Start training')

    for epoch in range(start_epoch, args.epochs):
        # ----------------------------
        # training
        mode = 'train'
        ave_loss = run(epoch, mode, model, train_loader, converter, device, optimizer)
        # ----------------------------
        # evaluation
        mode = 'val'
        ave_loss = run(epoch, mode, model, val_loader, converter, device, optimizer=None, show_metrics=True)

        scheduler.step()
        # save
        save_data = {
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'ave_loss': ave_loss,
            'ndisp': model.module.ndisp,
            'min_depth': model.module.min_depth,
            'level': model.module.level,
        }
        torch.save(save_data, join(log_folder, f'checkpoints_{epoch}.pth'))

    writer.close()
    logger.info('Finish training.')


if __name__ == '__main__':
    main()
