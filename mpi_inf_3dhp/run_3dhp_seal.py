import argparse
import os
import random
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from common.loss_net import (
    ContinuousGraphLossNet,
    adj_from_parents,
    NCELoss,
    MarginBasedLoss,
)

from mpi_inf_3dhp.common.load_data_3dhp_mae import Fusion
from mpi_inf_3dhp.common.utils import mpjpe_cal, AccumLoss, get_varialbe
from mpi_inf_3dhp.model.model_poseformerv2 import Model
from mpi_inf_3dhp.common.opt import opts


def create_dataloaders(opt):
    """Create train and test dataloaders for MPI-INF-3DHP."""
    train_set = Fusion(opt=opt, train=True, root_path=opt.root_path)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers),
        pin_memory=True,
    )
    test_set = Fusion(opt=opt, train=False, root_path=opt.root_path)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=int(opt.workers),
        pin_memory=True,
    )
    return train_loader, test_loader


def train_epoch(opt, train_loader, model, loss_net, optimizer, optimizer_loss, loss_fn):
    """One training epoch with SEAL energy model."""
    model.train()
    loss_net.train()
    losses = AccumLoss()

    for data in tqdm(train_loader, 0):
        loss_net.train()
        batch_cam, gt_3D, input_2D, seq, subject, scale, bb_box, cam_ind = data
        [input_2D, gt_3D, scale] = get_varialbe('train', [input_2D, gt_3D, scale])

        N = input_2D.size(0)
        inputs_2d = (
            input_2D.view(N, -1, opt.n_joints, opt.in_channels, 1)
            .permute(0, 3, 1, 2, 4)
            .contiguous()
        )
        inputs_3d = gt_3D.view(N, -1, opt.out_joints, opt.out_channels)
        B, T = inputs_3d.shape[:2]

        # -------- Loss network update --------
        model.eval()
        with torch.no_grad():
            pred_detached, _ = model(inputs_2d)
        pred_detached = (
            pred_detached.permute(0, 2, 3, 4, 1)
            .contiguous()
            .view(N, -1, opt.out_joints, opt.out_channels)
        )
        pred_detached = pred_detached * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        t_idx = getattr(model.module, "t_idx", inputs_2d.shape[2] // 2)
        inputs_2d_center = inputs_2d[
            :, :, t_idx : t_idx + pred_detached.shape[1], :, 0
        ].permute(0, 2, 3, 1)
        energy_hat = loss_net(inputs_2d_center, pred_detached).view(B, T)
        energy_gt = loss_net(inputs_2d_center, inputs_3d.detach()).view(B, T)
        loss_lossnet = loss_fn(
            pred_detached.detach(), inputs_3d.detach(), energy_hat, energy_gt
        )
        optimizer_loss.zero_grad()
        loss_lossnet.backward()
        optimizer_loss.step()

        # -------- Pose network update --------
        loss_net.eval()
        for p in loss_net.parameters():
            p.requires_grad = False
        model.train()
        pred_3d, _ = model(inputs_2d)
        pred_3d = (
            pred_3d.permute(0, 2, 3, 4, 1)
            .contiguous()
            .view(N, -1, opt.out_joints, opt.out_channels)
        )
        pred_3d = pred_3d * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        inputs_2d_center = inputs_2d[
            :, :, t_idx : t_idx + pred_3d.shape[1], :, 0
        ].permute(0, 2, 3, 1)
        loss_pose = mpjpe_cal(pred_3d, inputs_3d)
        energy_pred = loss_net(inputs_2d_center, pred_3d).mean()
        loss_total = loss_pose + opt.energy_weight * energy_pred
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        for p in loss_net.parameters():
            p.requires_grad = True

        losses.update(loss_pose.item() * B * T, B * T)

        del (
            gt_3D,
            input_2D,
            inputs_3d,
            inputs_2d,
            pred_detached,
            energy_hat,
            energy_gt,
            pred_3d,
            energy_pred,
            loss_pose,
            loss_total,
        )
        torch.cuda.empty_cache()

    return losses.avg


def evaluate(opt, test_loader, model):
    """Evaluate MPJPE on the test set."""
    model.eval()
    losses = AccumLoss()
    with torch.no_grad():
        for data in tqdm(test_loader, 0):
            batch_cam, gt_3D, input_2D, seq, scale, bb_box = data
            [input_2D, gt_3D, scale] = get_varialbe('test', [input_2D, gt_3D, scale])
            N = input_2D.size(0)
            inputs_2d = (
                input_2D.view(N, -1, opt.n_joints, opt.in_channels, 1)
                .permute(0, 3, 1, 2, 4)
                .contiguous()
            )
            inputs_3d = gt_3D.view(N, -1, opt.out_joints, opt.out_channels)
            pred_3d, _ = model(inputs_2d)
            pred_3d = (
                pred_3d.permute(0, 2, 3, 4, 1)
                .contiguous()
                .view(N, -1, opt.out_joints, opt.out_channels)
            )
            pred_3d = pred_3d * scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            loss = mpjpe_cal(pred_3d, inputs_3d)
            B, T = inputs_3d.shape[:2]
            losses.update(loss.item() * B * T, B * T)
    return losses.avg


if __name__ == "__main__":
    opt = opts().parse()

    # Additional arguments for energy model
    extra_parser = argparse.ArgumentParser()
    extra_parser.add_argument("--lr-loss", type=float, default=1e-4)
    extra_parser.add_argument("--energy-weight", type=float, default=0.1)
    extra_parser.add_argument(
        "--em-loss-type", type=str, default="nce", choices=["nce", "margin"]
    )
    extra_parser.add_argument(
        "--em-margin-type",
        type=str,
        default="mse",
        choices=["mse", "mpjpe", "l1"],
    )
    extra_parser.add_argument("--em-margin-ratio", type=float, default=1.0)
    extra_args, _ = extra_parser.parse_known_args()
    for k, v in vars(extra_args).items():
        setattr(opt, k.replace("-", "_"), v)

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    train_loader, test_loader = create_dataloaders(opt)

    model = nn.DataParallel(Model(opt)).cuda()

    parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 8, 10, 11, 0, 13, 14, 15]
    adj = adj_from_parents(parents)
    loss_net = ContinuousGraphLossNet(adj=adj, num_joints=opt.out_joints)
    loss_net = nn.DataParallel(loss_net).cuda()

    if opt.em_loss_type == "nce":
        loss_fn = NCELoss()
    else:
        loss_fn = MarginBasedLoss(
            margin_ratio=opt.em_margin_ratio, loss_type=opt.em_margin_type
        )

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    optimizer_loss = optim.AdamW(loss_net.parameters(), lr=opt.lr_loss)

    for epoch in range(1, opt.nepoch + 1):
        loss_train = train_epoch(
            opt, train_loader, model, loss_net, optimizer, optimizer_loss, loss_fn
        )
        print(f"Epoch {epoch}: train MPJPE {loss_train:.4f}")
        if opt.test:
            loss_test = evaluate(opt, test_loader, model)
            print(f"Epoch {epoch}: test MPJPE {loss_test:.4f}")