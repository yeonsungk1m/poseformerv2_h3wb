import argparse
import os
import random

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

from common.load_data_3dhp_mae import Fusion
from common.utils import mpjpe_cal, AccumLoss, get_varialbe
from model.model_poseformerv2 import Model
from common.opt import opts

LATEST_CHECKPOINT = "latest_epoch.pth"
BEST_CHECKPOINT = "best_epoch.pth"


def _resolve_resume_path(opt):
    """Return an absolute path to the checkpoint provided via --resume."""

    if not getattr(opt, "resume", ""):
        return ""

    if os.path.isfile(opt.resume):
        return opt.resume

    return os.path.join(opt.checkpoint, opt.resume)


def _collect_rng_states():
    """Collect RNG states so that training can be resumed deterministically."""

    rng_states = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        rng_states["cuda_random_state"] = torch.cuda.get_rng_state_all()

    return rng_states


def _restore_rng_states(checkpoint):
    """Restore RNG states stored in a checkpoint if available."""

    python_state = checkpoint.get("python_random_state")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = checkpoint.get("numpy_random_state")
    if numpy_state is not None:
        np.random.set_state(numpy_state)

    torch_state = checkpoint.get("torch_random_state")
    if torch_state is not None:
        torch.set_rng_state(torch_state)

    cuda_state = checkpoint.get("cuda_random_state")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)

def _get_module(model):
    """Return the underlying module, handling DataParallel wrappers."""

    return model.module if isinstance(model, nn.DataParallel) else model


def _infer_center_index(model, seq_len):
    """Infer the center frame index for a ground-truth sequence."""

    module = _get_module(model) if model is not None else None
    if module is not None:
        for attr in ("t_out_idx", "t_idx"):
            center_idx = getattr(module, attr, None)
            if center_idx is None:
                continue
            center_idx = int(center_idx)
            if 0 <= center_idx < seq_len:
                return center_idx

    return seq_len // 2


def _slice_target_sequence(target_seq, length, model=None):
    """Slice the target sequence around its center to match the prediction length."""

    assert length > 0, "Prediction length must be positive"
    total_len = target_seq.shape[1]
    if length >= total_len:
        return target_seq.contiguous()

    center_idx = _infer_center_index(model, total_len)
    start = center_idx - length // 2
    start = max(0, min(start, total_len - length))
    end = start + length
    return target_seq[:, start:end].contiguous()

def _save_checkpoint(
    opt,
    epoch,
    model,
    loss_net,
    optimizer,
    optimizer_loss,
    best_metric=None,
    is_best=False,
):
    """Persist the current training state to disk."""

    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "loss_net": loss_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_loss": optimizer_loss.state_dict(),
        "lr": optimizer.param_groups[0]["lr"] if optimizer.param_groups else None,
        "lr_loss": optimizer_loss.param_groups[0]["lr"]
        if optimizer_loss.param_groups
        else None,
        "best_metric": best_metric,
    }

    checkpoint.update(_collect_rng_states())

    os.makedirs(opt.checkpoint, exist_ok=True)
    latest_path = os.path.join(opt.checkpoint, LATEST_CHECKPOINT)
    torch.save(checkpoint, latest_path)

    if is_best:
        best_path = os.path.join(opt.checkpoint, BEST_CHECKPOINT)
        torch.save(checkpoint, best_path)
    return latest_path

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
        B = inputs_3d.shape[0]

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
        module = _get_module(model)
        t_idx = getattr(module, "t_idx", inputs_2d.shape[2] // 2)
        inputs_2d_center = inputs_2d[
            :, :, t_idx : t_idx + pred_detached.shape[1], :, 0
        ].permute(0, 2, 3, 1)
        target_detached = _slice_target_sequence(inputs_3d, pred_detached.shape[1], model)
        energy_hat = loss_net(inputs_2d_center, pred_detached).view(B, pred_detached.shape[1])
        energy_gt = loss_net(inputs_2d_center, target_detached.detach()).view(
            B, pred_detached.shape[1]
        )
        loss_lossnet = loss_fn(
            pred_detached.detach(), target_detached.detach(), energy_hat, energy_gt
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
        target_pose = _slice_target_sequence(inputs_3d, pred_3d.shape[1], model)
        loss_pose = mpjpe_cal(pred_3d, target_pose)
        energy_pred = loss_net(inputs_2d_center, pred_3d).mean()
        loss_total = loss_pose + opt.energy_weight * energy_pred
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        for p in loss_net.parameters():
            p.requires_grad = True

        B_pose, T_pose = pred_3d.shape[:2]
        losses.update(loss_pose.item() * B_pose * T_pose, B_pose * T_pose)

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
            target_pose = _slice_target_sequence(inputs_3d, pred_3d.shape[1], model)
            loss = mpjpe_cal(pred_3d, target_pose)
            B_eval, T_eval = pred_3d.shape[:2]
            losses.update(loss.item() * B_eval * T_eval, B_eval * T_eval)
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

    start_epoch = 1
    best_metric = None

    resume_path = _resolve_resume_path(opt)
    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"No checkpoint found at {resume_path}")

        print(f"INFO: Loading checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu")

        if isinstance(checkpoint, dict):
            model_state = checkpoint.get("model")
            if model_state is None:
                model_state = checkpoint
            model.load_state_dict(model_state, strict=False)

            loss_state = checkpoint.get("loss_net")
            if loss_state is not None:
                try:
                    loss_net.load_state_dict(loss_state, strict=False)
                except RuntimeError as err:
                    print(
                        "WARNING: Failed to load SEAL loss network weights from checkpoint:"
                        f" {err}"
                    )
            else:
                print("WARNING: Checkpoint is missing SEAL loss network weights.")

            optimizer_state = checkpoint.get("optimizer")
            if optimizer_state is not None:
                try:
                    optimizer.load_state_dict(optimizer_state)
                except ValueError as err:
                    print(
                        "WARNING: Failed to load optimizer state from checkpoint:"
                        f" {err}"
                    )
            else:
                print("WARNING: Checkpoint does not contain optimizer state. Initializing optimizer from scratch.")

            optimizer_loss_state = checkpoint.get("optimizer_loss")
            if optimizer_loss_state is not None:
                try:
                    optimizer_loss.load_state_dict(optimizer_loss_state)
                except ValueError as err:
                    print(
                        "WARNING: Failed to load loss optimizer state from checkpoint:"
                        f" {err}"
                    )
            else:
                print(
                    "WARNING: Checkpoint does not contain loss optimizer state. "
                    "Initializing loss optimizer from scratch."
                )

            resume_lr = checkpoint.get("lr")
            if resume_lr is not None:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = resume_lr

            resume_lr_loss = checkpoint.get("lr_loss")
            if resume_lr_loss is not None:
                for param_group in optimizer_loss.param_groups:
                    param_group["lr"] = resume_lr_loss

            start_epoch = max(checkpoint.get("epoch", 0) + 1, 1)
            best_metric = checkpoint.get(
                "best_metric", checkpoint.get("best_test", best_metric)
            )

            if best_metric is not None:
                print(f"INFO: Best recorded test MPJPE: {best_metric:.4f}")

            _restore_rng_states(checkpoint)

            print(f"INFO: Resumed training from epoch {start_epoch - 1}")
        else:
            model.load_state_dict(checkpoint, strict=False)
            print(
                "WARNING: Loaded checkpoint without optimizer information. "
                "Optimizers will be reinitialized."
            )

    for epoch in range(start_epoch, opt.nepoch + 1):
        loss_train = train_epoch(
            opt, train_loader, model, loss_net, optimizer, optimizer_loss, loss_fn
        )
        print(f"Epoch {epoch}: train MPJPE {loss_train:.4f}")
        is_best = False
        if opt.test:
            loss_test = evaluate(opt, test_loader, model)
            print(f"Epoch {epoch}: test MPJPE {loss_test:.4f}")
            
            if best_metric is None or loss_test < best_metric:
                best_metric = loss_test
                is_best = True
                print(f"New best test MPJPE: {best_metric:.4f}")

        ckpt_path = _save_checkpoint(
            opt,
            epoch,
            model,
            loss_net,
            optimizer,
            optimizer_loss,
            best_metric,
            is_best=is_best,
        )
        print(f"Saved checkpoint to {ckpt_path}")