"""
Training loop with per-epoch history tracking, live progress bar and
optional early stopping.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional
    def tqdm(x, **kwargs):
        return x


class History(dict):
    """Simple dict of lists with a save helper."""

    KEYS = ("train_loss", "test_loss", "train_acc", "test_acc", "lr", "epoch_time")

    def __init__(self):
        super().__init__({k: [] for k in self.KEYS})

    def append(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self:
                self[k] = []
            self[k].append(v)

    def save(self, path: str, extra: dict | None = None):
        payload = {"history": dict(self)}
        if extra:
            payload.update(extra)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path


# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    """Return (mean loss, accuracy) over a loader."""
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        bs = yb.size(0)
        loss_sum += loss.item() * bs
        correct += (logits.argmax(-1) == yb).sum().item()
        n += bs
    return loss_sum / n, correct / n


def train(model, train_loader, test_loader, cfg, device, majority: float = 0.0):
    """Train a model and return (history, best_test_acc)."""
    model = model.to(device)
    print(f"model[{cfg.MODEL}"
          + (f"/{cfg.THETA_MODE}" if cfg.MODEL.lower() == "s4" else "")
          + f"] trainable params: {model.num_params():,}")

    try:
        groups = model.param_groups(
            cfg.LR,
            ssm_mult=cfg.LR_SSM_MULT,
            theta_mult=cfg.LR_THETA_MULT,
            state_mult=getattr(cfg, "LR_STATE_MULT", 0.01),
        )
    except TypeError:  # baselines take only lr
        groups = model.param_groups(cfg.LR)
    optimizer = torch.optim.Adam(groups, weight_decay=cfg.WEIGHT_DECAY)
    use_sched = getattr(cfg, "USE_SCHEDULER", True)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS)
        if use_sched else None
    )
    print(f"lr schedule: {'cosine annealing' if use_sched else f'constant ({cfg.LR:.1e})'}")
    criterion = nn.CrossEntropyLoss()

    hist = History()
    best_acc, best_epoch, patience_left = 0.0, 0, cfg.EARLY_STOP_PATIENCE

    print(f"--- training: {cfg.NUM_EPOCHS} epochs x {len(train_loader)} batches ---")

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        t0 = time.time()

        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"ep {epoch:3d}/{cfg.NUM_EPOCHS}", leave=False)

        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            if cfg.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()

            bs = yb.size(0)
            loss_sum += loss.item() * bs
            correct += (logits.argmax(-1) == yb).sum().item()
            n += bs
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(loss=f"{loss_sum / n:.4f}", acc=f"{correct / n:.4f}")

        cur_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        train_loss, train_acc = loss_sum / n, correct / n
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        dt = time.time() - t0

        hist.append(train_loss=train_loss, test_loss=test_loss,
                    train_acc=train_acc, test_acc=test_acc,
                    lr=cur_lr, epoch_time=dt)

        marker = ""
        if test_acc > best_acc:
            best_acc, best_epoch = test_acc, epoch
            patience_left = cfg.EARLY_STOP_PATIENCE
            marker = "  *"
            if cfg.SAVE_BEST:
                os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, f"best_{cfg.run_name()}.pt"))
        else:
            patience_left -= 1

        margin_str = ""
        if getattr(cfg, "TRAIN_SSM_STATE", False):
            margins = [m.stability_margin() for m in model.modules()
                       if hasattr(m, "stability_margin")]
            if margins:
                worst = max(margins)
                margin_str = f" | Re(L)max={worst:+.3f}"
                if worst >= 0:
                    print(f"  WARNING: Re(Lambda) reached {worst:+.4f} -- "
                          f"SSM is at/over the stability boundary")

        if epoch % cfg.PRINT_EVERY == 0 or epoch == cfg.NUM_EPOCHS:
            print(f"ep {epoch:3d}/{cfg.NUM_EPOCHS} | "
                  f"train: loss={train_loss:.4f} acc={train_acc:.4f} | "
                  f"test: loss={test_loss:.4f} acc={test_acc:.4f} | "
                  f"lr={cur_lr:.2e}{margin_str} | {dt:.1f}s{marker}", flush=True)

        if cfg.EARLY_STOP_PATIENCE > 0 and patience_left <= 0:
            print(f"early stopping at epoch {epoch} "
                  f"(no improvement for {cfg.EARLY_STOP_PATIENCE} epochs)")
            break

    gain = best_acc - majority
    print(f"\n>>> BEST test acc = {best_acc:.4f} at epoch {best_epoch} "
          f"| majority = {majority:.4f} | gain = {gain:+.4f}")

    return hist, best_acc
