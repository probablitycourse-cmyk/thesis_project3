"""
Entry point. Run a single experiment defined by config.CFG.

Command line:
    python main.py --model s4 --theta order --epochs 80 --target valence

From Colab:
    from config import CFG
    from main import run
    CFG.THETA_MODE = "order"
    hist, best = run(CFG)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from config import CFG
from data import make_loaders, make_loaders_per_subject
from models import build_model
from training import train, plot_history


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(cfg=CFG, show_plot: bool = True):
    """Build data + model, train, save history, optionally plot."""
    cfg.show()
    set_seed(cfg.SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cpu":
        print("WARNING: running on CPU will be very slow -- enable a GPU runtime")

    train_loader, test_loader, info = make_loaders(cfg)

    model = build_model(cfg, input_dim=info["n_channels"], out_dim=info["n_classes"])
    hist, best = train(model, train_loader, test_loader, cfg, device,
                       majority=info["majority_baseline"])

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(cfg.OUTPUT_DIR, f"{cfg.run_name()}.json")
    hist.save(json_path, extra={
        "run_name": cfg.run_name(),
        "config": {k: str(v) for k, v in cfg.as_dict().items()},
        "best_test_acc": best,
        "majority": info["majority_baseline"],
        "data_info": info,
    })
    print(f"history saved: {json_path}")

    if show_plot:
        plot_history(
            hist,
            majority=info["majority_baseline"],
            title=f"{cfg.run_name()}  (params={model.num_params():,})",
            save_path=os.path.join(cfg.OUTPUT_DIR, f"{cfg.run_name()}.png"),
        )

    return hist, best


def run_per_subject(cfg=CFG, show_plot: bool = True):
    """
    Subject-dependent protocol: train a separate model for every subject
    (each subject's own windows split 80/20) and report the mean accuracy.
    This is the protocol most published DREAMER results use.
    """
    cfg.show()
    set_seed(cfg.SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    per_subject = {}
    histories = {}

    for subj, train_loader, test_loader, info in make_loaders_per_subject(cfg):
        print(f"\n{'=' * 58}\nSUBJECT {subj}  "
              f"(train={info['n_train']:,}  test={info['n_test']:,}  "
              f"majority={info['majority_baseline']:.4f})\n{'=' * 58}")

        set_seed(cfg.SEED)
        model = build_model(cfg, input_dim=info["n_channels"], out_dim=info["n_classes"])
        hist, best = train(model, train_loader, test_loader, cfg, device,
                           majority=info["majority_baseline"])

        per_subject[subj] = {"best_acc": best, "majority": info["majority_baseline"]}
        histories[subj] = dict(hist)

    accs = np.array([v["best_acc"] for v in per_subject.values()])
    majs = np.array([v["majority"] for v in per_subject.values()])

    print(f"\n{'=' * 58}")
    print(f"SUBJECT-DEPENDENT RESULTS  ({cfg.run_name()})")
    print(f"{'=' * 58}")
    print(f"{'subject':>8} {'best acc':>10} {'majority':>10} {'gain':>8}")
    print("-" * 58)
    for subj in sorted(per_subject):
        v = per_subject[subj]
        print(f"{subj:>8} {v['best_acc']:>10.4f} {v['majority']:>10.4f} "
              f"{v['best_acc'] - v['majority']:>+8.4f}")
    print("-" * 58)
    print(f"{'MEAN':>8} {accs.mean():>10.4f} {majs.mean():>10.4f} "
          f"{(accs - majs).mean():>+8.4f}")
    print(f"{'STD':>8} {accs.std():>10.4f}")
    print(f"{'MIN/MAX':>8} {accs.min():>10.4f} / {accs.max():.4f}")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(cfg.OUTPUT_DIR, f"{cfg.run_name()}_perSubject.json")
    with open(out_path, "w") as f:
        json.dump({
            "run_name": cfg.run_name() + "_perSubject",
            "protocol": "subject-dependent",
            "config": {k: str(v) for k, v in cfg.as_dict().items()},
            "per_subject": {str(k): v for k, v in per_subject.items()},
            "mean_acc": float(accs.mean()),
            "std_acc": float(accs.std()),
            "mean_majority": float(majs.mean()),
            "histories": {str(k): v for k, v in histories.items()},
        }, f, indent=2)
    print(f"\nresults saved: {out_path}")

    if show_plot:
        _plot_per_subject(per_subject, cfg,
                          save_path=os.path.join(cfg.OUTPUT_DIR,
                                                 f"{cfg.run_name()}_perSubject.png"))

    return per_subject, float(accs.mean())


def _plot_per_subject(per_subject: dict, cfg, save_path: str | None = None):
    """Bar chart of per-subject accuracy against each subject's majority baseline."""
    import matplotlib.pyplot as plt

    subjects = sorted(per_subject)
    accs = [per_subject[s]["best_acc"] for s in subjects]
    majs = [per_subject[s]["majority"] for s in subjects]
    x = np.arange(len(subjects))

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(x - 0.2, accs, width=0.4, label="model accuracy")
    ax.bar(x + 0.2, majs, width=0.4, label="majority baseline", alpha=0.6)
    ax.axhline(np.mean(accs), color="green", ls="--", lw=1.5,
               label=f"mean acc = {np.mean(accs):.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.set_xlabel("subject"); ax.set_ylabel("accuracy")
    ax.set_title(f"Subject-dependent results -- {cfg.run_name()}")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"figure saved: {save_path}")
    plt.show()
    return fig


def parse_args():
    p = argparse.ArgumentParser(description="S4 / baseline experiments on DREAMER EEG")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--model", type=str, default=None, choices=["s4", "lstm", "transformer"])
    p.add_argument("--theta", type=str, default=None, choices=["none", "const", "order"])
    p.add_argument("--target", type=str, default=None, choices=["valence", "arousal", "dominance"])
    p.add_argument("--split", type=str, default=None, choices=["random", "subject"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--H", type=int, default=None)
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--window-sec", type=float, default=None)
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--protocol", type=str, default="pooled",
                   choices=["pooled", "per-subject"],
                   help="pooled = single model on all subjects; "
                        "per-subject = one model per subject (subject-dependent)")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def apply_args(cfg, args):
    mapping = {
        "data_dir": "DATA_DIR", "model": "MODEL", "theta": "THETA_MODE",
        "target": "TARGET", "split": "SPLIT_MODE", "epochs": "NUM_EPOCHS",
        "batch_size": "BATCH_SIZE", "lr": "LR", "H": "H", "N": "N",
        "layers": "NUM_LAYERS", "dropout": "DROPOUT",
        "window_sec": "WINDOW_SEC", "seed": "SEED",
    }
    for arg_name, cfg_name in mapping.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            setattr(cfg, cfg_name, val)
    if args.no_baseline:
        cfg.USE_BASELINE = False
    if getattr(args, "window_sec", None) is not None:
        cfg.STRIDE_SEC = cfg.WINDOW_SEC
    return cfg


if __name__ == "__main__":
    args = parse_args()
    apply_args(CFG, args)
    if args.protocol == "per-subject":
        run_per_subject(CFG, show_plot=not args.no_plot)
    else:
        run(CFG, show_plot=not args.no_plot)
