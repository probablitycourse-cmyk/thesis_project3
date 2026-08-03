"""
Plotting helpers: single-run training curves and multi-run comparison.
"""

from __future__ import annotations

import json
import os

import numpy as np


def plot_history(hist, majority: float = 0.0, title: str = "", save_path: str | None = None):
    """Three-panel figure: loss, accuracy, overfitting gap + learning rate."""
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(hist["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.plot(epochs, hist["train_loss"], label="train", lw=1.8)
    ax.plot(epochs, hist["test_loss"], label="test", lw=1.8, ls="--")
    ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy")
    ax.set_title("Loss"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, hist["train_acc"], label="train", lw=1.8)
    ax.plot(epochs, hist["test_acc"], label="test", lw=1.8, ls="--")
    if majority > 0:
        ax.axhline(majority, color="red", ls=":", lw=1.5,
                   label=f"majority ({majority:.3f})")
    best = max(hist["test_acc"])
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy")
    ax.set_title(f"Accuracy -- best test = {best:.4f}")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    gap = np.array(hist["train_acc"]) - np.array(hist["test_acc"])
    ax.plot(epochs, gap, color="purple", lw=1.8, label="train-test gap")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("epoch"); ax.set_ylabel("gap", color="purple")
    ax.tick_params(axis="y", labelcolor="purple")
    ax.set_title("Overfitting gap & learning rate"); ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(epochs, hist["lr"], color="orange", lw=1.4, ls="-.", label="lr")
    ax2.set_ylabel("learning rate", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, labels, loc="upper left", fontsize=8)

    if title:
        plt.suptitle(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"figure saved: {save_path}")
    plt.show()
    return fig


def compare_runs(json_paths: list[str], save_path: str | None = None):
    """Overlay test accuracy and test loss from several saved runs."""
    import matplotlib.pyplot as plt

    runs = []
    for p in json_paths:
        with open(p) as f:
            payload = json.load(f)
        label = payload.get("run_name", os.path.basename(p).replace(".json", ""))
        runs.append((label, payload["history"], payload.get("majority", 0.0)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for label, hist, _ in runs:
        ep = np.arange(1, len(hist["test_acc"]) + 1)
        axes[0].plot(ep, hist["test_acc"], lw=1.8,
                     label=f"{label} (best {max(hist['test_acc']):.4f})")
        axes[1].plot(ep, hist["test_loss"], lw=1.8, label=label)

    majority = max((m for _, _, m in runs), default=0.0)
    if majority > 0:
        axes[0].axhline(majority, color="red", ls=":", lw=1.5,
                        label=f"majority ({majority:.3f})")

    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("test accuracy")
    axes[0].set_title("Test accuracy"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("test loss")
    axes[1].set_title("Test loss"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"figure saved: {save_path}")
    plt.show()

    print("\n" + "=" * 60)
    print(f"{'run':35s} {'best test acc':>15s}")
    print("=" * 60)
    for label, hist, _ in sorted(runs, key=lambda r: -max(r[1]["test_acc"])):
        print(f"{label:35s} {max(hist['test_acc']):>15.4f}")
    return fig
