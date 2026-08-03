"""
DREAMER EEG dataset: loading, baseline removal, windowing and split logic.

Expected directory layout (cfg.DATA_DIR):

    labels.pkl              (n_subjects, n_clips, 3)   valence / arousal / dominance
    baseline_data.pkl       (n_subjects, n_clips, T_b, n_channels)
    stimuli_{i}_clip.pkl    (n_subjects, T_i, n_channels)   for i = 0 .. n_clips-1

Each stimulus recording is cut into non-overlapping windows of WINDOW_SEC
seconds; every window inherits the label of its parent clip.
"""

from __future__ import annotations

import os
import pickle
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

TARGET_INDEX = {"valence": 0, "arousal": 1, "dominance": 2}


# ----------------------------------------------------------------------
def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_baseline_means(data_dir: str, n_clips: int) -> np.ndarray | None:
    """
    Return the per-(subject, clip, channel) temporal mean of the baseline
    recording, shape (n_subjects, n_clips, n_channels).

    The raw baseline array is large (a few GB in float64) so it is reduced
    immediately and then released.
    """
    path = os.path.join(data_dir, "baseline_data.pkl")
    if not os.path.exists(path):
        print("  baseline_data.pkl not found -- skipping baseline removal")
        return None

    baseline = _load_pickle(path)                # (n_subj, n_clips, T_b, n_ch)
    means = baseline.mean(axis=2)                # (n_subj, n_clips, n_ch)
    print(f"  baseline {baseline.shape} -> per-clip means {means.shape}")
    del baseline
    return means


# ----------------------------------------------------------------------
def build_arrays(cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the full windowed dataset.

    Returns
    -------
    X        (n_windows, window_len, n_channels) float32
    y        (n_windows,) int64
    subject  (n_windows,) int64   subject id of each window (for subject splits)
    """
    t0 = time.time()
    target_idx = TARGET_INDEX[cfg.TARGET]

    labels = _load_pickle(os.path.join(cfg.DATA_DIR, "labels.pkl"))
    n_subjects, n_clips, _ = labels.shape
    y_clip = labels[:, :, target_idx]
    print(f"  labels {labels.shape}  target={cfg.TARGET}")

    baseline_means = load_baseline_means(cfg.DATA_DIR, n_clips) if cfg.USE_BASELINE else None

    window = int(round(cfg.WINDOW_SEC * cfg.FS))
    stride = int(round(cfg.STRIDE_SEC * cfg.FS))

    X_parts, y_parts, subj_parts = [], [], []

    for clip in range(n_clips):
        path = os.path.join(cfg.DATA_DIR, f"stimuli_{clip}_clip.pkl")
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}, skipping")
            continue

        arr = _load_pickle(path)                      # (n_subj, T, n_ch)

        if baseline_means is not None:
            arr = arr - baseline_means[:, clip, None, :]

        T = arr.shape[1]
        starts = list(range(0, T - window + 1, stride))
        if not starts:
            print(f"  clip {clip}: too short ({T} < {window}), skipped")
            continue

        # (n_subj, n_win, window, n_ch)
        stacked = np.stack([arr[:, s:s + window, :] for s in starts], axis=1)
        n_win = stacked.shape[1]

        for subj in range(n_subjects):
            X_parts.append(stacked[subj].astype(np.float32))
            y_parts.append(np.full(n_win, y_clip[subj, clip], dtype=np.float32))
            subj_parts.append(np.full(n_win, subj, dtype=np.int64))

        if cfg.VERBOSE:
            print(f"  clip {clip:2d}: T={T:6d} -> {n_win:4d} windows x {n_subjects} subjects")

    X = np.concatenate(X_parts, axis=0)
    y_raw = np.concatenate(y_parts, axis=0)
    subject = np.concatenate(subj_parts, axis=0)

    print(f"  X={X.shape}  y={y_raw.shape}  ({time.time() - t0:.1f}s, {X.nbytes / 1e9:.2f} GB)")

    # --- labels ---
    if cfg.BINARY:
        y = (y_raw > cfg.THRESHOLD).astype(np.int64)
        n0, n1 = int((y == 0).sum()), int((y == 1).sum())
        print(f"  binary: class0={n0:,}  class1={n1:,}  "
              f"majority={max(n0, n1) / len(y):.4f}")
    else:
        y = (y_raw - 1).astype(np.int64)
        print(f"  multiclass counts: {np.bincount(y)}")

    # --- normalisation ---
    if cfg.NORMALIZE == "zscore":
        mu = X.mean(axis=(0, 1), keepdims=True)
        sd = X.std(axis=(0, 1), keepdims=True) + 1e-6
        X = (X - mu) / sd
        print("  z-score applied (per channel, global)")
    elif cfg.NORMALIZE == "zscore_subject":
        for s in range(n_subjects):
            m = subject == s
            if not m.any():
                continue
            mu = X[m].mean(axis=(0, 1), keepdims=True)
            sd = X[m].std(axis=(0, 1), keepdims=True) + 1e-6
            X[m] = (X[m] - mu) / sd
        print("  z-score applied (per channel, per subject)")

    return X, y, subject


# ----------------------------------------------------------------------
def split_indices(y: np.ndarray, subject: np.ndarray, cfg):
    """Return (train_idx, test_idx) according to cfg.SPLIT_MODE."""
    rng = np.random.RandomState(cfg.SEED)

    if cfg.SPLIT_MODE == "random":
        idx = rng.permutation(len(y))
        n_test = int(len(y) * cfg.TEST_RATIO)
        return idx[n_test:], idx[:n_test]

    if cfg.SPLIT_MODE == "subject":
        subjects = np.unique(subject)
        test_subjects = np.array(cfg.TEST_SUBJECTS)
        unknown = set(test_subjects) - set(subjects)
        if unknown:
            raise ValueError(f"TEST_SUBJECTS contains unknown ids: {sorted(unknown)}")
        test_mask = np.isin(subject, test_subjects)
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        print(f"  subject split: train subjects={sorted(set(subjects) - set(test_subjects))}, "
              f"test subjects={sorted(test_subjects)}")
        return train_idx, test_idx

    raise ValueError(f"unknown SPLIT_MODE {cfg.SPLIT_MODE!r} (expected random | subject)")


def make_loaders(cfg) -> tuple[DataLoader, DataLoader, dict]:
    """Build the dataset and return (train_loader, test_loader, info)."""
    print("--- building dataset ---")
    X, y, subject = build_arrays(cfg)
    train_idx, test_idx = split_indices(y, subject, cfg)

    Xtr = torch.from_numpy(X[train_idx])
    ytr = torch.from_numpy(y[train_idx])
    Xte = torch.from_numpy(X[test_idx])
    yte = torch.from_numpy(y[test_idx])

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    test_loader = DataLoader(
        TensorDataset(Xte, yte),
        batch_size=cfg.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    y_test = y[test_idx]
    majority = float(max((y_test == c).mean() for c in np.unique(y_test)))

    info = {
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_channels": X.shape[2],
        "window_len": X.shape[1],
        "n_classes": int(y.max()) + 1,
        "majority_baseline": majority,
    }
    print(f"  train={info['n_train']:,} ({len(train_loader)} batches)  "
          f"test={info['n_test']:,} ({len(test_loader)} batches)")
    print(f"  majority baseline on test = {majority:.4f}")

    return train_loader, test_loader, info


# ----------------------------------------------------------------------
def make_loaders_per_subject(cfg):
    """
    Subject-dependent protocol: yield one (train_loader, test_loader, info)
    pair per subject, where each subject's own windows are split 80/20.

    A separate model is meant to be trained for every subject and the
    resulting accuracies averaged -- this is the protocol most published
    DREAMER results use.

    Yields
    ------
    (subject_id, train_loader, test_loader, info)
    """
    print("--- building dataset (subject-dependent protocol) ---")
    X, y, subject = build_arrays(cfg)
    subjects = np.unique(subject)
    rng = np.random.RandomState(cfg.SEED)

    print(f"  {len(subjects)} subjects, splitting each {1 - cfg.TEST_RATIO:.0%}/"
          f"{cfg.TEST_RATIO:.0%} independently")

    for subj in subjects:
        mask = np.where(subject == subj)[0]
        perm = rng.permutation(len(mask))
        n_test = int(len(mask) * cfg.TEST_RATIO)
        te_local, tr_local = perm[:n_test], perm[n_test:]
        te_idx, tr_idx = mask[te_local], mask[tr_local]

        Xtr = torch.from_numpy(X[tr_idx]); ytr = torch.from_numpy(y[tr_idx])
        Xte = torch.from_numpy(X[te_idx]); yte = torch.from_numpy(y[te_idx])

        train_loader = DataLoader(
            TensorDataset(Xtr, ytr), batch_size=cfg.BATCH_SIZE, shuffle=True,
            num_workers=cfg.NUM_WORKERS, pin_memory=torch.cuda.is_available(),
            drop_last=len(tr_idx) > cfg.BATCH_SIZE,
        )
        test_loader = DataLoader(
            TensorDataset(Xte, yte), batch_size=cfg.BATCH_SIZE * 2, shuffle=False,
            num_workers=cfg.NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        )

        y_te = y[te_idx]
        majority = float(max((y_te == c).mean() for c in np.unique(y_te))) if len(y_te) else 0.0

        info = {
            "subject": int(subj),
            "n_train": len(tr_idx),
            "n_test": len(te_idx),
            "n_channels": X.shape[2],
            "window_len": X.shape[1],
            "n_classes": int(y.max()) + 1,
            "majority_baseline": majority,
        }
        yield int(subj), train_loader, test_loader, info
