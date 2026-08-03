"""
Central configuration. Every hyperparameter lives here.

From Colab you can override any field after importing, e.g.

    from config import CFG
    CFG.MODEL = "s4"
    CFG.THETA_MODE = "order"
    CFG.NUM_EPOCHS = 80
"""

from __future__ import annotations


class CFG:
    # ---------------- data ----------------
    DATA_DIR        = "/content/data/EEG_data"
    TARGET          = "valence"        # valence | arousal | dominance
    BINARY          = True
    THRESHOLD       = 3.0              # label > THRESHOLD -> class 1
    FS              = 128              # sampling rate (Hz)
    WINDOW_SEC      = 1.0
    STRIDE_SEC      = 1.0              # == WINDOW_SEC means no overlap
    USE_BASELINE    = True             # subtract per-(subject, clip) baseline mean
    NORMALIZE       = "zscore"         # zscore | zscore_subject | none

    # ---------------- split ----------------
    SPLIT_MODE      = "random"         # random | subject
    TEST_RATIO      = 0.2              # used when SPLIT_MODE == "random"
    TEST_SUBJECTS   = [18, 19, 20, 21, 22]   # used when SPLIT_MODE == "subject"

    # ---------------- model ----------------
    MODEL           = "s4"             # s4 | eeg_s4 | lstm | transformer
    THETA_MODE      = "none"           # none | const | order   (s4 only)

    # --- eeg_s4 components (each ablatable independently) ---
    USE_SPATIAL     = True             # region-aware channel projection
    USE_MULTIBAND   = True             # per-band Delta ranges
    USE_GATING      = False            # Mamba-style multiplicative gate

    # --- SSM state parameters (Lambda, q) ---
    # The official S4 trains these; HiPPO only initialises them. Re(Lambda) is
    # kept negative by a softplus reparameterisation so stability is preserved.
    TRAIN_SSM_STATE = False
    LR_STATE_MULT   = 0.01             # lr multiplier for Lambda / q
    H               = 64               # internal channel width
    N               = 64               # SSM state dimension    (s4 only)
    NUM_LAYERS      = 4
    NUM_HEADS       = 4                # transformer only
    DROPOUT         = 0.1
    POOLING         = "mean"           # mean | last | max      (s4 only)

    # ---------------- training ----------------
    BATCH_SIZE      = 256
    NUM_EPOCHS      = 80
    LR              = 1e-3
    USE_SCHEDULER   = False            # False -> constant lr; True -> cosine annealing
    LR_SSM_MULT     = 0.1              # multiplier for log_dt
    LR_THETA_MULT   = 0.1              # multiplier for the Theta generator G
    WEIGHT_DECAY    = 0.0
    GRAD_CLIP       = 1.0
    EARLY_STOP_PATIENCE = 0            # 0 disables early stopping
    SEED            = 0

    # ---------------- runtime ----------------
    NUM_WORKERS     = 0                # 0 is fastest when data already sits in RAM
    PRINT_EVERY     = 1
    VERBOSE         = True
    OUTPUT_DIR      = "outputs"
    SAVE_BEST       = False

    # ------------------------------------------------------------------
    @classmethod
    def run_name(cls) -> str:
        parts = [cls.MODEL]
        if cls.MODEL.lower() == "s4":
            parts.append(cls.THETA_MODE)
        elif cls.MODEL.lower() in ("eeg_s4", "eegs4"):
            tag = "".join([
                "S" if cls.USE_SPATIAL else "",
                "M" if cls.USE_MULTIBAND else "",
                "G" if cls.USE_GATING else "",
            ])
            parts.append(tag if tag else "plain")
        parts += [cls.TARGET, f"H{cls.H}", f"N{cls.N}", f"L{cls.NUM_LAYERS}", cls.SPLIT_MODE]
        if getattr(cls, "TRAIN_SSM_STATE", False):
            parts.append("trainState")
        parts.append("cos" if cls.USE_SCHEDULER else "fixedlr")
        return "_".join(parts)

    @classmethod
    def as_dict(cls) -> dict:
        return {k: v for k, v in vars(cls).items()
                if not k.startswith("_") and not callable(v)
                and not isinstance(v, (classmethod, staticmethod))}

    @classmethod
    def show(cls) -> None:
        print("=" * 58)
        print(f"CONFIG  ({cls.run_name()})")
        print("=" * 58)
        for k, v in cls.as_dict().items():
            print(f"  {k:20s} = {v}")
        print("=" * 58)
