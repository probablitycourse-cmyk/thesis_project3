from .hippo import hippo_legs_matrices, nplr_decompose, verify_nplr
from .s4_layer import S4Layer, THETA_MODES
from .s4_model import S4Block, S4Model
from .baselines import LSTMModel, TransformerModel
from .eeg_components import (
    MultiBandS4Layer,
    SpatialFilter,
    EEG_BANDS,
    EMOTIV_REGIONS,
    EMOTIV_CHANNELS,
    band_to_dt_range,
)
from .eeg_model import EEGS4Block, EEGS4Model
from .ssm_state import SSMState, ssm_param_names

__all__ = [
    "hippo_legs_matrices", "nplr_decompose", "verify_nplr",
    "S4Layer", "THETA_MODES", "S4Block", "S4Model",
    "LSTMModel", "TransformerModel",
    "MultiBandS4Layer", "SpatialFilter", "EEG_BANDS",
    "EMOTIV_REGIONS", "EMOTIV_CHANNELS", "band_to_dt_range",
    "EEGS4Block", "EEGS4Model", "SSMState", "ssm_param_names",
    "build_model",
]


def build_model(cfg, input_dim: int, out_dim: int):
    """Factory that maps cfg.MODEL to a concrete model instance."""
    name = cfg.MODEL.lower()

    if name == "s4":
        return S4Model(
            input_dim=input_dim,
            H=cfg.H,
            N=cfg.N,
            num_layers=cfg.NUM_LAYERS,
            out_dim=out_dim,
            theta_mode=cfg.THETA_MODE,
            dropout=cfg.DROPOUT,
            pooling=cfg.POOLING,
            train_ssm_state=getattr(cfg, "TRAIN_SSM_STATE", False),
        )

    if name in ("eeg_s4", "eegs4"):
        model = EEGS4Model(
            input_dim=input_dim,
            H=cfg.H,
            N=cfg.N,
            num_layers=cfg.NUM_LAYERS,
            out_dim=out_dim,
            fs=float(cfg.FS),
            use_spatial=getattr(cfg, "USE_SPATIAL", True),
            use_multiband=getattr(cfg, "USE_MULTIBAND", True),
            use_gating=getattr(cfg, "USE_GATING", False),
            theta_mode=cfg.THETA_MODE,
            dropout=cfg.DROPOUT,
            pooling=cfg.POOLING,
            train_ssm_state=getattr(cfg, "TRAIN_SSM_STATE", False),
        )
        print(model.describe())
        return model

    if name == "lstm":
        return LSTMModel(
            input_dim=input_dim,
            H=cfg.H,
            num_layers=cfg.NUM_LAYERS,
            out_dim=out_dim,
            dropout=cfg.DROPOUT,
        )

    if name == "transformer":
        return TransformerModel(
            input_dim=input_dim,
            H=cfg.H,
            num_layers=cfg.NUM_LAYERS,
            num_heads=cfg.NUM_HEADS,
            out_dim=out_dim,
            dropout=cfg.DROPOUT,
        )

    raise ValueError(f"unknown MODEL {cfg.MODEL!r} (expected s4 | eeg_s4 | lstm | transformer)")
