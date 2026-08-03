# S4-EEG

Modular implementation of S4 (Structured State Space) with an optional
cross-channel Theta reparameterisation, benchmarked on the DREAMER EEG dataset.

## Structure

```
config.py              all hyperparameters
main.py                entry point
models/
  hippo.py             HiPPO-LegS matrices + NPLR decomposition
  s4_layer.py          S4Layer with theta_mode: none | const | order
  s4_model.py          S4Block + S4Model
  baselines.py         LSTM / Transformer baselines
data/
  dreamer.py           loading, baseline removal, windowing, splits
training/
  trainer.py           training loop + history
  plots.py             training curves + run comparison
```

## Theta modes

| mode    | description                                              |
|---------|----------------------------------------------------------|
| `none`  | standard S4, internal channels independent                |
| `const` | one learned mixing matrix shared across Legendre orders   |
| `order` | `Theta^(n) = expm(n*G)`, a different mixing per order     |

## Usage

```bash
python main.py --model s4 --theta order --epochs 80 --target valence
python main.py --model lstm --epochs 40
python main.py --model s4 --theta none --split subject
```

## Data layout

`DATA_DIR` must contain:

```
labels.pkl              (n_subjects, n_clips, 3)
baseline_data.pkl       (n_subjects, n_clips, T_b, n_channels)
stimuli_{i}_clip.pkl    (n_subjects, T_i, n_channels)
```
