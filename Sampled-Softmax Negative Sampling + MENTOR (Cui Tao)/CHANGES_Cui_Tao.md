# MENTOR — Modifications by Cui Tao

This folder contains the files I modified from the original **MENTOR** (AAAI 2025)
codebase. Each file mirrors its location in the original repo (`src/...`), so it can
be dropped in over the corresponding original file.

The **main contribution** is a change to the training objective: replacing the pairwise
BPR loss with a **sampled-softmax loss over multiple random negatives**. The residual
adapters and ID-residual scoring are included as **ablations** (flag-gated, off by default).

Everything is controlled by command-line flags / config values; with all flags at their
defaults the model reproduces the original MENTOR (BPR, 1 negative).

---

## Main change — Sampled-softmax ranking loss with multi-negative sampling

Original MENTOR trains the ranking objective with **BPR** and **one** uniformly-random
negative per interaction. I replaced this with a **sampled-softmax** loss over **K
uniformly-random negatives** (default K = 16). Negatives are still drawn uniformly at
random with the user's interacted items filtered out — only the *number* of negatives
(1 → K) and the *loss form* (pairwise sigmoid → softmax over 1 positive + K negatives)
change. The modality fusion, GCN towers, item–item graph, and self-supervised losses are
untouched.

Motivation: with a single random negative the ranking task is usually trivially easy
(a random item is obviously wrong → vanishing gradient). A softmax over K negatives
provides a richer training signal and implicitly concentrates gradient on the
hardest negatives *within the random pool*, without explicitly mining hard negatives.

**Run it:**
```bash
python main.py -d baby --loss_type softmax --num_negatives 16 --ssm_temp 1.0 --hard_neg_ratio 0.0
```

**Baseline (original MENTOR, BPR):**
```bash
python main.py -d baby --loss_type bpr --num_negatives 1 --hard_neg_ratio 0.0
```

### Files changed for the main contribution
- **`src/models/mentor.py`**
  - `score_user_items_multi(...)` — vectorised scoring of a positive against K negatives.
  - `calculate_loss(...)` — sampled-softmax ranking term (`loss_type='softmax'`), falls
    back to the original BPR term when `loss_type='bpr'`.
- **`src/utils_package/dataloader.py`**
  - `_sample_neg_ids(...)` — returns `[num_negatives, batch]` (K independent random draws
    per interaction) instead of a single negative row; history filtering unchanged.
- **`src/main.py`** — CLI flags: `--loss_type {bpr,softmax}`, `--num_negatives`, `--ssm_temp`.
- **`src/configs/model/MENTOR.yaml`** — defaults: `loss_type: bpr`, `num_negatives: 1`,
  `ssm_temp: 1.0` (so defaults = original MENTOR).

> **Note on loss scale (honest caveat):** the sampled-softmax term has a larger natural
> magnitude (~log(K+1)) than the BPR term, so with the self-supervised loss weights held
> fixed, the SSL/alignment/regularization losses are effectively down-weighted *relative*
> to the ranking loss. A scale-normalised control run would isolate the ranking-signal
> effect from this incidental reweighting.

---

## Ablation A — Content-guided hard negative sampling (negative result)

`--hard_neg_ratio r` draws a fraction `r` of negatives from the positive item's content
kNN lookalikes (the model's `mm_adj` graph) instead of at random, with a warmup/ramp
schedule. **This degraded performance monotonically with `r`** and is not used in the
final method (`hard_neg_ratio: 0.0`).

Reason (documented for completeness): in these e-commerce datasets, content-similar items
are substitutes, so content lookalikes the user has not interacted with are
disproportionately **false negatives** (unobserved positives). Sampling them as negatives
penalises likely-positives. The final method therefore uses uniform-random negatives.

- **`src/utils_package/dataloader.py`** — `_sample_hard_negative(...)`,
  `_load_item_hard_neighbors(...)`, curriculum schedule.
- **`src/main.py`** — `--hard_neg_ratio`.

## Ablation B — Parameter-efficient residual feature adapters

`--use_feature_adapter` inserts a small **zero-initialised residual adapter**
(LayerNorm → down-projection → GELU → up-projection, ~1.3% added params) between the
frozen pretrained image/text features and the rest of the model, making the fixed
features recommendation-aware without fine-tuning the encoders. Zero-init means the
adapter starts as an exact identity and only drifts from the original features when it
reduces the loss (stable adaptation).

- **`src/models/mentor.py`** — `ResidualFeatureAdapter`, `adapted_visual_features()`,
  `adapted_text_features()`.
- **`src/configs/model/MENTOR.yaml`** — `use_feature_adapter` (default `False`),
  `feature_adapter_dim/dropout/alpha`.
- Optional: `--mm_adj_refresh_interval N` rebuilds the item kNN graph from *adapted*
  features every N epochs (in-memory; never overwrites the cached graph file).

## Ablation C — Behaviour-aware ID residual scoring

`--use_id_residual` adds a small, bounded, learnable fraction of the ID (collaborative)
tower's score to the final prediction, letting the pure interaction signal contribute
directly at inference (in the original MENTOR the ID tower is used only as a training-time
alignment target).

- **`src/models/mentor.py`** — `id_residual_scale()`, applied in
  `score_user_item_pairs(...)` and `full_sort_predict(...)`.
- **`src/configs/model/MENTOR.yaml`** — `use_id_residual` (default `False`).

---

## Supporting / infrastructure changes

- **`src/common/trainer.py`**
  - Checkpoint **save/resume** (`--resume_checkpoint`, `--checkpoint_interval`,
    `--checkpoint_run_id`) for long/interrupted runs.
  - `torch.load(..., weights_only=False)` for PyTorch ≥ 2.6 compatibility when resuming.
- **`src/models/mentor.py`**
  - Memory-efficient **chunked InfoNCE** (`_infonce_denominator`, gradient-checkpointed) —
    mathematically identical to the original contrastive loss, but computes the N×N
    similarity denominator in row-blocks so larger datasets (Sports/Clothing) fit in GPU
    memory.
  - **Optional user-graph** loading — makes the (unused) `user_graph_dict` file optional so
    datasets lacking it can still run; its outputs are never used in `forward()`.
- **`src/main.py`** — `--seed`, `--use_gpu`/`--cpu` overrides.

## Flag summary

| Flag | Default | Effect |
|------|---------|--------|
| `--loss_type {bpr,softmax}` | `bpr` | ranking loss (softmax = main contribution) |
| `--num_negatives K` | `1` | negatives per interaction |
| `--ssm_temp T` | `1.0` | sampled-softmax temperature |
| `--hard_neg_ratio r` | `0.0` | fraction of content-kNN hard negatives (Ablation A) |
| `--use_feature_adapter` | off | residual feature adapters (Ablation B) |
| `--mm_adj_refresh_interval N` | `0` | rebuild item graph from adapted features |
| `--use_id_residual` | off | ID residual scoring (Ablation C) |

With every flag at its default, this code reproduces the original MENTOR.
