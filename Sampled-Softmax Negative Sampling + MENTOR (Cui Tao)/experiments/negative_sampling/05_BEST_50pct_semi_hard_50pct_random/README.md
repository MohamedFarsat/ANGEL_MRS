# 05 - 50% Semi-hard + 50% Random (BEST)

## Simple Meaning

Use 8 semi-hard negatives plus 8 random negatives.

## Motivation

Semi-hard negatives give the model useful difficult examples. Random negatives keep the training signal stable and preserve broad item coverage. This balances difficulty and robustness better than using only hard or semi-hard negatives.

## Key Settings

| Setting | Value |
|---|---:|
| `loss_type` | `softmax` |
| `num_negatives` | `64` |
| `dns_strategy` | `mixed_semi_hard` |
| `dns_keep` | `8` |
| `dns_random_keep` | `8` |
| `ssm_temp` | `1.0` |
| `pop_neg_ratio` | `0.0` |

## Implementation Details

This method uses the same 64-negative candidate pool as the hard and semi-hard variants, but it does not train only on mined negatives.

The model first selects 8 semi-hard negatives:

```text
dns_strategy = mixed_semi_hard
dns_keep = 8
```

These are high-scoring negatives that are still below the positive score.

Then it adds 8 random negatives from the remaining candidate pool:

```text
dns_random_keep = 8
```

The final negative set has 16 negatives:

```text
8 semi-hard negatives + 8 random negatives
```

In `src/models/mentor.py`, the selected semi-hard scores and random scores are concatenated before sampled softmax:

```text
all_neg_scores = cat(picked_scores, random_scores)
logits = cat(positive_scores, all_neg_scores)
```

This is why the method is more stable than pure hard mining. Semi-hard negatives add difficulty, while random negatives preserve broad item coverage and reduce the risk of over-training on ambiguous substitutes.

## Example Command

```bash
cd src
python3 -u main.py -d baby --use_gpu --epochs 400 --stopping_step 20 \
  --valid_metric Recall@20 \
  --hard_neg_ratio 0.0 \
  --loss_type softmax --num_negatives 64 --ssm_temp 1.0 \
  --dns_strategy mixed_semi_hard --dns_keep 8 --dns_random_keep 8 \
  --hard_bpr_weight 0.0 --pop_neg_ratio 0.0 \
  --checkpoint_interval 3 --checkpoint_run_id baby-ssm-k64-mixsemi8-rand8
```

## Baby Result

Selected by validation `Recall@20`.

| Best Epoch | Valid R@20 | Test R@10 | Test R@20 | Test N@10 | Test N@20 |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.1026 | 0.0677 | 0.1035 | 0.0362 | 0.0454 |

## Improvement over MENTOR Baby Baseline

MENTOR Baby baseline: R@10 `0.0658`, R@20 `0.1005`, N@10 `0.0348`, N@20 `0.0438`.

| Metric | Baseline | This Method | Relative Gain |
|---|---:|---:|---:|
| R@10 | 0.0658 | 0.0677 | +2.89% |
| R@20 | 0.1005 | 0.1035 | +2.99% |
| N@10 | 0.0348 | 0.0362 | +4.02% |
| N@20 | 0.0438 | 0.0454 | +3.65% |
