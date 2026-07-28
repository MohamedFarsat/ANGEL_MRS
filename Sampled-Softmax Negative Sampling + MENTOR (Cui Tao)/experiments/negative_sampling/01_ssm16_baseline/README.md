# 01 - SSM-16 Baseline

## Simple Meaning

For each clicked item, compare it against 16 randomly sampled unclicked items using sampled softmax.

## Motivation

Original BPR compares one positive item against one negative item. This variant gives the model a richer ranking signal by making the positive item compete against 16 negatives at once.

## Key Settings

| Setting | Value |
|---|---:|
| `loss_type` | `softmax` |
| `num_negatives` | `16` |
| `ssm_temp` | `1.0` |
| `dns_keep` | `0` |
| `hard_bpr_weight` | `0.0` |
| `pop_neg_ratio` | `0.0` |

## Implementation Details

This is the simplest sampled-softmax version.

The dataloader samples `num_negatives=16` negatives for every positive interaction. The batch passed into the model has this structure:

```text
interaction[0] = user ids
interaction[1] = positive item ids
interaction[2:] = 16 negative item-id rows
```

Inside `src/models/mentor.py`, `calculate_loss()` scores the clicked item once and scores the 16 negatives with `score_user_items_multi()`. It then builds a `17 x batch_size` logits matrix:

```text
row 0      = positive scores
rows 1-16 = negative scores
```

Because `dns_keep=0`, no hard-negative filtering is applied. All 16 negatives are random negatives from the dataloader.

The objective is sampled softmax:

```text
loss = -mean(log_softmax(logits / 1.0, dim=0)[0])
```

Compared with original BPR, this gives one positive item 16 competitors instead of only one competitor.

## Example Command

```bash
cd src
python3 -u main.py -d baby --use_gpu --epochs 400 --stopping_step 20 \
  --valid_metric Recall@20 \
  --hard_neg_ratio 0.0 \
  --loss_type softmax --num_negatives 16 --ssm_temp 1.0 \
  --dns_keep 0 --hard_bpr_weight 0.0 --pop_neg_ratio 0.0 \
  --checkpoint_interval 3 --checkpoint_run_id baby-ssm-k16-temp10
```

## Baby Result

Selected by validation `Recall@20`.

| Best Epoch | Valid R@20 | Test R@10 | Test R@20 | Test N@10 | Test N@20 |
|---:|---:|---:|---:|---:|---:|
| 37 | 0.1023 | 0.0673 | 0.1032 | 0.0360 | 0.0453 |
