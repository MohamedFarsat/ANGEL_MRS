# 06 - Popularity-balanced Sampling

## Simple Meaning

Replace some random negatives with items sampled from popularity buckets: popular, medium, and tail items.

## Motivation

Uniform random negatives can be obscure items that the user was unlikely to click anyway. Popular unclicked items are often more realistic negatives because the user was more likely to have been exposed to them.

## Key Settings

| Setting | Value |
|---|---:|
| `loss_type` | `softmax` |
| `num_negatives` | `64` |
| `pop_neg_ratio` | `0.5` |
| `pop_head_weight` | `0.4` |
| `pop_mid_weight` | `0.4` |
| `pop_tail_weight` | `0.2` |
| `dns_keep` | `0` |
| `ssm_temp` | `1.0` |

## Implementation Details

This method changes where negatives come from before the loss is computed.

In `src/utils_package/dataloader.py`, item popularity is computed from training-set interaction counts. Items are split into three buckets:

```text
head = top 20% most interacted items
mid  = middle 60%
tail = bottom 20%
```

With `pop_neg_ratio=0.5`, each negative draw has a 50% chance of coming from a popularity bucket instead of uniform random sampling.

The bucket weights are:

```text
head: 0.4
mid:  0.4
tail: 0.2
```

So the method samples more negatives from popular and medium-popularity items, while still keeping some tail coverage.

Because `dns_keep=0`, there is no dynamic hard-negative selection after scoring. The sampled negatives go directly into the sampled-softmax loss.

The final loss is sampled softmax over:

```text
1 positive + 64 popularity-balanced/random negatives
```

## Example Command

```bash
cd src
python3 -u main.py -d baby --use_gpu --epochs 400 --stopping_step 20 \
  --valid_metric Recall@20 \
  --hard_neg_ratio 0.0 \
  --loss_type softmax --num_negatives 64 --ssm_temp 1.0 \
  --dns_keep 0 \
  --pop_neg_ratio 0.5 --pop_head_weight 0.4 --pop_mid_weight 0.4 --pop_tail_weight 0.2 \
  --hard_bpr_weight 0.0 \
  --checkpoint_interval 3 --checkpoint_run_id baby-ssm-k64-pop50-h4m4t2
```

## Baby Result

Selected by validation `Recall@20`.

| Best Epoch | Valid R@20 | Test R@10 | Test R@20 | Test N@10 | Test N@20 |
|---:|---:|---:|---:|---:|---:|
| 24 | 0.1014 | 0.0658 | 0.1016 | 0.0356 | 0.0449 |
