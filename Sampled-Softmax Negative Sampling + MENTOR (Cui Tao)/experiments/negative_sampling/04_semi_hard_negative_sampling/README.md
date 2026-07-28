# 04 - Semi-hard Negative Sampling

## Simple Meaning

Sample 64 negatives, then train on 16 high-scoring negatives that are still scored below the positive item.

## Motivation

Pure hard mining can select negatives that are too ambiguous. Semi-hard mining keeps negatives that are informative but avoids the most suspicious cases where a negative scores above the clicked item.

## Key Settings

| Setting | Value |
|---|---:|
| `loss_type` | `softmax` |
| `num_negatives` | `64` |
| `dns_strategy` | `semi_hard` |
| `dns_keep` | `16` |
| `dns_random_keep` | `0` |
| `ssm_temp` | `1.0` |
| `pop_neg_ratio` | `0.0` |

## Implementation Details

This is a safer version of dynamic hard negative sampling.

The dataloader still samples 64 candidate negatives for each positive interaction. The model scores all 64, but it only prefers negatives that satisfy:

```text
negative_score < positive_score
```

In `src/models/mentor.py`, this is implemented with a semi-hard mask:

```text
semi_mask = sel_scores < pos_scores.unsqueeze(0)
semi_scores = sel_scores.masked_fill(~semi_mask, -inf)
top_idx = semi_scores.topk(16, dim=0).indices
```

So the selected negatives are difficult, but not so difficult that the model currently ranks them above the clicked item.

If a batch example has fewer than 16 semi-hard negatives, the code falls back to ordinary hardest negatives for the missing slots. This keeps the tensor shape fixed and avoids dropping training examples.

The final loss is still sampled softmax over:

```text
1 positive + 16 selected semi-hard negatives
```

## Example Command

```bash
cd src
python3 -u main.py -d baby --use_gpu --epochs 400 --stopping_step 20 \
  --valid_metric Recall@20 \
  --hard_neg_ratio 0.0 \
  --loss_type softmax --num_negatives 64 --ssm_temp 1.0 \
  --dns_strategy semi_hard --dns_keep 16 --dns_random_keep 0 \
  --hard_bpr_weight 0.0 --pop_neg_ratio 0.0 \
  --checkpoint_interval 3 --checkpoint_run_id baby-ssm-k64-semihard16
```

## Baby Result

Selected by validation `Recall@20`.

| Best Epoch | Valid R@20 | Test R@10 | Test R@20 | Test N@10 | Test N@20 |
|---:|---:|---:|---:|---:|---:|
| 30 | 0.1020 | 0.0675 | 0.1030 | 0.0363 | 0.0454 |
