# 03 - Dynamic Hard Negative Sampling

## Simple Meaning

Sample 64 negatives first, then train on the 16 negatives the model currently scores highest.

## Motivation

Random negatives are often too easy. This method tries to train on harder negatives by asking the model which sampled negatives it currently confuses with the positive item.

## Key Settings

| Setting | Value |
|---|---:|
| `loss_type` | `softmax` |
| `num_negatives` | `64` |
| `dns_strategy` | `hard` |
| `dns_keep` | `16` |
| `dns_random_keep` | `0` |
| `ssm_temp` | `1.0` |
| `pop_neg_ratio` | `0.0` |

## Implementation Details

This is dynamic because hard negatives are selected using the model's current scores during training.

The dataloader first samples a pool of 64 random negatives:

```text
interaction[2:] = 64 candidate negative rows
```

Then `MENTOR.calculate_loss()` scores all 64 negatives:

```text
all_neg_scores = score_user_items_multi(user, neg_ids_matrix)
```

With `dns_strategy=hard` and `dns_keep=16`, the model keeps the 16 highest-scoring negatives for each user in the batch:

```text
top_idx = all_neg_scores.topk(16, dim=0).indices
selected_neg_scores = all_neg_scores.gather(0, top_idx)
```

The final sampled-softmax loss is computed over:

```text
1 positive + 16 selected hard negatives
```

The key difference from random SSM-16 is that the final negatives are not just random. They are the negatives the current model finds most confusing within the 64-negative candidate pool.

## Example Command

```bash
cd src
python3 -u main.py -d baby --use_gpu --epochs 400 --stopping_step 20 \
  --valid_metric Recall@20 \
  --hard_neg_ratio 0.0 \
  --loss_type softmax --num_negatives 64 --ssm_temp 1.0 \
  --dns_strategy hard --dns_keep 16 --dns_random_keep 0 \
  --hard_bpr_weight 0.0 --pop_neg_ratio 0.0 \
  --checkpoint_interval 3 --checkpoint_run_id baby-ssm-k64-dns16
```

## Baby Result

Selected by validation `Recall@20`. This underperformed the SSM-16 baseline, likely because the highest-scoring negatives can include false negatives or substitute products.

| Best Epoch | Valid R@20 | Test R@10 | Test R@20 | Test N@10 | Test N@20 |
|---:|---:|---:|---:|---:|---:|
| 29 | 0.0994 | 0.0643 | 0.0987 | 0.0350 | 0.0439 |
