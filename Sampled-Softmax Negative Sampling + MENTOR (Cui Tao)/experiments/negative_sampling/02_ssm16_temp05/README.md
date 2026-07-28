# 02 - SSM-16 Baseline, Temp 0.5

## Simple Meaning

Keep the same 16 random negatives, but lower the sampled-softmax temperature so high-scoring negatives receive more focus.

## Motivation

Lower temperature makes the softmax distribution sharper. This increases pressure on negatives that the model currently finds confusing.

## Key Settings

| Setting | Value |
|---|---:|
| `loss_type` | `softmax` |
| `num_negatives` | `16` |
| `ssm_temp` | `0.5` |
| `dns_keep` | `0` |
| `hard_bpr_weight` | `0.0` |
| `pop_neg_ratio` | `0.0` |

## Implementation Details

This uses the same implementation path as `01_ssm16_baseline`.

The only change is:

```text
ssm_temp = 0.5
```

In `src/models/mentor.py`, the sampled-softmax logits are divided by `ssm_temp` before applying `log_softmax`:

```text
loss = -mean(log_softmax(logits / 0.5, dim=0)[0])
```

Lower temperature makes score differences larger before the softmax. A negative item with a high model score receives more loss pressure than it would under temperature `1.0`.

This does not explicitly mine hard negatives. The 16 negatives are still randomly sampled. The method only changes how strongly the softmax loss focuses on high-scoring negatives after they are sampled.

## Example Command

```bash
cd src
python3 -u main.py -d baby --use_gpu --epochs 400 --stopping_step 20 \
  --valid_metric Recall@20 \
  --hard_neg_ratio 0.0 \
  --loss_type softmax --num_negatives 16 --ssm_temp 0.5 \
  --dns_keep 0 --hard_bpr_weight 0.0 --pop_neg_ratio 0.0 \
  --checkpoint_interval 3 --checkpoint_run_id baby-ssm-k16-temp05
```

## Baby Result

Selected by validation `Recall@20`. This was a negative ablation: sharpening the loss hurt performance.

| Best Epoch | Valid R@20 | Test R@10 | Test R@20 | Test N@10 | Test N@20 |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.0923 | 0.0615 | 0.0934 | 0.0328 | 0.0411 |
