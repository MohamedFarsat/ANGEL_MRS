# Negative Sampling Ablations for MENTOR

This folder documents six negative-sampling directions tested on MENTOR for Amazon Baby.

These are experiment recipes, not six separate model forks. They all use the same implementation under `src/` and differ by CLI/config settings.

Final comparison protocol: select the best checkpoint by validation `Recall@20`, then report test metrics.

## Implementation Map

The code for these methods lives in the main MENTOR training pipeline:

| File | Role |
|---|---|
| `src/main.py` | Adds CLI flags for sampled softmax, temperature, dynamic negative selection, popularity sampling, checkpointing, and validation metric selection. |
| `src/utils_package/dataloader.py` | Samples multiple negatives per positive item and supports popularity-balanced negative sampling. |
| `src/models/mentor.py` | Computes sampled-softmax loss and implements hard, semi-hard, and mixed semi-hard/random negative selection. |
| `src/configs/model/MENTOR.yaml` | Defines default config values for the new negative-sampling options. |
| `src/common/trainer.py` | Saves and resumes checkpoints so long Colab runs can continue safely. |
| `src/configs/overall.yaml` | Uses `Recall@20` as the validation selection metric for these experiments. |
| `src/utils_package/metrics.py` | Keeps metric computation compatible with modern NumPy versions. |

## Shared Training Flow

All six variants keep the original MENTOR architecture and change only the negative-sampling and ranking-loss path.

1. `TrainDataLoader._sample_neg_ids()` samples one or more negative items for each `(user, positive_item)` interaction.
2. The training batch is expanded from `[user, positive, negative]` to `[user, positive, negative_1, ..., negative_K]`.
3. `MENTOR.forward()` computes the positive score using the usual MENTOR user-item representation.
4. `MENTOR.score_user_items_multi()` scores all sampled negatives as a `[K, batch_size]` score matrix.
5. `MENTOR.calculate_loss()` either uses original BPR or sampled softmax, depending on `loss_type`.
6. Dynamic variants select a subset from the sampled negative pool after scoring, so the choice of negatives changes as the model learns.

For sampled softmax, the logits for each training example are:

```text
[positive_score, negative_score_1, ..., negative_score_K]
```

The loss is:

```text
-log_softmax(logits / ssm_temp)[positive_index]
```

So the model is trained to assign higher probability to the clicked item than to the sampled unclicked items.

## Method Overview

| Order | Folder | Method | Simple Meaning |
|---:|---|---|---|
| 1 | `01_ssm16_baseline` | SSM-16 Baseline | Compare each clicked item against 16 randomly sampled unclicked items using sampled softmax. |
| 2 | `02_ssm16_temp05` | SSM-16 Baseline, Temp 0.5 | Keep 16 random negatives, but sharpen sampled softmax so high-scoring negatives receive more focus. |
| 3 | `03_dynamic_hard_negative_sampling` | Dynamic Hard Negative Sampling | Sample 64 negatives first, then train on the 16 negatives the model currently scores highest. |
| 4 | `04_semi_hard_negative_sampling` | Semi-hard Negative Sampling | Sample 64 negatives, then train on 16 high-scoring negatives that are still scored below the positive item. |
| 5 | `05_BEST_50pct_semi_hard_50pct_random` | 50% Semi-hard + 50% Random (BEST) | Use 8 semi-hard negatives plus 8 random negatives, balancing difficulty with stable random coverage. |
| 6 | `06_popularity_balanced_sampling` | Popularity-balanced Sampling | Replace some random negatives with negatives sampled from popular, medium, and tail item buckets. |

## Baby Results

MENTOR Baby baseline reference: R@10 `0.0658`, R@20 `0.1005`, N@10 `0.0348`, N@20 `0.0438`.

| Method | Source | Best Epoch | Valid R@20 | Test R@10 | Test R@20 | Test N@10 | Test N@20 |
|---|---|---:|---:|---:|---:|---:|---:|
| MENTOR Baseline | Colab | - | - | 0.0658 | 0.1005 | 0.0348 | 0.0438 |
| SSM-16 Baseline | Colab | 37 | 0.1023 | 0.0673 | 0.1032 | 0.0360 | 0.0453 |
| SSM-16 Baseline, Temp 0.5 | Colab | 16 | 0.0923 | 0.0615 | 0.0934 | 0.0328 | 0.0411 |
| Dynamic Hard Negative Sampling | Colab | 29 | 0.0994 | 0.0643 | 0.0987 | 0.0350 | 0.0439 |
| Semi-hard Negative Sampling | Colab | 30 | 0.1020 | 0.0675 | 0.1030 | 0.0363 | 0.0454 |
| 50% Semi-hard + 50% Random (BEST) | Colab | 20 | 0.1026 | 0.0677 | 0.1035 | 0.0362 | 0.0454 |
| Popularity-balanced Sampling | Colab | 24 | 0.1014 | 0.0658 | 0.1016 | 0.0356 | 0.0449 |

## Final Takeaway

The best direction was `50% Semi-hard + 50% Random`.

It achieved the highest validation `Recall@20` and improved test `Recall@20` from `0.1005` to `0.1035`, a `+2.99%` relative gain over the MENTOR Baby baseline.
