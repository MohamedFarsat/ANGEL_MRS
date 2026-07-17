# Recommendation Experiment Results Log

Last updated: 2026-07-10 09:10 UTC

## Run Settings

- Framework: local MMRec codebase with added `MENTOR`, `MMGF`, and `hybrid_static.py`.
- Dataset split protocol: MMRec dataset files with `x_label` train/valid/test split.
- Metrics: `Recall@10`, `Recall@20`, `NDCG@10`, `NDCG@20`.
- Seed: `999`.
- GPU runtime used for completed Baby runs: Google Colab Tesla T4.
- MENTOR config: first materialized hyperparameter values from `MENTOR.yaml`.
- Hybrid calibration: per-user z-score.
- Hybrid validation tuning metric: `Recall@20`.
- Hybrid weight grid: `0.00, 0.05, ..., 1.00`.

## Confirmed Baby Results

| Dataset | Model / Run | Validation Weight | R@10 | R@20 | N@10 | N@20 |
|---|---|---:|---:|---:|---:|---:|
| baby | MM-GF only, test | 0.00 | 0.0655 | 0.0976 | 0.0368 | 0.0451 |
| baby | MENTOR only, test | 1.00 | 0.0658 | 0.1005 | 0.0348 | 0.0438 |
| baby | MENTOR + MM-GF static hybrid, test | 0.50 | 0.0737 | 0.1095 | 0.0406 | 0.0498 |

## Baby Validation Details

### MENTOR Only

Best validation checkpoint:

| R@10 | R@20 | N@10 | N@20 |
|---:|---:|---:|---:|
| 0.0630 | 0.0981 | 0.0339 | 0.0428 |

Test:

| R@10 | R@20 | N@10 | N@20 |
|---:|---:|---:|---:|
| 0.0658 | 0.1005 | 0.0348 | 0.0438 |

### MM-GF Only

Validation:

| R@10 | R@20 | N@10 | N@20 |
|---:|---:|---:|---:|
| 0.0675 | 0.0984 | 0.0371 | 0.0450 |

Test:

| R@10 | R@20 | N@10 | N@20 |
|---:|---:|---:|---:|
| 0.0655 | 0.0976 | 0.0368 | 0.0451 |

### Static Hybrid

Best validation blend:

| Weight | R@10 | R@20 | N@10 | N@20 |
|---:|---:|---:|---:|---:|
| 0.50 | 0.0727 | 0.1094 | 0.0397 | 0.0490 |

Test:

| R@10 | R@20 | N@10 | N@20 |
|---:|---:|---:|---:|
| 0.0737 | 0.1095 | 0.0406 | 0.0498 |

Hybrid validation endpoints:

| Weight | Model Meaning | R@10 | R@20 | N@10 | N@20 |
|---:|---|---:|---:|---:|---:|
| 0.00 | MM-GF only | 0.0675 | 0.0984 | 0.0371 | 0.0450 |
| 1.00 | MENTOR only | 0.0635 | 0.0986 | 0.0340 | 0.0429 |

## Baby Density Buckets, Hybrid Test

| Bucket | R@10 | R@20 | N@10 | N@20 |
|---|---:|---:|---:|---:|
| overall | 0.0737 | 0.1095 | 0.0406 | 0.0498 |
| user<=5 | 0.0783 | 0.1113 | 0.0426 | 0.0509 |
| user6-20 | 0.0668 | 0.1075 | 0.0371 | 0.0477 |
| user>20 | 0.0499 | 0.0865 | 0.0382 | 0.0523 |
| item<=5 | 0.0109 | 0.0163 | 0.0066 | 0.0080 |
| item6-20 | 0.0185 | 0.0298 | 0.0092 | 0.0120 |
| item>20 | 0.1115 | 0.1645 | 0.0612 | 0.0748 |

## Sports / Clothing Status

- Local repo originally contained only `MMRec/data/baby`.
- Public MMRec Google Drive folder contains preprocessed `sports` and `clothing`.
- Sports and Clothing were downloaded into Colab during one session.
- Their folders did not include `user_graph_dict.npy`, which MENTOR requires.
- Generated `user_graph_dict.npy` in Colab using sparse train user co-occurrence:
  - sports: 35,598 users, 18,357 items, 296,337 interactions, train/valid/test `{0: 218409, 1: 37899, 2: 40029}`.
  - clothing: 39,387 users, 23,033 items, 278,677 interactions, train/valid/test `{0: 197338, 1: 40150, 2: 41189}`.
- Initial Sports MENTOR run hit CUDA OOM in full item-item KNN graph construction.
- Patched `MENTOR.get_knn_adj_mat()` to compute KNN in blocks (`knn_block_size`, default 1024), avoiding full dense similarity materialization.
- After patch, Sports MENTOR initialized and trained successfully through at least epoch 6 before Colab/tunnel reset.

Last confirmed Sports validation before losing the Colab runtime:

| Epoch | R@10 | R@20 | N@10 | N@20 |
|---:|---:|---:|---:|---:|
| 0 | 0.0328 | 0.0524 | 0.0172 | 0.0222 |
| 1 | 0.0388 | 0.0621 | 0.0204 | 0.0264 |
| 2 | 0.0434 | 0.0692 | 0.0230 | 0.0296 |
| 3 | 0.0471 | 0.0731 | 0.0249 | 0.0315 |
| 4 | 0.0499 | 0.0762 | 0.0264 | 0.0331 |
| 5 | 0.0517 | 0.0790 | 0.0275 | 0.0345 |
| 6 | 0.0526 | 0.0799 | 0.0281 | 0.0351 |

Final MENTOR-only Sports test result has now been completed and preserved. Clothing has not been completed yet.

## Confirmed Sports Results

### MENTOR Only

Best validation checkpoint:

| R@10 | R@20 | N@10 | N@20 |
|---:|---:|---:|---:|
| 0.0728 | 0.1098 | 0.0389 | 0.0483 |

Test:

| R@10 | R@20 | N@10 | N@20 |
|---:|---:|---:|---:|
| 0.0740 | 0.1122 | 0.0398 | 0.0497 |

## Active / Latest Sports Rerun

Current Sports-only rerun on Colab T4:

- Started: 2026-07-10 03:28 UTC.
- Remote log path: `/content/test_ali/MMRec/src/run_logs/mentor_only_sports.log`.
- Local autosynced log path: `colab_live_logs/mentor_only_sports.log`.
- Local autosync process: `colab_live_logs/autosync_colab_logs.ps1`, pulls the log every 120 seconds while the bore tunnel is alive.
- Completed: 2026-07-10 08:51 UTC.
- Early stopping: epoch 82.
- Final test: R@10=0.0740, R@20=0.1122, N@10=0.0398, N@20=0.0497.

Latest confirmed Sports validation from this rerun:

| Epoch | R@10 | R@20 | N@10 | N@20 |
|---:|---:|---:|---:|---:|
| 0 | 0.0328 | 0.0524 | 0.0172 | 0.0222 |
| 1 | 0.0388 | 0.0621 | 0.0205 | 0.0264 |
| 2 | 0.0435 | 0.0691 | 0.0230 | 0.0296 |
| 3 | 0.0472 | 0.0732 | 0.0250 | 0.0316 |
| 4 | 0.0497 | 0.0759 | 0.0264 | 0.0330 |
| 5 | 0.0517 | 0.0789 | 0.0275 | 0.0344 |
| 6 | 0.0526 | 0.0801 | 0.0281 | 0.0351 |
| 7 | 0.0548 | 0.0828 | 0.0289 | 0.0361 |
| 8 | 0.0560 | 0.0844 | 0.0296 | 0.0369 |
| 9 | 0.0572 | 0.0864 | 0.0302 | 0.0376 |
| 10 | 0.0581 | 0.0891 | 0.0309 | 0.0387 |
| 11 | 0.0594 | 0.0898 | 0.0314 | 0.0391 |

## Preserved Local Logs

- `hybrid_baby_mentor_mmgf_colab.log`
- `mentor_only_baby_colab.log`

## Active / Latest Clothing Run

Current Clothing-only MENTOR run on Colab T4:

- Restarted again: 2026-07-13 05:54 UTC on new Colab runtime.
- Remote log path: `/content/test_ali/MMRec/src/run_logs/mentor_only_clothing.log`.
- Local autosynced log path: `colab_live_logs/mentor_only_clothing.log`.
- Current Python PID: 3359 on Colab.
- GPU status at restart: Tesla T4, about 14.2 GiB VRAM during training.
- Current best validation as of 2026-07-13 08:54 UTC: epoch 79, R@10=0.0638, R@20=0.0942, N@10=0.0341, N@20=0.0418.
- Latest validation as of 2026-07-13 08:54 UTC: epoch 79, R@10=0.0638, R@20=0.0942, N@10=0.0341, N@20=0.0418.
- Rechecked from local files on 2026-07-14: no Clothing `Early stopping` or `MENTOR_ONLY_TEST` line was present locally; last Colab tunnel on port 25032 refused connections.
- No final test result yet.

Validation checkpoints for 2026-07-13 restarted run:

| Epoch | R@10 | R@20 | N@10 | N@20 |
|---:|---:|---:|---:|---:|
| 0 | 0.0255 | 0.0391 | 0.0133 | 0.0168 |
| 10 | 0.0472 | 0.0721 | 0.0252 | 0.0314 |
| 18 | 0.0526 | 0.0798 | 0.0282 | 0.0350 |
| 30 | 0.0577 | 0.0862 | 0.0311 | 0.0383 |
| 40 | 0.0592 | 0.0891 | 0.0319 | 0.0394 |
| 51 | 0.0614 | 0.0906 | 0.0329 | 0.0403 |
| 70 | 0.0631 | 0.0934 | 0.0338 | 0.0414 |
| 79 | 0.0638 | 0.0942 | 0.0341 | 0.0418 |

## Previous Clothing Runs

Previous Clothing-only MENTOR run on Colab T4:

- Restarted: 2026-07-11 09:20 UTC after the previous Colab runtime reset.
- Remote log path: `/content/test_ali/MMRec/src/run_logs/mentor_only_clothing.log`.
- Local autosynced log path: `colab_live_logs/mentor_only_clothing.log`.
- Current Python PID: 13505 on Colab.
- GPU status at restart: Tesla T4, about 14.2 GiB VRAM during training.
- Current best validation as of 2026-07-11 12:20 UTC: epoch 79, R@10=0.0638, R@20=0.0947, N@10=0.0341, N@20=0.0418.
- Latest validation as of 2026-07-11 12:20 UTC: epoch 84, R@10=0.0638, R@20=0.0941, N@10=0.0341, N@20=0.0417.
- No final test result yet.
- Previous interrupted run reached epoch 77 on validation but had no final test line.
- Previous interrupted best validation as of 2026-07-10 11:56 UTC: epoch 77, R@10=0.0634, R@20=0.0936, N@10=0.0339, N@20=0.0415.

Validation checkpoints for restarted run:

| Epoch | R@10 | R@20 | N@10 | N@20 |
|---:|---:|---:|---:|---:|
| 0 | 0.0253 | 0.0391 | 0.0133 | 0.0168 |
| 1 | 0.0296 | 0.0467 | 0.0155 | 0.0198 |
| 10 | 0.0474 | 0.0724 | 0.0252 | 0.0315 |
| 20 | 0.0534 | 0.0815 | 0.0288 | 0.0358 |
| 40 | 0.0594 | 0.0892 | 0.0318 | 0.0394 |
| 60 | 0.0615 | 0.0918 | 0.0331 | 0.0407 |
| 79 | 0.0638 | 0.0947 | 0.0341 | 0.0418 |
| 84 | 0.0638 | 0.0941 | 0.0341 | 0.0417 |

Validation checkpoints for previous interrupted run:

- Started: 2026-07-10 09:12 UTC.
- Remote log path: `/content/test_ali/MMRec/src/run_logs/mentor_only_clothing.log`.
- Drive synced log path: `/content/drive/MyDrive/mmrec_results/mentor_only_clothing.log`.
- Current best validation as of 2026-07-10 11:56 UTC: epoch 77, R@10=0.0634, R@20=0.0936, N@10=0.0339, N@20=0.0415.
- Latest validation as of 2026-07-10 11:56 UTC: epoch 77, R@10=0.0634, R@20=0.0936, N@10=0.0339, N@20=0.0415.
- No final test result yet.
- Rechecked from local files on 2026-07-11: no Clothing `Early stopping` or `MENTOR_ONLY_TEST` line was present locally; last Colab tunnel on port 47534 refused connections.

| Epoch | R@10 | R@20 | N@10 | N@20 |
|---:|---:|---:|---:|---:|
| 0 | 0.0255 | 0.0391 | 0.0134 | 0.0168 |
| 1 | 0.0297 | 0.0465 | 0.0156 | 0.0198 |
| 2 | 0.0337 | 0.0531 | 0.0179 | 0.0228 |
| 3 | 0.0372 | 0.0581 | 0.0197 | 0.0249 |
| 4 | 0.0391 | 0.0610 | 0.0207 | 0.0262 |
| 5 | 0.0415 | 0.0635 | 0.0220 | 0.0275 |
| 10 | 0.0473 | 0.0724 | 0.0252 | 0.0315 |
| 20 | 0.0535 | 0.0817 | 0.0288 | 0.0359 |
| 30 | 0.0575 | 0.0864 | 0.0310 | 0.0383 |
| 40 | 0.0592 | 0.0887 | 0.0318 | 0.0393 |
| 49 | 0.0604 | 0.0906 | 0.0328 | 0.0404 |
| 52 | 0.0614 | 0.0904 | 0.0332 | 0.0405 |
| 60 | 0.0618 | 0.0911 | 0.0332 | 0.0406 |
| 70 | 0.0632 | 0.0930 | 0.0337 | 0.0413 |
| 77 | 0.0634 | 0.0936 | 0.0339 | 0.0415 |
