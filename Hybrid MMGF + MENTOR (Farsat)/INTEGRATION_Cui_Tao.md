# Cui Tao Integration

The Farsat hybrid implementation now includes Cui Tao's sampled-softmax MENTOR changes.

Merged into `MMRec/src`:
- `models/mentor.py`: sampled-softmax ranking loss, multi-negative scoring, optional feature adapters, optional ID residual scoring, optional user graph loading, and chunked InfoNCE. The existing Farsat chunked KNN graph construction is preserved for Colab memory safety.
- `utils/dataloader.py`: K negatives per positive interaction, with optional content-kNN hard negatives disabled by default.
- `main.py`: CLI flags for sampled-softmax and ablation controls.
- `configs/model/MENTOR.yaml`: defaults keep original BPR behavior unless flags override them.
- `common/trainer.py`: epoch-aware negative sampling and optional checkpoint resume/save support.

Default behavior remains original MENTOR:

```bash
python main.py -m MENTOR -d baby
```

Cui Tao sampled-softmax behavior:

```bash
python main.py -m MENTOR -d baby --loss_type softmax --num_negatives 16 --ssm_temp 1.0
```

The existing `hybrid_static.py` MMGF + MENTOR blending flow remains available for combining trained MENTOR scores with MMGF scores.
