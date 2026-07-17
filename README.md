
# Hybrid MMGF + MENTOR (Farsat)

The combined implementation is in `Hybrid MMGF + MENTOR (Farsat)/`.

It merges:
- Hybrid MMGF + MENTOR static blending from the Farsat implementation.
- Sampled-softmax negative sampling + MENTOR from the Cui Tao implementation.

Colab entrypoint:

```bash
cd "Hybrid MMGF + MENTOR (Farsat)/MMRec/src"
python main.py -m MENTOR -d baby --loss_type softmax --num_negatives 16 --ssm_temp 1.0
```
