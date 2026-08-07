# Training Runs

## 2026-08-07 · Smoke Test

Run directory: `runs/20260807_143512`

- Config: `configs/smoke.yaml`
- Device: RTX 3050 (CUDA)
- Precision: bf16
- Resume: No
- Best validation loss: 9.6161
- Checkpoint: `best.pt`

Resume from the latest checkpoint:

```bash
python train.py \
    --config configs/train.yaml \
    --resume runs/20260807_143512/latest.pt
```

Resume from the best checkpoint:

```bash
python train.py \
    --config configs/train.yaml \
    --resume runs/20260807_143512/best.pt
```

Notes:
- Resume functionality verified.
- AMP stable.
- No NaNs observed.
- Atomic checkpointing works.