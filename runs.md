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





## M4.6-full-001

Commit: 9ed266f c993cc7415ccf20457b2fa6ad926c2340ac98562
Seed: 42
Parameters: 20,610,880

Batch size: 8
Gradient accumulation: 4
Effective batch size: 32

Best validation loss: ...
Final validation loss: ...

Average tokens/sec: ...

Peak GPU memory: ...

Checkpoint resume: verified

Observations:
- Training remained stable.
- No NaN losses.
- Validation loss decreased from ...
- ...