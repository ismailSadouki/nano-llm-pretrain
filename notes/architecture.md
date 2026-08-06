# GPT Architecture

This project implements a decoder-only Transformer from scratch in PyTorch with grouped-query attention (GQA), rotary positional embeddings (RoPE), KV-cache inference, RMSNorm and configurable sampling.

## Model configuration

| Component | Value |
|-----------|------:|
| Vocabulary | 16,000 |
| Context length | 1024 |
| Decoder blocks | 12 |
| Hidden dimension | 768 |
| Query heads | 12 |
| KV heads | 4 |
| Head dimension | 64 |
| FFN multiplier | 4 |
| Normalization | RMSNorm |
| Gated activations | SwiGLU |
| Positional encoding | RoPE |
| Attention | Grouped Query Attention |
| Inference | KV Cache |
| Sampling | Greedy / Temperature / Top-k / Top-p |

---

# Tensor Shapes

Input IDs
```
[B, T]
```

↓

Token Embeddings

```
[B, T, C]
```

↓

12 × Decoder Blocks

```
[B, T, C]
```

↓

Final RMSNorm

```
[B, T, C]
```

↓

LM Head

```
[B, T, V]
```

where

- B = batch size
- T = sequence length
- C = hidden dimension (768)
- V = vocabulary size (16000)

---

# Decoder Block

```
                 Residual
                    │
                    ▼
Input ──► RMSNorm ─► Attention (GQA + RoPE)
  │                 │
  └─────────────────┘
          │
          ▼
      RMSNorm
          │
          ▼
      FeedForward
          │
          ▼
     Residual Add
          │
          ▼
       Output
```

---

# Attention

```
Input
  │
  ├──► Q Projection
  ├──► K Projection
  └──► V Projection

          │
          ▼
        RoPE

          │
          ▼
Grouped Query Attention

          │
          ▼
     Output Projection
```

---

# Generation Pipeline
```
Prompt
   │
   ▼
Allocate KV caches
   │
   ▼
Forward(prompt)
   │
   ├── Compute logits
   └── Populate KV caches
   │
   ▼
──────────────────────────────────────────
Generation Loop
──────────────────────────────────────────
   │
   ▼
Take last-token logits
   │
   ▼
Temperature / Top-k / Top-p
   │
   ▼
Select next token
   │
   ▼
Append token to output
   │
   ▼
Forward(new token only)
   │
   ├── Reuse cached K,V
   ├── Compute new Q,K,V
   └── Update KV caches
   │
   ▼
Repeat until:
   • EOS token
   • max_new_tokens
   • context window full
   ```
# Validation

The implementation is validated using:

- Shape tests
- Decoder tests
- Attention tests
- Naive vs SDPA equivalence
- Causal masking tests
- KV cache equivalence tests
- Tiny overfit test
- Optimizer and backward tests

All validation tests pass before training.