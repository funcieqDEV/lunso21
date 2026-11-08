# lunso-2.1 specification

## Architecture

| Parameter | Value |
| --- | --- |
| Model type | Decoder-only |
| Vocabulary size | 49,152 tokens |
| Context length | 4,096 tokens |
| Number of decoder layers | 20 |
| Hidden size | 1,024 |
| Intermediate size | 4,192 |
| Attention heads | 16 |
| KV heads | 4 |
| Head dimension | 64 |
| Attention mechanism | GQA (Grouped-Query Attention) |
| Local attention | Causal sliding window of 1,024 tokens |
| Global attention | Full causal attention over the context |
| FFN | SwiGLU |
| Normalization | RMSNorm |
| Position encoding | RoPE |

### Layer layout

The five-layer pattern **Local, Local, Local, Local, Global** repeats four times:

| Layers | Attention |
| --- | --- |
| 1–4 | Local |
| 5 | Global |
| 6–9 | Local |
| 10 | Global |
| 11–14 | Local |
| 15 | Global |
| 16–19 | Local |
| 20 | Global |

Model flow: **token embeddings (1,024) → 20 decoder layers → final RMSNorm → LM head**. The LM head shares its weights with the token embeddings (tied weights).

### Parameter count

Assuming bias-free linear projections and two RMSNorms per decoder layer:

| Component | Parameters |
| --- | ---: |
| Token embeddings (tied to LM head) | 50,331,648 |
| Attention projections (20 layers) | 52,428,800 |
| SwiGLU FFN projections (20 layers) | 257,556,480 |
| RMSNorm scales (two per layer plus final) | 41,984 |
| **Total** | **360,358,912 (~360.4M)** |

RoPE adds no learned parameters. The count will change if biases or other learned components are added.

## Tokenization and training

| Parameter | Value |
| --- | --- |
| Tokenizer | Byte-level BPE |
| Precision | BF16 |
| Optimizer | Fused AdamW |
| Learning-rate schedule | WSD (warmup, stable, decay) |

### Training plan

- Initial pretraining target: **50 billion cumulative training tokens**.
- Continue training beyond 50 billion tokens if evaluation shows room for improvement and compute is available.
- Training will run across multiple Colab sessions, resuming from checkpoints rather than restarting.
- Checkpoints must preserve model weights, optimizer state, learning-rate scheduler state, cumulative token/step counts, and the data sampler and random-number-generator states needed for a faithful resume.
- The WSD decay phase should be planned around the actual intended stopping point; reaching 50 billion tokens does not automatically end training.
