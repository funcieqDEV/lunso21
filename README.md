# Lunso 2.1

A 360M-parameter decoder-only language model in development.

## Model at a glance

- 20 decoder layers with a repeating 4 local + 1 global attention pattern
- 4,096-token context and 1,024-token local attention window
- GQA, SwiGLU, RMSNorm, RoPE, and tied input/output embeddings
- Initial pretraining target: 50 billion tokens, with further training possible

See [SPEC.md](SPEC.md) for the full architecture and training notes.

> **Documentation note:** All project documents, excluding scripts, are AI-written and intended solely as notes for me, the developer.
