---
name: PyCUDA GPT Training
overview: Build a NumPy+PyCUDA character GPT stack (tokenizer, model, CUDA ops, training loop, checkpointing) plus four CLIs—train.py, auto_train.py, generate.py, interactive.py—wired to setup/training_config.json, with CLI-gated logit/token/neuron tracing for Kepler GT 730.
todos:
  - id: cuda-env-kernels
    content: Add model/cuda env bootstrap + gemm/softmax/layernorm/gelu SourceModule kernels and ops wrappers
    status: pending
  - id: tokenizer-model
    content: Implement CharacterGPTTokenizer, GPTConfig, weight alloc/init, GPTModel forward (+ backward for train)
    status: pending
  - id: trace-context
    content: Implement TraceContext and CLI-gated dump helpers (tokens/logits/neurons/vectorization)
    status: pending
  - id: training-loop
    content: Implement dataset batches, AdamW, loss, checkpoint, probe
    status: pending
  - id: clis
    content: Ship train.py, auto_train.py, generate.py, interactive.py + shared argparse
    status: pending
  - id: verify-smoke
    content: Smoke-test tiny auto_train + traced generate on GT 730
    status: pending
isProject: false
---

`