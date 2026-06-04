# Auralis Memory Kernel Prototype

This is a separate experiment for the later idea:

```text
chat JSON -> compact memories -> adapter training pairs -> User-DoRA/LoRA
```

It does not touch the current pretraining or model code.

## Why

Raw chats should not be trained directly. They contain temporary status, wrong
intermediate assumptions, repeated text, and debugging noise. The prototype
extracts only compact memories and marks each item as either trainable or
non-trainable.

## Run

```bash
python scripts/experimental/memory_kernel.py smoke \
  --output-dir data/eval/memory_kernel_smoke
```

Outputs:

- `memories.jsonl`: structured memory objects
- `adapter_train.jsonl`: tiny SFT-style examples for future LoRA/DoRA tests
- `kernel_memory.txt`: special-token text blocks for tokenizer/pretraining experiments
- `report.md`: short extraction report

## Rule

Persistent preferences and stable project facts may become adapter data.
Temporary status and exact paths stay in JSON/memory storage and should not be
burned into an adapter.

