# Knowledge Kernel Experiment

This tests the “Duden/DNA” idea separately from the main pipeline.

The experiment creates two corpora with the same facts:

- `plain_corpus.txt`: normal prose baseline
- `current_kernel.txt`: structured blocks using current tokenizer tags
- `future_kernel.txt`: cleaner future format with dedicated definition/fact tags
- `qa_eval.jsonl`: probes to test whether the facts were learned

Run:

```bash
python scripts/experimental/knowledge_kernel.py sample \
  --output-dir data/eval/knowledge_kernel_smoke
```

The current tokenizer already has `<memory>` and `</memory>`, so the current
kernel can be tested now. Dedicated tags such as `<|definition|>` should only be
added when we intentionally retrain the tokenizer and all checkpoints from
scratch.

