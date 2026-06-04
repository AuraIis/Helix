# Model Weights License — Helix v2 / Auralis

The **trained model weights** (checkpoints + the `helix_v2_tokenizer.model`) are
released under the **BigScience OpenRAIL-M** license — a "Responsible AI License".

> Canonical license text (authoritative): https://www.licenses.ai/ai-licenses
> (BigScience OpenRAIL-M). The weights are distributed under that document; this
> file is a pointer + summary, the canonical text governs.

## Why a separate license for the weights
The repository **code** is Apache-2.0 (see `LICENSE`) — fully open and permissive.
The **weights** are released under OpenRAIL-M instead, because the rights holder
wants the model to remain broadly and commercially usable **while** carrying the
standard responsible-use conditions.

## In plain terms
- ✅ **Permitted:** broad use, including commercial use, fine-tuning, redistribution
  and creation of derivatives — provided downstream users are bound by the same
  use restrictions.
- 🚫 **Not permitted:** the categories of harmful application listed in **Attachment A
  ("Use Restrictions") of the OpenRAIL-M license** (the canonical document defines
  the exact list). The rights holder explicitly intends these restrictions to apply.
- 📎 You must pass these use restrictions through to anyone you share the model or a
  derivative with (Section 5 / Attachment A of OpenRAIL-M).

## Attribution
Helix v2 / Auralis — a from-scratch ~0.9B hybrid (Mamba-2 / GLA / Sparse-Attention)
language model. See `docs/PROJEKT_STAND.md` for the full project context and the
honest status of what the model can and cannot do.

> Note: the weights are **not** OSI "open source" in the strict sense, because the
> OpenRAIL-M use restrictions intentionally limit certain applications. This is a
> deliberate choice by the rights holder. The repository **code** remains OSI-open
> under Apache-2.0.
