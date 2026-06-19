# Design Note — Long-Context Exact-Recall for a Code-Focused Helix

**Status:** design-only / parked. NOT a retrofit for the current 0.9B. Revisit only when
scaling a larger (3B+), code-trained, long-context Helix. Premature today (the 0.9B is
undertrained, trained context is only 2048, and cannot yet code in *short* context).

## Why this matters (and only for code at scale)

Helix is a hybrid (Mamba-2 + GLA + windowed Sparse-Attention). Its structural weakness vs a
full-attention Transformer is **exact recall over long distances**: the recurrent state
compresses history into a fixed size, and that compression is lossy. The windowed attention
layers see *within their window* exactly, but tokens far outside the window are only present
in the compressed state.

For prose this is forgiving (a slightly different word rarely changes meaning). **For code it
is costly**: one wrong identifier, signature, or import = broken code. Coding over a large
file/repo needs to pull an *exact* symbol that may be defined thousands of tokens earlier.

The flip side — Helix's advantage — stays real: it needs ~15-20× less KV-cache for the same
long context, so it can *hold a whole large repo* affordably where a same-size Transformer
OOMs. The goal of this note is to close the exact-recall gap **without** giving up that
efficiency advantage.

## The core tension (read this first)

Every recall fix below erodes some of the KV-cache / efficiency advantage that makes the
hybrid attractive. It is a **dial: efficiency ↔ exact recall**. You cannot have both maxed.
Most good hybrids land on *a few* global-attention layers + recall-oriented training, which
buys roughly Transformer-level recall at most of the efficiency saving.

## The levers (bake in at design time — these are architecture changes = from-scratch retrain)

1. **A few FULL / global attention layers as retrieval-heads.**
   2-3 of ~80 layers use global (not windowed) attention so they can attend to any distance
   exactly. KV stays far below a full Transformer (3 global layers ≪ 80). Helix **already has
   a `global_tokens` mechanism** — it was set to `0` in the `_flash` variant to enable
   flash-attn (`global_tokens != 0` forces the native O(L²) path). That is the built-in dial.

2. **Interleave attention through the stack** (not all-at-end).
   The 0.9B orders layers `6 Mamba → 16 GLA → 6 Sparse-Attn` (attention only at the top).
   Distributing attention layers across depth (à la Jamba) gives exact-recall capability at
   multiple representation depths.

3. **Larger recurrent state (`d_state`, currently 128).**
   More state capacity = less lossy compression = better recall straight from the SSM/GLA
   layers. Direct lever on the compression-loss term. Costs memory/compute, scales recall.

4. **Train on copy / associative-recall + cross-reference code.**
   Synthetic associative-recall tasks and long code with cross-file references, mixed into
   pretraining. The architecture *provides* the capability (via the attention layers); the
   model must *learn to use* those layers as retrieval-heads. Without such training the
   capability often stays dormant.

5. **Serving-side symbol injection (architecture-free; works at any size).**
   Rather than relying on internal recall over 100K tokens, an LSP / symbol-index / RAG pulls
   the *exact* definition (signature, imports) near the cursor. Sidesteps the weakness
   entirely and reuses the existing RAG infrastructure. Moot until a model can actually code,
   but the cheapest fix once one can.

## Recommended sketch for a future code-focused big Helix

- Keep the hybrid + 200k tokenizer (proven; German fertility).
- Train at long context (≥4096, ideally higher for repo-scale), sparse-attn window matched.
- **2-3 global-attention layers, interleaved** (re-enable `global_tokens` on those; accept
  they fall off the flash-attn fast path — the recall is worth it on a few layers).
- Bump `d_state` modestly if the recall gates demand it.
- Heavy real-code pretraining (file-level, not snippets) + copy/recall training mixed in.
- A **long-context exact-recall gate** (needle-in-haystack over code: retrieve an exact
  identifier defined N tokens back) from early checkpoints — promote on it, not val_loss.
- Keep serving-side symbol/LSP injection as the belt-and-suspenders layer.

## What NOT to do now

- Do **not** retrofit the 0.9B. These are from-scratch architecture changes; you cannot
  continue-pretrain `step_60000` with a different layer layout / `d_state` / global tokens.
- Do not optimize recall before the model can code in short context (capacity/size first).
- The realistic current step is unchanged: finish pretraining the 0.9B (continued-pretrain
  pilot), then decide on a bigger model. These notes are input for *that* model's design.
