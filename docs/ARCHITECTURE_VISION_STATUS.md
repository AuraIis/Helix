# Auralis / Helix v2 — Architecture Vision & Maturity Map

One place for the **full vision** and an **honest maturity status** of each piece,
because the design is otherwise scattered across several specs. Updated 2026-05-31.

Status legend: ✅ done/proven · ⏳ in progress · 🟡 partial/scaffolded ·
🔬 prototyped-but-unproven · 📋 spec'd, not built.

---

## The vision (brain-inspired, modular)

A small, strong **universal base model** + composable modules loaded on demand —
compute scales with task complexity. From `SPEC_PHASE_5_LORA_SYSTEM.md`:

```
Base model        → language + base world knowledge   (Broca/Wernicke)
Router-LoRA       → complexity decision               (Thalamus)
Denk-/Logik-LoRA  → reasoning + self-verification      (Prefrontal cortex)
Memory-LoRA       → persistent knowledge               (Hippocampus)
Topic-LoRAs       → domains on demand (Medizin/Recht/…) (Temporal lobe)
Tools             → Python / web / code-exec           (hands/eyes)
Level 0 (~100ms, base only)  …  Level 5 (~15s, everything + self-verify)
```

Three structural layers, each with its own maturity:
1. **Representation** — how text ↔ vectors (the tokenizer).
2. **Knowledge** — dense curated facts/definitions ("Knowledge-DNA").
3. **Skills** — LoRA/DoRA/MoRA adapters + routing + tools.

---

## Where we are: Phase 1 (build the base)

Everything modular sits **on top of a strong base**. `SPEC_PHASE_5` itself gates on
"Phase 4 done (aligned base)". We are in **Phase 1 (pretraining)**. So the current
focus — the foundation run + the German-data scale-up — is **the foundation the whole
tower stands on, not a detour from the vision.** Base first; modules later.

Rough phase chain: **Pretrain (now) → SFT (`SPEC_PHASE_3_SFT`) → Alignment (Phase 4)
→ LoRA system (`SPEC_PHASE_5_LORA_SYSTEM`)**.

---

## Maturity map

| Component | Status | Evidence | Next step |
|---|---|---|---|
| **Tokenizer / representation** | ✅ done | 200k SP, en50/de40/code10, `byte_fallback`, all module special-tokens (`<lora>`/`<route>`/`<memory>`/`<recall>`/`<tool>`/code tags). DE ~150 tok/100w (v1: 220). | none — deliberate universal base. (`docs/HELIX_V3_BYTE_LEVEL_SKETCH.md` = far-future alt.) |
| **Base model** (Helix v2 1B hybrid) | ⏳ in progress | foundation run @ ~step 4700/50k; 6 Mamba + 16 GLA + 6 sparse; bpb_de 1.485→~1.27 (improving), bpb_en 0.41. | finish run; close DE gap via more data. |
| **Multi-GPU / DDP** | ✅ done | trainer + torchrun launcher, DDP-agnostic ckpts (committed). | use when scaling. |
| **Data pipeline** (edu quality) | ⏳ active | edu classifier (Pearson 0.87); fineweb2_de filtered (1.55M); german_commons scoring (~25% keep) + fineweb2-v2 (9.9M docs) queued; RedPajama next. | grow unique DE pool 1.8B → ~5B+ to cut 12× repetition. |
| **Knowledge layer** (DNA/kernel) | 🔬 unproven | `scripts/experimental/knowledge_dna_v2.py`, `knowledge_kernel.py`; corpus builder + ablation harness run; **fair ablation: `plain` ≥ `kernel`** on models too tiny to answer → NO-GO. | fair **100M ablation** + bigger/cleaner concept set (3090). |
| **Skills layer — LoRA mechanism** | ✅ proven (proxy) | `scripts/eval/mora_smoke.py`: PEFT + MoRA, freeze base → train adapter → save/reload/verify, on GPT-2. `peft>=0.13.0` in `pyproject`. | **LoRA-on-real-Helix smoke** — does PEFT attach cleanly to the hybrid Mamba/GLA layers? |
| **Skills layer — routing system** | 📋 spec'd | `SPEC_PHASE_5_LORA_SYSTEM.md` (router/meta/topic LoRAs, complexity levels). | build after base is aligned (Phase 5). |
| **Tools** (Python/web/code-exec) | 📋 spec'd | Phase-5 spec; tokens (`<tool>`/`<tool_result>`) exist. | Phase 5. |
| **Adaptive training controller** | 🟡 exists | `src/auralis/adaptive/` (stages, signals, scoring, margin-probes, frozen_gate, bpb). Scope = *training-time* curriculum control. | clarify vs Phase-5 *inference* router (not yet audited). |
| **perf_lab** (kernel opt) | ✅ concluded | clean-3090 study: keep PyTorch full-logits CE (fused 18–117× slower at our scale); Liger RMSNorm 1.2–4× (minor). | not a current lever; revisit only if batch×seq grows. |

---

## Dependency ordering (what unblocks what)

```
Tokenizer ✅ ──▶ Base model ⏳ ──▶ SFT ──▶ Alignment ──▶ LoRA system 📋 / Tools 📋
                     │
                     ├──▶ Knowledge-DNA 🔬  (needs a decent base to test fairly)
                     └──▶ Skills mechanism ✅ (proven; full system needs aligned base)
```

The base model is the gate for almost everything. Its current limiter is **German
data quantity/quality**, which is exactly what the active data pipeline addresses.

---

## Continual learning & self-knowledge (Phase 5+ design guardrails)

Target: a system that keeps acquiring skills/knowledge, **recognizes what it already
knows**, spends learning on the gaps, and does not forget. What is real vs not:

**Real & buildable — "does the model already know X?" is measurable (functionally).**
- Probe the model with candidate knowledge and read its **loss / margin / confidence**:
  low surprise = already known → skip; high-but-learnable = a real gap → train on it;
  high-and-chaotic = noise → ignore. (Principle: reducible-loss / RHO-LOSS data
  selection, Mindermann 2022; "LMs Mostly Know What They Know", Kadavath 2022.)
- After learning, **re-measure the known set** to catch forgetting (a cheap
  catastrophic-forgetting guard).
- We already have the measurement substrate: `src/auralis/adaptive/` (**margin
  probes + bpb + frozen_gate + stages**) is exactly a "do-I-know-this?" meter and a
  signal-gated trainer. The missing piece is the knowledge-inventory + gap-driven
  selection loop on top.

**Verifier-gated self-improvement — real where correctness is checkable.**
- Code (run tests), math (checkers), games (win/loss): generate → verify → train on
  the verified-correct. This is the only domain where a *self*-improvement loop
  reliably climbs. The planned **Logik-LoRA** (self-verification) lives here.

**Honest limits (do NOT design around these being solved):**
- You **cannot read the weights** to enumerate known facts — knowledge is
  distributed/superposed (localization research like ROME/MEMIT finds rough regions,
  not a lookup table). Measure knowledge *functionally* (loss/probe), not by
  inspecting weights.
- **Low loss ≠ true.** Models are confidently wrong; familiarity ≠ correctness. Use
  loss for "is this familiar", but require an **external verifier / source** for "is
  this correct" — else the system skips fixing things it wrongly "knows".
- **Self-data-only loops collapse** (Shumailov 2024). Genuine improvement needs
  external grounding: verifiers, fresh real data, tool/world feedback.

Placement: Phase 5+, on the aligned base; `src/auralis/adaptive/` is the natural home.

## Learning levers beyond data (PARKED — one at a time, gated on a baseline)

The discipline: **change ONE thing at a time, measured against a stable baseline.**
Changing several at once = you cannot attribute the result. Finish the base
relaunch first (it gives the baseline), then run these as separate, measured
experiments — in priority order:

1. **Knowledge distillation** (biggest lever for a small model) — train the 1B to
   mimic a strong teacher's output distribution instead of only raw text. Needs a
   teacher (strong German/multilingual model) + its logits/outputs. The single most
   effective "beyond data" technique at this size.
2. **LR schedule + optimizer** — WSD (warmup-stable-decay) schedule; optionally a
   modern optimizer (Muon, promising but newer). Cheap, no new data.
3. **Data mixture / curriculum** — ratios (de/en/code) + ordering (clean/easy →
   complex). Better *use* of data, not more data.
4. **Multi-Token Prediction (MTP)** — NOT a switch: needs MTP heads + a changed
   objective = an architecture change = a NEW model (v3). Tokens are reserved; not
   activatable mid-run.
5. **muP / FP8 / fancy optimizers** — efficiency or single-digit-% gains; low
   priority for a solo run, high complexity.

Already in the current design (not missing): RMSNorm, SwiGLU, RoPE, bf16.

### Correction: perf_lab kernels are NOT a training lever
A natural assumption is "use the perf_lab CUDA/Triton kernels to make training
faster/more stable." The clean-GPU study found the **opposite**: the custom fused
CE is **18–117× slower** than PyTorch full-logits **and** drifts more. Conclusion
(already in `perf_lab/docs/`): **keep full-logits** — the training already uses the
fastest + most stable path. The only real perf_lab win is **Liger RMSNorm** (~1.2–4×,
3090-tested); worth a Blackwell head-to-head confirm, nothing more. Do NOT swap the
CE kernels into training — it would regress.

## Deployment / serving (later — where C++ actually earns its keep)

Reminder on the recurring "should we write it in C++" question: **training stays
Python/PyTorch** (the compute is already C++/CUDA under the hood; a C++ rewrite of
the training buys ~nothing and hurts maintainability + community). The real C++
payoff is **inference/serving**, not training — and it belongs to the "go public"
phase:

- **Goal:** let people *run* the model easily — fast, on consumer hardware, ideally
  a single binary with no Python install. That's what wins a public demo.
- **Tools:** `llama.cpp` / **GGUF** (C++, CPU + consumer-GPU, single binary),
  TensorRT / ONNX Runtime (server-side fast inference).
- **HONEST caveat:** our architecture is a **non-standard hybrid** (Mamba-2 + GLA +
  sparse attention). llama.cpp/GGUF target standard transformers (+ some Mamba), so
  it will **not** "just work" — supporting our stack needs custom C++ architecture
  code (real effort), or a runtime that handles it.
- **Sequencing (pragmatic):**
  1. **First public demo = PyTorch** (a HF Space or a tiny server) — works day-one,
     no C++, no architecture port. Simplest path to "look, try it".
  2. **Later:** a C++/GGUF runtime for efficient, easy, no-Python distribution —
     the real C++ project, after there is a model worth shipping.

So the C++ instinct is correct — just for the **serving phase**, after a good model,
and likely after a first PyTorch demo. Not for training.

## Honest bottom line

- The vision is **coherent and largely scaffolded** — the tokenizer reserves slots for
  every module; the LoRA mechanism is proven; the knowledge idea is prototyped.
- Two things are **genuinely unproven**: the Knowledge-DNA *helps* (NO-GO so far), and
  LoRA *attaches cleanly to the hybrid Helix* (only proven on a GPT-2 proxy).
- Nothing is blocked by missing ideas — it is blocked by **needing a strong base
  first**. Current priority (foundation run + German data) is correct and on-path.
- Two cheap, decisive 3090 experiments are queued for *after* the data work:
  (1) fair 100M Knowledge-DNA ablation, (2) LoRA-on-real-Helix smoke.
