# Helix v3 — byte-level / small-vocab sketch (research note, NOT the current run)

Status: **idea sketch for a future experiment.** The live foundation run and the
German-data pipeline (fineweb2-v2, german_commons, RedPajama) are the priority and
are unaffected by this. This note exists so the idea is captured concretely and can
be judged on cost/benefit later, not built on impulse.

The seed: "build an AI that understands text closer to the machine level, or finds
a denser pattern/language." Translated into a real, testable design for the
**representation layer**: drop the 200k tokenizer and operate at (or near) the byte
level. Below is what that buys, what it costs, and the honest verdict.

> Terminology note: in this repo, **"knowledge DNA" already means something specific**
> — the `knowledge_dna_v2` / `knowledge_kernel` track (dense *knowledge* blocks), NOT
> byte-level representation. This doc is only about representation. See §0.

## 0. Relation to the existing Auralis vision

Auralis is designed as three layers, and the tokenizer special-token spec already
scaffolds all three:

1. **Representation** — text → vectors. Today: 200k SentencePiece (en 50 / de 40 /
   code 10, `byte_fallback`). This sketch's byte-level idea lives HERE.
2. **Knowledge** — `knowledge_dna_v2` / `knowledge_kernel`: dense curated fact/
   definition blocks via `<memory>`/`<recall>`. **THIS is the repo's "DNA."** Status:
   prototyped (corpus builder + ablation harness run) but **UNPROVEN** — the only fair
   ablation had `plain` ≥ `kernel` on models too tiny to answer; explicitly NO-GO /
   experimental. Next real test: a 100M+ fair ablation (the 3090 can run it).
3. **Skills** — LoRA/DoRA adapters + routing (`<lora>`/`<route>`) to load
   code/languages/domains onto the frozen base.

Byte-level (this doc) and knowledge-DNA are **complementary layers, not the same
idea.** And the 200k vocab is a **deliberate universal-base choice** for the adapter
ecosystem (broad en/de/code + Unicode coverage) — not an oversight.

---

## 1. Why this is even worth considering — the 200k-vocab tax

Helix v2: `d_model=1280`, `n_layers=28`, `vocab=200000`, tied embeddings.

- Embedding/LM-head table = `200000 × 1280 = 256M` parameters.
- Total model ≈ 900M parameters.
- → **~28% of the entire model is a lookup table**, not reasoning layers. A quarter
  of the "capacity" is spent mapping tokens↔vectors.
- Measured (`bench_model_breakdown.py`, clean 3090): the LM head + CE is **~22% of
  forward+backward time** at our token count — the single most expensive op.

So the 200k vocab is expensive on **both** axes: parameters and compute. A tiny
vocab (256 bytes) would:
- shrink the table to `256 × 1280 = 0.33M` params → **free ~256M params** to either
  make the model smaller/cheaper OR add ~8–10 more real layers (or widen d_model),
- collapse the LM-head compute (predicting over 256 classes instead of 200000).

There is also a **quality** argument, specifically for German:
- Subword tokenizers fragment German **compounds** ("Donaudampfschifffahrts­gesell­schaft")
  and rich morphology inconsistently; a 200k multilingual vocab also spends most of
  its slots on the dominant languages, under-serving German sub-patterns.
- Byte/char-level has **no OOV, no tokenizer language bias, uniform handling of code,
  numbers, typos, scripts.** This is exactly the kind of thing that can move
  `bpb_german`, which is our actual bottleneck.

## 2. The naive byte idea — and its honest catch

Naive "just feed raw bytes" (vocab=256) has a real cost: German is ~4–5 bytes per
current token, so a 2048-token sequence becomes ~8k–10k **byte** positions. Every
mixer/FFN layer runs **per position**, so:

- mixer + FFN compute scales ~**4.5× up** (more positions),
- the LM head gets ~**170× cheaper** (256 vs 200k classes),
- net per-document FLOPs ≈ **~3× MORE expensive** than the token model. The longer
  sequence dominates; the cheap head does not compensate.

That is precisely why production models keep tokenizers — raw bytes trade compute
for purity. So naive bytes is *not* the design. The clever fix is patching.

## 3. The real design — dynamic byte patching (BLT-style), and why Mamba fits

Meta's **Byte Latent Transformer (BLT, 2024)** and **MambaByte** solve the cost: a
small, cheap byte-level model groups raw bytes into **variable-length patches**
(e.g., by next-byte entropy — split where the text gets "surprising"), and the big
backbone runs on the **patch** sequence, not raw bytes. You get byte-level inputs/
outputs with roughly **token-count** sequence lengths → byte benefits without the
4.5× blow-up.

Why this fits Helix specifically:
- We already run a **Mamba-2 + GLA hybrid**. Linear-time mixers (Mamba/GLA) are the
  natural backbone for long/byte sequences — no quadratic attention blow-up. The 6
  **sparse-attention** layers are the part that suffers most from long sequences and
  would need care (smaller window, or move them to operate on patches only).
- A small byte encoder/decoder + entropy patcher bolts onto the existing stack; the
  backbone stays mostly as-is, just fed patches.

## 4. Three concrete tiers (increasing ambition / risk)

| tier | what changes | params freed | compute | risk | German upside |
| --- | --- | ---: | --- | --- | --- |
| **A. Smaller vocab (e.g. 48k–64k)** | retrain SentencePiece smaller; same architecture | ~215M (32k) / ~195M (48k) / ~175M (64k) | ~neutral (cheaper head vs ~15–25% longer seqs) | **low** | small — still subword |
| **B. Naive byte/char (vocab 256)** | tokenizer-free, raw bytes | ~256M | **~3× more** (longer seq) | medium | high (no OOV/compounds) but pays compute |
| **C. Patched byte (BLT/MambaByte-style)** | byte encoder + entropy patcher + backbone on patches | ~256M | ~comparable to tokens | **high** (new components, harder to train) | high, *and* compute-neutral |

- **Tier A** frees ~175–215M params BUT **conflicts with the universal-base/adapter
  goal**: a smaller vocab means less efficient coverage of code + non-de/en text, and
  **LoRA cannot add vocab back later**. The 200k was chosen deliberately (en/de/code +
  adapter/route/memory tokens) to be a broad base — shrinking it trades exactly that
  flexibility away. So Tier A only makes sense if abandoning the universal ambition.
  For the adapter ecosystem, **byte-level (Tier C) is the aligned path**: it is the
  *most* universal coverage (any script/code) AND frees the params, with no
  "can't add tokens" wall. (Any vocab change = full retrain regardless.)
- **Tier C** is the "krasse" version that matches the original intuition and could
  genuinely help German — but it is a real research build (entropy patcher, byte
  encoder/decoder, training stability), i.e. a from-scratch Helix v3, not a tweak.

## 5. The evaluation is *fair by construction* — bits-per-byte

Key elegance: our headline metric, **`bpb` (bits-per-byte), is tokenizer-independent.**
It normalizes by raw bytes, so a byte-level model and a token model can be compared
**directly** — there is no apples-to-oranges problem. We are already measuring
`bpb_german` / `bpb_english`. The whole question reduces to a clean experiment:

> Does a byte/small-vocab Helix reach a **lower `bpb_german`** at the same param and
> compute budget than the 200k-token Helix?

If yes → real win. If no → the tokenizer recipe stays. No hand-waving needed.

## 6. Cheap validation plan (when the 3090 / a GPU is free — NOT now)

Do this small before committing to a v3 build:
1. Train two **tiny** models (~50–100M) on the same German bytes, same compute:
   (a) 200k-token, (b) char/byte-level (Tier B, naive — simplest to stand up).
2. Compare `bpb_german` (fair, byte-normalized) and tokens/sec.
3. If byte-level's `bpb_german` is competitive *despite* the compute handicap, that
   is strong evidence Tier C (patched) — which removes the handicap — would win.
4. Only then prototype the entropy patcher (Tier C) at small scale.

This is days of small-model work, fully isolated from the production run.

## 7. Risks / honest verdict

- **It does not help the current run.** `bpb_german` today is data-limited, not
  representation-limited. A new vocab changes nothing about needing more German data.
- **Tier C is a genuine research bet.** Byte/patched models are recent and not yet a
  clearly-dominant recipe; training stability and the patcher are non-trivial.
- **Compute reality:** only Tier A (and Tier C *if* the patcher works) are
  compute-sane; naive bytes (B) costs ~3× and is only a research probe.
- **Best first step (given the universal-base goal):** the **Tier-B tiny-model bpb
  probe** — cheap, decides whether byte-level beats tokens on German. If yes, **Tier C
  (patched byte)** is the path that *fits* the universal-base/adapter vision (universal
  coverage + frees params). Tier A is off-goal here and only worth it if dropping the
  multilingual/code ambition.

**Verdict:** Real, on-theme, and the bpb metric makes it cleanly testable — but it is
a **Helix v3 experiment for after** the current run and the German-data scale-up, not
a mid-run change. Capture now, test cheap later, build only if the tiny-model bpb
probe says byte-level beats tokens on German.

---

## 8. Addendum (2026-06): external review + v2 capacity evidence

Trigger: the v2 architecture diagram was reviewed by two external models (GPT 5.5,
Gemini). Both validated the hybrid design and both independently flagged the **same**
point — the **200k vocab is large for a sub-1B model**. This doesn't change the §4
verdict (naive shrink = Tier A conflicts with the universal-base goal; byte-level =
Tier C is the aligned param-freeing path), but it adds hard numbers + a real v2
datapoint, so capture it.

### Architect's rationale — why 200k follows from the vision (capture, in the architect's own logic)
The 200k vocab is **not** an accident of fertility tuning. It follows directly from the
core Auralis design intent:

1. **ONE broad, frozen BASE model** that is *universal* — broadly capable across many
   languages, code, and domains — rather than a narrow specialist.
2. **Knowledge and skills are NOT baked into the base.** They are loaded *on top* as
   **DoRA/LoRA adapters** (the §0 "Skills" layer via `<lora>`/`<route>`) plus the
   knowledge_dna/kernel track for facts. The base = the substrate; adapters = the
   specialization. You ship one base and many small adapters, not many full models.
3. **Therefore the base must cover the token space of everything an adapter might ever
   target.** A large vocab (200k: en/de/code + adapter/route/memory special tokens) is
   the *universal substrate* so a future adapter for any language/domain "fits" without
   vocab gaps or heavy fragmentation.

The decisive constraint that makes this a *deliberate* choice, not waste:
> **A LoRA/DoRA adapter changes weights, not the token table.** You can add a *skill*
> with an adapter; you cannot add *vocabulary* with one. A small vocab is therefore a
> permanent ceiling — you'd be locked out of efficiently representing anything the small
> vocab didn't anticipate, with no fix short of a full retrain. A large vocab now
> *future-proofs* the universal base for the whole adapter ecosystem.

Under this goal the 200k is an **intentional investment in extensibility**. It is also
exactly why §4 rates Tier A (shrink to ~128k) as *off-goal* for a universal base, and
byte-level (Tier C) as the *aligned* param-freeing path — Tier C keeps universal
coverage (any script/code) *and* frees the 256M, with no "can't add vocab" wall.

**Honest caveat (the filter's job):** the universal-vocab bet pays off *more* at larger
base sizes. At ~954M (~700M body) the base is small, so 27% on embeddings is a steep tax
paid *now* for universality exploited *later* — and the adapter-knowledge thesis
(knowledge_dna/kernel) is itself still UNPROVEN. So the philosophy is coherent and 200k
follows from it logically; the open v3 question is whether a sub-1B base is large enough
to "afford" universality, or whether v3 should (a) grow the base so 256M embeddings are
a smaller fraction, or (b) go byte-level (Tier C) to get universality *and* the params
back. **The vision picks the destination (universal base + adapters); v2's evidence +
base-size pick the route.**

### Hard cost numbers (confirms §4's Tier-A premise)
- Embedding / LM-head = `200_000 × 1280 ≈ 256M` params (tied → counted once).
- That is **~27% of the ~954M total**. The actual mixer "thinking body" is only **~700M**.
- So v2's *reasoning capacity* is ~700M-class, not 954M-class — the headline oversells
  the compute body by a quarter. (GPT's exact point; it stands.)

### The benefit that justifies it (why 200k, not a blunder)
- **Measured fertility is good:** tokens/byte ≈ **0.176 (de) / 0.196 (en)**
  (`eval_diagnostic.py`). A 64k vocab raises tokens/text → shorter effective context,
  slower per-char gen. This is real and on the FOR side.
- German morphology/compounds + multilingual + code ambition all favour a larger vocab.
- Precedent: Gemma uses 256k. 200k is high-end, not exotic.

### NEW empirical signal from v2 (concrete)
The step 9k→15k→25k generation probes show **language fluent, facts bind slowly /
flip-flop** ("capital of Germany" = Munich→Berlin→Munich across checkpoints). That is
*consistent with a capacity-limited thinking body* — the 256M spent on embeddings is
capacity the knowledge layers never get. Not proof, but the first concrete datapoint
that the representation cost may be biting where it matters (fact recall).

### What this changes for v3 (tie-in to the tiers above)
- It **raises the priority of the §6 tiny-model bpb probe** — we now have a motive
  (capacity), not just curiosity.
- The *aligned* param-freeing fix remains **Tier C (byte/patched)**, which frees the
  256M without the "can't add vocab back" wall and keeps universal coverage.
- **Tier A (~100–128k) is valid ONLY if the universal-base ambition is dropped** to a
  pure DE/EN(+code) model — which is, in practice, exactly what v2 *is* today. So the
  reviewers' "use 128k" is correct *under that narrower scope*; it's a real fork:
  universal base (→ Tier C) vs. focused DE/EN+code model (→ Tier A ~128k).
- **Locked for v2:** tokenizer + tied embeddings + the live run cannot change vocab
  mid-stream. From-scratch v3 decision only.

**Decision rule for v3:** if v2's final fact-recall stays weak *despite enough clean
data*, treat that as evidence the body is capacity-starved → prioritise freeing the
256M (Tier C if staying universal, Tier A ~128k if going focused). Re-measure fertility
at 128k vs 200k first (cheap, tokenizer-only) before committing.

### Scaling intent — this largely RESOLVES the fraction concern
Explicit architect intent: **1B is the foundation, not the target — the base is meant to
grow** (~3B → ~7B+ later). That directly dissolves the "27% in embeddings" caveat,
because the embedding (`vocab × d_model`) grows only *linearly* while the body grows
*quadratically in width / linearly in depth*. So the **vocab fraction shrinks as the
model scales**: roughly **27% @ 1B → ~13% @ 3B → ~10% @ 7B+**. The 200k is therefore
"pre-paid" universality that pays off *precisely when* the base is scaled — which is the
plan. This reframes the GPT/Gemini critique: their "200k is big" is true *at 1B*, but
1B is explicitly a stepping stone.

Corollaries already in motion (the threads are one coherent plan, not separate detours):
- Scaling needs **much more data** (Chinchilla ~20 tok/param → 3B≈60B, 7B≈140B tokens)
  → exactly why the RedPajama/HPLT cleaning pipeline matters now.
- Scaling needs **multi-GPU** → the DDP/torchrun support is already built.
- Scaling keeps the **universal vocab + adapter ecosystem** intact (no tokenizer churn).

So the honest v3 question narrows: not "is 200k too big?" but "at the *target* scale, is
token-200k or byte-level (Tier C) the better universal representation?" — decided by the
§6 bpb probe.
