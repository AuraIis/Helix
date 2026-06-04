# AURALIS — Projektstand & Gesamthistorie (v1 → Helix v2 → jetzt)

> **Zweck:** Single-Source-of-Truth-Überblick für Michael und neue Reviewer/KIs.
> Synthese — die *detaillierten* append-only Logs sind `HISTORY.md` (Chronik) und
> `LESSONS.md` (nummerierte Lektionen L-001…L-022). Stand: v3-Foundation ~step 35k/50k.

---

## 0) Vorgeschichte: Auralis **v1** (der Vorgänger)
Ein früheres Projekt, das v2 erst möglich machte — vor allem durch seine **Fehler**:
- **L-001:** Trainings- vs. Inference-Prompt-Mismatch (`<|user|>` vs `User:\n`) verschleierte *wochenlang* die Qualität → v2: EIN Prompt-Builder, byte-exakter Test.
- **L-002:** LoRA lernte *Patterns, keine Fakten* (Loss 0.0099 = Memorization, neue Fragen scheiterten) → v2: MoRA für Fakten, DoRA für Patterns, disjunkte Val-Splits.
- **L-003:** GPT-2-Tokenizer ~50% zu ineffizient für DE, aber fest verbacken → v2: **eigener 200k SentencePiece, einmal richtig**.
- **L-004:** german-commons verzerrte Richtung historisches Deutsch → v2: bewusste Mix-Ratios + Stichproben-Reviews.
- **L-005/006:** keine Baseline-Tests; 3 Modellversionen durch vergessenes `--reset-optimizer` verloren.

v1 lieferte ~23,7 GB dedupliziertes deutsches Pretraining-Material (~4,7B Tokens), in v2 wiederverwendet.

## 1) v2 Phase 0–1 (Apr 2026): Fundament
- **Phase 0 (Tokenizer):** eigener **200k SentencePiece** (50 EN/40 DE/10 Code), Quality-PASS (DE 134 tok/100w vs v1 ~220), byte-exakter Roundtrip. Lektionen L-007..L-012 (SP-Normalisierung `identity`, NUL-Strip, Threads, RAM, HF-`datasets`-v4-Ban, Code-Metrik `tokens/KB`).
- **Phase 0.5 (Architektur):** **Helix = 28L Hybrid (6× Mamba-2 + 16× GLA + 6× Sparse-Attn)**, RMSNorm/SwiGLU/RoPE, tied 200k, d_model 1280, ~954M. Pure-torch-Referenz + GPU-Kernels. Init-Loss 12.37 ≈ ln(200k) ✓, 50/50 Tests grün.
- **Blackwell-Bringup:** mamba_ssm/fla/flash-attn auf RTX PRO 5000 — der **`TRITON_OVERRIDE_ARCH=sm89`-Trick** (sm_120 sonst abgelehnt). Kernels sparen v.a. **VRAM**.
- **Phase 1:** Trainer/DataLoader/Tokenisierung/Smoke. Fehler L-013 (`shuf` killt Trainer-Disk-IO), L-014 (Checkpoint-Rotation crasht bei `_emergency`), L-015 (RoPE nur bei Sparse-Layer gebaut), L-016 (`pgrep`-Wrapper matcht sich selbst). NVMe-Staging **42× schneller**. **1B-Hauptlauf** (80k Steps, Mix 70/25/5).

## 2) Phase 2/3 + die 500M-Sackgasse (Apr–Mai 2026)
- **SFT-Daten-Pipeline** (DeepSeek-V4 via OpenRouter), **Anti-Halluzinations-Prompt** (L-017: generisches „lüg nicht" reicht nicht → verbotene Spekulations-Marker + Few-Shot → 0% statt 3% Halluzination). WSL2/3090-Inference.
- **500M-Experimente (v5/v6):** Forensik, v6-Datenpläne (Gutenberg-Books, contamination-checks), **SFT-Reparatur-Sweep v3→v9** gegen *Interferenz* (Bonn/Berlin, Photosynthese, Faust/Goethe). **Frozen/Live-Response-Gates → kein 500M-Checkpoint promotable**.
- **Diagnose:** 500M nicht produktionsreif; Ursache = **Interferenz** → Weg = *sauber gewichteter 1B-Mix* statt Mini-Patches. → Pivot auf 1B.

## 3) Bilingualer 1B-Ramp + Deutsch-Edu-Filter (Mai 30/31)
- 1B de55/en45 bis Step ~3400 enttäuschend. **Diagnose:** nicht Eval, nicht Architektur, sondern **Under-Training (~16% Chinchilla) + qualitäts-invertierter DE-Mix**.
- **Edu-Filter (FineWeb-Edu-Methodik):** Judge **qwen3-235b** (non-thinking) statt gemini (L-018 Thinking-Kostenfalle €24; L-019 billiger Judge ist *besser*); günstiger **e5+Ridge-Klassifikator** (Pearson 0.87), Schwelle kalibriert (L-021). **german_commons gedroppt** (L-020: OCR-historisch). **DDP/Multi-GPU** additiv+gated (L-022).

## 4) Aktuelle Session: Warm-start v2/v3 + Mess-Post-mortem + Wissensprofil
- Warm-start-Continued-Pretraining; **5 Fehldiagnosen aufgelöst** (alle „schien Daten, war Messung/LR/Decoding"): Warm-start-LR zu hoch · ungültiger Baseline-Vergleich · kaputte Eval (falsche tokens/byte, stochastisch, wiki-only-Tail → Gap-Fata-Morgana 3.2 vs echt 1.04) · Guard-False-Stop (step 4250) · „keine Fakten" = Greedy-Decoding-Artefakt. (Details: `POSTMORTEM_messung_vs_daten.md`.)
- **Gebaut:** Step-0-Diagnose, deterministische Eval, fixierter Guard, **rigorose Fakten-Margin-Batterie** (Härtegrade, 5 Kategorien), ehrliches Dashboard, parallele CPU-Daten-Pipeline (RedPajama+HPLT → 74 GB sauber), 3 Blueprints + Post-mortem + Datenstrategie-Doku.
- **Wissensprofil (n=57, step 35k):** 95%-easy-Boden, Gradient 95→89→77; **Geschichte 92 / Geografie 83 / Tech-Konzepte 80 / Wissenschaft 67 / Sprache 64** (strict). Code = nur Konzepte (**0% Code trainiert**).
- **Stand:** v3 ~35k/50k gesund (val 2.34, 0 Alarme); 50k-Check armiert (Gen + Profil); SFT danach (Format ≠ Wissen); Skalierung 1B→3B→7B+ + gezielte Daten (Wissenschaft/cross-lingual) als Plan.

## Wiederkehrende Fehler-Muster (die eigentliche Reife)
1. **Erst Messung verdächtigen, dann Daten** (Eval-Bias, tokens/byte, Decoding ≠ Wissen).
2. **Infra/Disk-Koordination** (L-013 `shuf`, NVMe-Staging, Cross-Chat-Kollisionen).
3. **Judge-/Tooling-Wahl** (L-018/019: billiger non-thinking-Judge > teures Thinking).
4. **Interferenz statt „dumm"** (500M-Sackgasse → größerer, sauberer Mix).

## Begriffe (nicht vermischen)
**Margin = Wissen · Top-k = Abrufnähe · Greedy = Antwortverhalten · SFT = Format/Steuerbarkeit.**

## Reifegrad (Stand jetzt)
- ✅ Stabiles Training · ✅ Sprachlernen (DE/EN flüssig, getrennt) · ✅ Faktenbindung (überraschend stark, Geschichte/Geografie)
- ⚠️ Wissenschaft + Übersetzung schwächer · ⚠️ freies Decoding noch roh
- ▷ offen: Instruction-Following (SFT) · Skalierung 3B+ · knowledge_dna/kernel (unbewiesen, optionaler Boost)

## Leitsatz
> Datensammlung erfolgt auf Basis von **Wissensprofilen**, nicht des Gesamt-Val-Loss.
> Und: bevor eine schlechte Zahl „die Daten" sind — prüfe, ob die Zahl überhaupt misst, was du glaubst.
