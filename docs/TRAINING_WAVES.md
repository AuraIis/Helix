# Training Waves — von 250M zu 1B

Verbindlicher Ablauf für Phase 1. Basiert auf Michaels Vorgehen
(2026-04-23): **250M = Entscheidungs-Simulator, 1B = Produktionslauf.**
Nicht alles gleichzeitig kippen.

Vorher festziehen:
- Daten-Mix: [`configs/data/phase1_mix.yaml`](../configs/data/phase1_mix.yaml)
- Canary-Modell: [`configs/model/helix_v2_mid_500m.yaml`](../configs/model/helix_v2_mid_500m.yaml) (bevorzugt) oder [`helix_v2_250m.yaml`](../configs/model/helix_v2_250m.yaml) (wenn Tokenbudget knapp)
- Produktions-Modell: [`configs/model/helix_v2_1b.yaml`](../configs/model/helix_v2_1b.yaml)

## Warum überhaupt Wellen?

Ein 250M-Run macht das 1B-Training nicht schneller. Er macht es
**ent-risktr**. Zeitvorteil ergibt sich aus weniger Fehlstarts, nicht aus
weniger FLOPs. Deshalb ist das 250M ein **skalierter Zwilling** — gleicher
Tokenizer, gleiche Layer-Reihenfolge-Logik, gleiche Pipeline, nur schmaler.

## Parameter-Größen im Überblick

| Config | Rolle | Params | Begründung |
|---|---|--:|---|
| `helix_v2_100m.yaml` | CPU/Test | 135 M | schnelle Unit-/Smoke-Tests |
| `helix_v2_250m.yaml` | Canary (wenn Budget knapp) | 261 M | d=768, 12 Layer |
| **`helix_v2_mid_500m.yaml`** | **Canary (bevorzugt)** | **517 M** | d=1024, 20 Layer — echter skalierter Zwilling |
| `helix_v2_1b.yaml` | Hauptlauf | 954 M | d=1280, 28 Layer |

*Anmerkung zur 250M-Benamung:* Die explizit gewünschte Architektur
(`d_model=1024, 20 Layer, n_heads=16, d_head=64, d_ffn=2816, tied, 200k
vocab`) ergibt arithmetisch ~517 M, weil das 200 k-Vocab allein 205 M
Embedding-Params frisst. Wir benennen sie daher sauber als „mid_500m" und
behalten `helix_v2_250m.yaml` als Fallback für engere Budgets.

## Die vier Runden

### Runde 1 — Infrastruktur (Kurz-Canary)
**Ziel:** Bugs finden, nicht benchmarken.

- **Modell:** `helix_v2_250m.yaml` oder `helix_v2_mid_500m.yaml`
- **Tokens:** 50 M – 200 M
- **Dauer:** 1-3 h auf H100, ~$3-10
- **Eingangsgate:** `inference_compat.py` auf einem frischen Checkpoint PASS
- **Ausgangsgate alle JA:**
  - Forward/Backward stabil, keine NaN
  - `grad_norm` gesund (keine explosion / keine collapse)
  - BF16-autocast stabil
  - Checkpoint save + reload OK
  - Val-loss fällt sichtbar in 20-50 Eval-Punkten
  - Keine Health-Alerts mit `level=STOP`

### Runde 2 — Datenmix-Ablation
**Ziel:** den besseren Datenmix wählen — nicht raten.

- **Modell:** `helix_v2_mid_500m.yaml` (oder 250m falls Budget knapp)
- **Konfig:** [`configs/ablation/mix_variants.yaml`](../configs/ablation/mix_variants.yaml)
- **Drei Kandidaten (nicht mehr):**
  1. `baseline_75_20_5` — Referenzpunkt
  2. `de_heavy_70_25_5` — mehr modernes Deutsch
  3. `code_heavy_72_20_8` — mehr Struktur
- **Tokens:** 0.75-1.0 B pro Variante (≈ 2.5-3 B total)
- **Dauer:** 6-12 h auf H100, ~$30-60
- **Ausgangsgate:**
  - ein Sieger nach `decision_gates` in `mix_variants.yaml`
  - kein Per-Language-Regress > 0.05 gegenüber baseline
  - Tiebreak: DE val_loss bei Gleichstand

Starter: `python scripts/pretrain/mix_ablation.py`

### Runde 3 — Sieger-Validierung
**Ziel:** finaler Go/No-Go für 1B.

- **Modell:** gleicher Canary
- **Mix:** Runde-2-Sieger
- **Tokens:** 1.5 – 2.0 B
- **Dauer:** 12-18 h auf H100, ~$60-90
- **Ausgangsgate:**
  - val_loss-Trend zeigt weitere Verbesserung (nicht nur Rauschen)
  - per-Language-val_loss gleichmäßig fallend
  - kumulierte Health-Alerts < 5 (WARN), 0 STOP
  - Baseline-Score (50 Fragen) zeigt messbaren Lerneffekt

### Runde 4 — 1B Hauptlauf
**Ziel:** produktive Phase 1, keine offenen Fragen mehr.

- **Modell:** `helix_v2_1b.yaml`
- **Mix:** Runde-3-validierter Mix (wahrscheinlich `baseline_75_20_5`)
- **Tokens:** 21 B (tatsächlich verfügbar) bzw. 25 B (Brief-Ziel)
- **Dauer:** 3-4 Wochen H100 / ~16 Tage H100 bei guten Kerneln
- **Kosten:** $500-800

## Was in welcher Runde **nicht** erlaubt ist

| Runde | Tabu |
|---|---|
| 1 | Neue Hyperparameter-Ideen, LR-Sweeps, Daten-Experimente |
| 2 | Seq-Length-Sweeps, Architektur-Änderungen, zusätzliche Quellen |
| 3 | Mix-Änderungen, LR-Änderungen |
| 4 | Alles außer genau dem validierten Setup |

## Was zwischen den Runden dokumentiert wird

Nach jeder Runde:
1. `MANIFEST.yaml` im Checkpoint-Ordner (automatisch von `run_report.py`)
2. `scripts/eval/regression_dashboard.py --ckpt-dir …` ausführen
3. Entscheidung in `HISTORY.md` festhalten mit Datum + Referenz auf Dashboard

## Downloads die der Pod zusätzlich ziehen muss

Nicht auf NAS:
- `HuggingFaceFW/fineweb-2` `eng_Latn` (für EN-Top-up)
- `bigcode/the-stack-v2` (ersetzt StarCoderData)

Scripts dafür werden in den Download-Modulen vorbereitet
(`scripts/data/download_english.py --sources fineweb2_en`,
`scripts/data/download_code.py --sources the_stack_v2`). Nichts davon
startet lokal — nur auf dem Pod mit Gigabit-HF-Anbindung.
