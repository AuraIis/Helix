# LoRA-Eval-Standard (Phase 5 Vorarbeit)

Dieses Dokument legt fest, welche Gates jede neue LoRA-Adapter-Trainings-Runde
durchlaufen muss, bevor sie als „produktiv" markiert wird. Inspiration:
v1-Lessons L-002 (LoRA-Memorization) und der Brief §3.3 §6.

## 1. Pflicht-Artefakte pro Adapter

Bei jedem LoRA-Merge / jeder veröffentlichten Version liegt im Adapter-Ordner
zwingend:

```
adapter_<name>_vNN/
    adapter.safetensors
    adapter_config.yaml            # rank, target_modules, alpha, dropout, method
    facts.yaml                     # nur für Topic-LoRAs (Fakten-Tabelle, Quellen)
    train.jsonl                    # ~1000 Samples, 80 Fakten × Paraphrasen
    val.jsonl                      # 20 Fakten DISJUNKT zu Train
    oov_probes.jsonl               # >=20 Fragen außerhalb des Topics
    manifest.yaml                  # training run details (siehe §4)
    quality_report.md              # Scores aus den Gates unten
```

## 2. Pflicht-Gates (alle müssen bestehen)

### 2.1 Disjunkter Train/Val-Split (v1-Lesson L-002)

- ``facts.yaml``-Fakten disjunkt zwischen Train und Val, nicht nur verschiedene Paraphrasen
- Val enthält mind. 20 Fakten, mind. 3 pro Unter-Kategorie
- Code: `python scripts/lora/check_split_disjoint.py --adapter <dir>`

### 2.2 OOD-Fragen (style bleed / regression)

- mindestens 20 Fragen AUSSERHALB des Topics (Mathe-LoRA ⇒ Literatur-Fragen etc.)
- Metric: `base_oov_score` (Base-Modell ohne Adapter) vs. `adapter_oov_score`
- Gate: `adapter_oov_score >= base_oov_score - 0.02` (max 2 %-Punkte Regression)

### 2.3 Base-Regression auf den 50 Baseline-Fragen

- Wir laufen ``eval/baseline_questions.yaml`` MIT geladenem Adapter
- Gate: ≥ 95 % vom Base-Score (kein allgemeiner Qualitätsverlust)

### 2.4 Factual Retention (Core Gate)

- Val-Fakten, 1:1 Formulierung aus ``facts.yaml``
- Gate: ≥ 70 % korrekt (bei 20 Fakten = ≥ 14)
- KEIN Train-Loss-Gate unter 0.05 (= v1-Memorization-Trap), stattdessen Val-Loss-Plateau bei 0.2–0.3

### 2.5 Style Bleed (neu)

- „Neutrale" Fragen ohne Topic, Adapter MUSS wie Base antworten (kein Style-Drift)
- Automatisch: Cosine-Sim der Embeddings von (base_answer, adapter_answer) ≥ 0.9

## 3. Router-LoRA spezielle Gates (aus Punkt 16)

### 3.1 Gelabelte Routing-Daten

Der Router-LoRA wird gegen einen `router_eval.jsonl` getestet mit
mindestens 200 Samples, verteilt über sechs Klassen:

- `level_0_smalltalk`
- `level_1_factual_recall`
- `level_2_tools_needed`
- `level_3_topic_knowledge`
- `level_4_reasoning`
- `level_5_high_risk`

Metric: **Per-Class Accuracy**, nicht nur Overall.

### 3.2 Per-Class Gates

| Klasse | Min-Accuracy | Confusion-Constraint |
|---|--:|---|
| level_0_smalltalk | 95 % | darf nicht zu level_5 routen |
| level_1_factual_recall | 90 % | |
| level_2_tools_needed | 80 % | |
| level_3_topic_knowledge | 80 % | korrektes Topic-Label ≥ 75 % |
| level_4_reasoning | 75 % | |
| level_5_high_risk | **95 %** | darf niemals zu level_0 routen — Safety |

### 3.3 Safety-Fence

- level_5_high_risk → level_0 Routing: **strict zero tolerance**.
- Jeder Router-Release mit einem einzigen solchen Fehler ist ein Release-Blocker.

## 4. Manifest-Pflichtfelder

```yaml
adapter:
  name: ...
  version: ...
  method: dora | mora | galore    # MUST for factual vs pattern distinction
  rank: 16 | 32 | 64
  alpha: ...
  target_modules: [q_proj, k_proj, v_proj, ...]
train:
  base_model: checkpoints/phase4_aligned/best.pt
  base_model_sha: ...
  tokenizer_sha: ...
  samples_train: 800-1500
  samples_val: 20+ (disjunct)
  optimizer: {name, lr, betas, weight_decay}
  early_stop: {patience, monitor: val_loss, min_delta}
  final_val_loss: ...
  final_val_factual_acc: ...
results:
  baseline_base_score: 0.82
  baseline_adapter_score: 0.80      # ≥ 0.78 to pass
  oov_base_score: 0.65
  oov_adapter_score: 0.63           # ≥ 0.63 (base - 0.02)
  factual_val_acc: 0.78             # ≥ 0.70
  style_bleed_cos_sim: 0.94         # ≥ 0.90
```

## 5. Ablauf vor jedem Adapter-Release

```
1. Train läuft mit early-stop auf val_loss
2. python scripts/lora/gate_eval.py --adapter <dir> --base <ckpt>
   Prüft alle fünf Gates. Exit-Code != 0 ⇒ Release blockiert.
3. Bei PASS: git tag -a lora/<name>/vNN
4. Adapter-Manifest automatisch in STATUS.md referenzieren
```

Gate-Script (`scripts/lora/gate_eval.py`) wird in Phase 5 implementiert;
dieses Dokument beschreibt WAS es tun muss.

## 6. Anti-Patterns (v1-Erfahrung)

- ❌ Train-Loss 0.0099 = Memorization, nicht Lernen
- ❌ Val-Fakten sind Paraphrasen der Train-Fakten
- ❌ Base-Regression nicht gemessen
- ❌ Style-Drift auf neutralen Fragen nicht gemessen
- ❌ Router ohne gelabelte Routing-Daten („klingt plausibel")
- ❌ Release ohne Manifest-SHAs (später nicht nachvollziehbar)

Jeder dieser Punkte ist in v1 mindestens einmal vorgekommen und hat
Arbeit kostet. Gates sind die Gegenmaßnahme.
