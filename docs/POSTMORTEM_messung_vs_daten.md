# Post-mortem: „Erst messen, dann die Daten verdächtigen"

**Kernlektion in einem Satz:** Fast alles, was bei Helix v2 wie *„das Modell lernt
nicht / die Daten sind schlecht"* aussah, war in Wahrheit **kaputte oder irreführende
Messung, ein zu hoher Learning-Rate, ein Guard-Bug oder rohes Decoding** — und wir
waren mehrfach nahe daran, es fälschlich den Trainingsdaten zuzuschreiben. Erst
*sauberes Messen* hat den wahren Stand aufgedeckt.

Kontext: Helix v2, ~954M Hybrid (6× Mamba-2 + 16× GLA + 6× Sparse-Attn), bilingual
DE/EN, 200k Vokab, Warm-start Continued-Pretraining. Dieses Dokument ist die ehrliche
Chronik der Fehldiagnosen, damit sie sich nicht wiederholen.

---

## Die Fälle (jeweils: Symptom → erste (falsche) Vermutung → wahre Ursache)

### 1. Vermeintlicher val-loss-Rückschritt beim Warm-start
- **Symptom:** Nach dem LR-Peak stieg der val_loss.
- **Erste Vermutung:** Die neuen deutschen Daten sind schuld.
- **Test:** Auf altem **und** neuem Pool reproduziert → **nicht die Daten**.
- **Wahre Ursache:** LR zu hoch für einen Warm-start (frisches AdamW + Re-Ramp auf den
  From-scratch-Peak destabilisiert einen bereits konvergierten Checkpoint).
- **Fix:** niedriger Continuation-LR (3e-5), kurzer Warmup.

### 2. „1.172 → 1.222"-Rückschritt = ungültiger Vergleich
- **Symptom:** bpb_de scheinbar von 1.172 auf 1.222 verschlechtert.
- **Erste Vermutung:** Das Modell ist schlechter geworden.
- **Wahre Ursache:** Die alte „beste" 1.172 war auf einem **anderen Val-Set** gemessen
  als die neue 1.222. Äpfel mit Birnen.
- **Fix:** Step-0-Eval (Checkpoint laden, **ohne** Training, identisches Set) → wahre
  Baseline → kein echter Rückschritt.

### 3. Die Messung selbst war der Hauptschuldige (mehrfach kaputt)
- `tokens_per_byte` war geraten (0.2338) statt gemessen (**0.176**) → bpb_de ~33%
  aufgebläht.
- Die Eval war **stochastisch** (stateful RNG lief weiter → jede Eval zog *andere*
  Tokens) → die „Kurve" war Sampling-Rauschen, kein sauberes Signal.
- Das Deutsch-Val war nur der **Wikipedia-Tail** (nicht repräsentativ); der
  Englisch-Tail war zufällig trivial → der bpb-**Gap sah aus wie 3.2** (Fata Morgana),
  echt ~**1.04**.
- **Folge:** „Regression" und „riesige Sprachlücke" waren **Mess-Artefakte**.
- **Fixes:** deterministische Eval (`reset_rngs`), gemessene tokens/byte,
  repräsentatives Val-Sampling, Step-0-Diagnose (Kernels an/aus).

### 4. Die Notbremse stoppte den Lauf fälschlich (step 4250)
- **Symptom:** Auto-Stop „val_regression".
- **Erste Vermutung:** Das Modell regressiert.
- **Wahre Ursache:** Der Guard zählte „kein neuer Bestwert" als Rückschritt (statt
  *echter* aufeinanderfolgender Anstiege, ohne Toleranz). Das Modell war **gesund**.
- **Fix:** echte Folge-Logik + `min_delta`; `error_if_nonfinite`; harter Tokenizer-Check.
  Danach resume → sauber bis ~35k.

### 5. „Modell lernt keine Fakten" (München-Flip) = Decoding-Artefakt
- **Symptom:** Greedy-Generierung: „Hauptstadt = München→Berlin→München" über
  Checkpoints.
- **Erste Vermutung (auch von zwei externen Reviewern geteilt):** Wissen ist nicht
  verankert → Daten-/Skalierungsproblem.
- **Wahre Ursache:** Eine **rigorose Margin-Messung** (`NLL(falsch) − NLL(richtig)`,
  mehrere Distraktoren, 5 Kategorien) ergab: **Geschichte 100%, Geografie 86%, gesamt
  72%** (2-Wege sogar 87,5%). Das Wissen **ist da**. Greedy maß nur das
  **Antwortverhalten** (driftet beim freien Generieren), nicht das interne Wissen.
- **Korrektur:** *nicht* „Wissen fehlt", sondern „**Wissen da, Decoding/Answering noch
  roh**".

---

## Begriffe sauber trennen (damit wir nicht wieder vermischen)
| Messung | misst |
|---|---|
| **Recall-Margin** `NLL(falsch)−NLL(richtig)` | **Wissen im Modell** |
| **Top-k nach Fakt-Prompt** | **Abruf-Nähe** (kommt der richtige Kandidat oben an?) |
| **Greedy-Generierung** | **Antwortverhalten** beim freien Erzeugen |
| **SFT / Instruction** | **Format & Steuerbarkeit** (≠ Faktenwissen) |

## Korrigierte Meilenstein-Sicht (Stand ~35k/50k)
- **A — Stabiles Training:** ✅ bestätigt (val↓, grad stabil, 0 Alarme)
- **B — Sprachlernen (DE/EN flüssig, getrennt):** ✅ bestätigt
- **C — Instruction Following:** offen (SFT-Phase)
- **D — Faktenbindung:** ✅ überraschend stark (Geschichte/Geografie), nur
  **Wissenschaft schwach (29%)**
- **E — Knowledge-DNA:** unbewiesen, *optionaler* Boost — nicht nötig, um über
  Faktenbindung zu reden

## Was DATEN wirklich betrifft
Nur **Wissenschafts-Fakten** sind echt schwach (Au/Ag, Jupiter/Mars, Siedepunkte). Da
ist **„mehr science-dichte Daten"** der korrekte, *spezifische* Hebel — nicht „mehr
Daten" pauschal. Alles andere war Messung/Tooling/Decoding.

## Konkrete Schutzmaßnahmen (bereits umgesetzt)
- Deterministische Eval · gemessene tokens/byte · repräsentatives Val · Step-0-Diagnose
- Korrigierter Regression-Guard (`min_delta`, echte Folge-Logik) · non-finite-Schutz
- Rigorose Fakten-Recall-Batterie (Margin + Top-k, Mehrfach-Distraktoren, 5 Kategorien)
  → ab step 50k Pflichtmetrik, nicht Augenmaß.

**Merksatz fürs Team:** Bevor eine schlechte Zahl „die Daten" sind — prüfe, ob die
Zahl überhaupt misst, was du glaubst.
