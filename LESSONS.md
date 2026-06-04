# LESSONS — Auralis v2

Append-Only. Beste Erkenntnisse aus v1 (Tag 0) und jeder neuen Erfahrung in v2.

---

## Aus Auralis v1 übernommen (2026-04-22)

### L-001 — Prompt-Format-Konsistenz ist kritisch
Ein einziger Unterschied zwischen Training- und Inference-Prompt (`<|user|>` vs. `User:\n`) hat in v1
wochenlang die Inference-Qualität verschleiert.
**Regel v2:** EIN Prompt-Builder für Training + Inference + Eval + API. Byte-weiser Test Pflicht.

### L-002 — LoRA lernt Patterns, nicht zwingend Fakten
Blutdruck-LoRA v1 erreichte bei 212 Samples Loss 0.0099 (reine Memorization), neue Fragen
scheiterten trotzdem.
**Regel v2:** MoRA für Fakten, DoRA für Patterns. Val-Split mit **disjunkten** Fakten.
Early-Stopping bei Val-Loss 0.2–0.3. Min. 800–1500 Samples pro Topic.

### L-003 — Tokenizer ist nicht nachträglich tauschbar
GPT-2-Tokenizer war für Deutsch ~50 % zu ineffizient — aber jedes trainierte Gewicht hing dran.
**Regel v2:** Eigener 200 k SentencePiece. Einmal richtig, dann nie wieder anfassen.

### L-004 — Daten-Mix vor dem Training prüfen
german-commons Cultural Subset hat v1 Richtung historisches Deutsch verzerrt.
**Regel v2:** Bewusste Mix-Ratios im Config, Stichproben-Reviews vor jedem Run.

### L-005 — Baseline-Tests ab Tag 1
Ohne feste Eval-Fragen keine ehrliche Progress-Messung.
**Regel v2:** 50 Baseline-Fragen in `eval/baseline_questions.yaml` committed, automatisch bei
jedem Checkpoint laufend.

### L-006 — Optimizer-State bewusst behandeln
Drei Modellversionen (v20, v28, v30) in v1 durch vergessene `--reset-optimizer` verloren.
**Regel v2:** `--reset-optimizer` ist **Default** bei SFT / Continued Pretrain, nicht Ausnahme.

---

## Neue Lessons aus Phase 0 (2026-04-22)

### L-007 — SentencePiece `normalization_rule_name: nmt_nfkc` normalisiert Newlines zu Space
Beim ersten SP-Training sind alle `\n` im Chat-Template zu `" "` geworden → byte-exakter Roundtrip gescheitert (exakt der L-001-Bug-Typ aus v1).
**Regel v2:** `normalization_rule_name: identity` + `byte_fallback: true`. Dann wird jeder Byte, inklusive `\n` (`<0x0A>`), erhaltend kodiert und dekodiert. `quality_report.py` prüft den Roundtrip byte-exakt — **Pflichttest vor jedem Tokenizer-Commit**.

### L-008 — StarCoderData enthält NUL-Bytes (und andere C0-Controls)
Rohe Code-Files (Binärdateien, Editor-Artefakte, generated code) haben verstreute `\x00`-Bytes. SentencePiece-Training emittiert für jedes NUL eine Warnung und kann bei genug NUL-Bytes crashen.
**Regel v2:** Bytes unter `0x20` (außer `\t` `\n` `\r`) vor dem Tokenizer-Training aus dem Korpus strippen — Schritt läuft automatisch nach `prepare_corpus.py`.

### L-009 — SentencePiece `num_threads=0` wird nicht als „alle Kerne" interpretiert
SP verlangt `1 ≤ num_threads ≤ 1024` — ein 0 führt zum sofortigen Abort.
**Regel v2:** im Training-Script `args.num_threads or max(1, os.cpu_count())` — CLI-Default 0 bedeutet auto.

### L-010 — 15 GB Korpus × 10 M Sentences × 200 k Vocab sprengt 32 GB RAM
EM-Training blähte beim ersten Versuch die RAM-Belegung auf bis zum OOM-Kill (exit 127).
**Regel v2:** `input_sentence_size = 5_000_000` (bei 32-64 GB RAM). Erhöhen nur wenn explizit auf 128-GB+-Pod trainiert wird.

### L-011 — HuggingFace `datasets` v4+ blockt Script-basierte Loader
SlimPajama (`cerebras/SlimPajama-627B`), Dolma (`allenai/dolma`), Proof-Pile-2 (`EleutherAI/proof-pile-2`) sind alle nicht mehr ladbar: `Dataset scripts are no longer supported, but found *.py`.
**Regel v2:** Vor Aufnahme einer neuen Quelle in `download_*.py` prüfen, ob sie **parquet-only** verfügbar ist (z.B. `open-web-math/open-web-math` statt `proof-pile-2`). Im Zweifel: `datasets.load_dataset(name, streaming=True)` lokal smoke-testen, bevor das Multi-GB-Download gestartet wird.

### L-012 — „Tokens pro 100 Wörter" ist für Code-Pretraining keine gute Metrik
Code-Zeilen bestehen oft aus 2-3 „Wörtern" (`return x;`), aber aus vielen Tokens. Das /100-Words-Ziel trieb nach oben, obwohl die Kompression gut war.
**Regel v2:** Code-Gate auf `tokens_per_kb` umgestellt (Target ≤350 tokens/KB ≈ ≥2.9 bytes/token). EN/DE bleiben auf /100-Words (dort ist die Metrik stabil).

### L-013 - shuf auf grossen Trainings-Files killt Disk-IO des Trainers
Beim v1-Lessons-Audit am 2026-04-26 wurde \`shuf -n 5 english.txt\` (56 GB) waehrend eines aktiven Trainings ausgefuehrt.
\`shuf\` muss die gesamte Datei lesen fuer die Reservoir-Sampling-Garantie - mehrere parallele shuf-Prozesse
haben den Disk-Channel blockiert, Trainer-tok/s stuerzte von 33k auf 2.4k, data_wait stieg auf 93%.

Plus: parent-bash-Loops haben shuf-Aufrufe nach SIGKILL respawned. Erst nach Killen des parent-bash war Schluss.

**Regel v2:**
- NIE \`shuf\` auf grossen Trainings-Files waehrend Training laeuft
- Fuer Sample-Review: \`head -n 5\` (deterministisch, sub-second), oder \`sed -n 12345,12349p\` fuer mid-file
- Fuer echtes Random-Sampling: vorher einmal offline mit \`shuf > sample.txt\` machen, dann nur diesen kleinen sample lesen
- Generell: jeder Bash-Befehl auf Files > 1 GB sollte vor dem Ausfuehren geprueft werden auf Disk-Last (\`iostat -xz 1 2\`)

### L-014 — Checkpoint-Rotation muss Nicht-Step-Suffixe tolerieren
Health-Guards können Notfall-Checkpoints `step_<n>_emergency.pt` schreiben. Die Rotation in `trainer.py` matchte `step_*.pt` und parste den Namen positions-basiert (`int(name.split("_")[1])`) → `ValueError: invalid literal for int() with base 10: '10_emergency'`. Crash genau im Auto-Stop-Pfad direkt nachdem der Emergency-Ckpt geschrieben war — der zu rettende Run starb am Aufräumen.
**Regel v2:** Step-Extraktion aus Checkpoint-Filenames immer regex-basiert (`r"step_(\d+)"`), nie positions-basiert spalten. Auto-Stop-Pfad muss idempotent sein und darf nicht an Sekundärfehlern aus dem eigenen Cleanup sterben.

### L-015 — Cross-Cutting Module müssen ALLE Trigger-Layer enumerieren
Shared `RotaryEmbedding` wurde in `helix_model.py` nur instanziiert wenn mindestens eine `sparse_attention`-Layer im Stack war. Architekturen rein aus `plain_attention` mit `use_rope=true` bekamen `rope=None` durchgereicht — Position-Encoding stumm fehlend, da PlainAttentionLayer mit `rope=None` einfach ohne Rotation rechnete. Die existierenden Tests prüften nur Shape und Causality, der Bug wäre erst beim Eval (Long-Context-Regression) aufgefallen.
**Regel v2:** Wenn ein geteiltes Modul von ≥2 Layer-Typen benötigt werden kann, muss die Build-Condition ALLE relevanten Layer-Configs prüfen (`any(needs_rope(l) for l in layers)`), nicht nur einen Trigger-Typ. Plus: für jede Sub-Architektur einen Smoke-Test mit numerisch-sensitiver Assertion (z.B. dass Permutation der Position-IDs das Ergebnis ändert) — nicht nur shape/causality.

### L-016 — `pgrep -f` in Wait-Wrapper matcht den Wrapper selbst
Ein chained Trainings-Wrapper `bash -c 'while pgrep -f "train_phase.*runde3"; do sleep 30; done; python sweep.py ...'` hat den Trainer überlebt aber den Sweep nie gestartet. Grund: das python-Argument im wrapper-bash enthielt beide Strings, also matchte `pgrep -f` die wrapper-bash selbst → wait-Loop nie terminierbar.
**Regel v2:** In Wait-Wrappern entweder
- Pattern unmatchable für die eigene command-line machen — Trick: erste Buchstaben in Char-Class verstecken, z.B. `pgrep -f "[t]rain_phase.*runde3"` (matcht den Original-Prozess, nicht das pgrep-Argument selbst weil `[t]` als regex-class ≠ literales `[t]`).
- Oder PID-basiert warten: vor dem `wait`-Loop Trainer-PID festhalten und `kill -0 $PID` als Liveness-Check nutzen — kein String-Matching.
- Generell: Wrapper-Skripte vor detached-Start mit `bash -n` syntax-checken UND mit kurzem dummy-Trainer trockenlauf.

### L-017 — Helpful-Elaboration-Trap bei honest_refusal-SFT-Generierung
Bei Phase-3-SFT-Daten-Generierung mit DeepSeek-V4-Flash via OpenRouter: ein generisches "Du bist ehrlich, halluziniere nicht"-System-Prompt erreichte trotzdem ~3% Halluzinations-Rate auf historische false-premise-Prompts. Konkretes Beispiel: bei "Wer entwarf den Bürostuhl in Goethes Arbeitszimmer?" produzierte das Modell in 2 von 9 Samples konfidente Fabrikationen — einmal "Johann Friedrich Funk (1706-1775)", einmal "Friedrich Justin Bertuch ließ 1794...". Beide klangen plausibel, beide waren erfunden.

Root cause: das Modell versucht aus Höflichkeit Kontext zu liefern und konfabuliert dabei spezifische Details (Namen, Daten, Jahre). Reines Verbot reicht nicht — das Modell weiß nicht *welche* Details es nicht sagen darf.

**Regel v2 — Anti-Halluzinations-System-Prompt für SFT-Generation:**
- ❌ NICHT ausreichend: *"Niemals halluzinieren. Sag offen wenn du nicht weißt."*
- ✅ AUSREICHEND mit drei Komponenten:
  1. **Explizit verbotene Spekulations-Marker** ("vermutlich", "wahrscheinlich", "soll", "angeblich", "wohl", "vielleicht war")
  2. **Few-Shot-Beispiele für GUTE vs SCHLECHTE Refusals** (zeig dem Modell konkret was du willst)
  3. **Erlaubt: verifizierbarer Kontext-Debunk** (z.B. "Goethe besuchte kein klassisches Gymnasium..." — verifizierbar, hilft Frage einzuordnen) vs verboten: alternative spezifische Details ("Bürostuhl entwarf vermutlich X im Jahr Y...")
- A/B-Test: 0% Halluzinations-Rate auf 310 Test-Records (vs ~3% Baseline), avg out-tokens 143 statt 241 (konziser durch verbotenes Filler-Geschwafel).
- Zusätzlich: Refusal-Auto-Detection-Regex muss BREIT sein — "Ich weiß **es** nicht" matchet nicht "weiß nicht" (nicht-zusammenhängende Wörter), Regex auf einzelne Schlüsselwörter ("weiß", "unbekannt", "nicht überliefert") robuster.

---

## Neue Lessons aus dem Edu-Filter + Multi-GPU (2026-05-31)

### L-018 — Thinking-Modelle: `max_tokens` deckt Reasoning UND Antwort (Kostenfalle)
gemini-3.5-flash als 0-5-Edu-Judge: bei `max_tokens=200` kamen nur ~6 sichtbare Tokens an — die ~190 Thinking-Tokens fressen dasselbe Budget, die `Bewertung:`-Zeile wird abgeschnitten (25/25 unparsed). Schlimmer: Thinking-Tokens werden als **teurer Output** abgerechnet → ein vermeintlich "billiger Flash" kostete **€24** für ~12k Annotationen.
**Regel v2:** Bei Thinking-Modellen `max_tokens` großzügig (≥512) **und** `reasoning_effort` drosseln. Für reine Klassifikations-/Rating-Tasks (0-5) ein **non-thinking**-Modell wählen — schneller, vorhersagbar, ~10-50× billiger. Token-Budget VOR dem Vollauf an einem 25-Doc-Smoke prüfen (succeeded vs parsed getrennt zählen).

### L-019 — Judge-Wahl für Daten-Filterung: billig ≠ schlechter, streng ≠ falsch
Umstieg gemini-3.5-flash → `qwen3-235b-a22b-2507` (non-thinking, OpenRouter): ~40× billiger UND der **bessere** Judge. Qwen bewertete Web-Spam/Reviews/EuroParl-Fragmente korrekt mit 0-1, wo Gemini großzügig 3 gab — genau diese Laschheit erzeugte den Over-Keep des Gemini-trainierten Klassifikators auf german_commons (64 % statt 45 %). FineWeb-Edu selbst nutzte Llama-3-70B (non-thinking), kein Frontier-Thinking-Modell.
**Regel v2:** Für Edu-/Quality-Rating ein solides dichtes/MoE-Instruct-Modell (Gemma-3-27B, Qwen3-235B-2507, Llama-3.x-70B) statt teurem Frontier-Thinking. EINEN Judge konsistent halten (kein Judge-Mix im Trainingsset). Judge an konkreten Roh-Begründungen validieren, nicht nur an der Score-Verteilung.

### L-020 — german-commons ist OCR-historisch-dominiert (verstärkt L-004)
Beim Versuch, german-commons als Skalierungsquelle neu zu ziehen: der HF-Stream ist **front-loaded mit digitalisierten historischen Büchern** (Quellen `BLBooks`, `DiBiLit`, `DiBiPhil`, `GermanPD`; Perplexity 500-1000+; `subset`-Feld nutzlos = `'0'`). In 8000 gestreamten Docs **kein einziges** modernes (<200 ppl). Die "72B News / 54B Cultural" der Dataset-Karte sind großteils OCR-Archive (falsches Register, Fraktur-Fehler). Unser alter `max_perplexity=500` + `cultural_keep_ratio=0.05`-Filter hat das (richtig) rausgehalten — wir landeten bei der sauberen, aber edukativ dünnen Parlaments-Schicht.
**Regel v2:** german-commons ist **kein** Modern-Deutsch-Skalierungsgewinn. Für mehr hochwertiges modernes Deutsch: RedPajama-V2-de (3T, mit Quality-Signals) + mehr fineweb2_de, beide edu-gefiltert. Bei jeder neuen Streaming-Quelle erst die Subset-/ppl-Verteilung der ersten N Docs proben, bevor man auf Token-Budget streamt.

### L-021 — Ridge-Regressor schrumpft zum Mittel → Entscheidungsschwelle kalibrieren
Der Edu-Klassifikator (Ridge auf e5-Embeddings) sagt Scores Richtung Mittelwert geschrumpft vorher. Eine harte Schwelle bei 3.0 ergab Precision 0.99 / **Recall 0.66** (warf ~1/3 echter ≥3-Docs weg). Die auf dem Train-Split max-F1-kalibrierte Schwelle (~2.4) brachte F1 0.79→0.89 und traf die echte Keep-Rate.
**Regel v2:** Bei Regressions-Filtern nie die nominale Label-Marke als Entscheidungsschwelle nehmen — Schwelle auf Train kalibrieren (max-F1 oder Ziel-Keep-Rate) und im Artefakt speichern. Per-Source-Keep-Raten aus einem billigen 400er-Verteilungs-Sweep liefern die Kalibrierungs-Anker.

### L-022 — DDP additiv + gated einbauen, Checkpoints DDP-agnostisch
Multi-GPU (DistributedDataParallel) in den Single-Process-Trainer eingebaut: streng `WORLD_SIZE>1`-gated, damit der Single-GPU-Pfad bit-identisch bleibt (der laufende Foundation-Run darf nicht brechen). Zwei Fallen: (a) `DDP(model).state_dict()` hängt jedem Key `module.` an → ein Multi-GPU-Checkpoint lädt nicht mehr single-GPU; Lösung: immer `model.module` (das unwrapped Core-Modell) speichern/laden. (b) Eval/Stop/Logging müssen rang-koordiniert sein, sonst hängt ein DDP-Collective: Rank-0-Eval (forward-only, kein Collective) + Barrier, globaler Stop via `all_reduce(MAX)`.
**Regel v2:** Distributed-Code additiv und gated einbauen; Single-GPU-Pfad per py_compile + dry-run verifizieren. Checkpoints immer ohne `module.`-Prefix schreiben. Multi-GPU-Korrektheit braucht echte 2-GPU-Validierung (RunPod) — eine Single-GPU-Box beweist sie nicht.
