# HISTORY — Auralis v2

Chronologisch, Milestones. Append-Only.

---

## 2026-04-22 — Projekt-Start Auralis v2
- Brief + 7 Phasen-SPECs + `SPEC_DATASETS.md` in `Doc/` finalisiert
- Projekt-Skelett angelegt (`pyproject.toml`, Verzeichnisbaum, `.gitignore`)
- `STATUS.md`, `LESSONS.md`, `HISTORY.md` initialisiert
- Memory-System für Claude Code befüllt (User / Projekt / Feedback / v1-Lessons / v1-Datasets)
- **Entscheidung: Modellgröße 1B** (statt 2-3B) — schneller, günstiger, v1-Kapazität verdoppelt aber bleibt klein.
- **Entscheidung: Daten-Root `<DATA-ROOT>`** (NAS, 25 TB frei); v1-SFT-Pool `I:/Auralis/NEWGPT/data/` bleibt lokal.
- Phase-0-Vorarbeit komplett: Baseline (50 Fragen), byte-gleicher Prompt-Builder + Tests, Daten-Config + 3 Phase-1-Download-Scripts + v1-Inventory-Script.
- Leitlinie verankert: synthetische Daten-Generierung ist erwünscht, wenn Open-Source-Quellen dünn sind (DeepSeek V3 / Qwen 3.5 30B lokal — v1-erprobt).

## 2026-04-22 — Phase 0 abgeschlossen (Tokenizer + Phase-1-Daten)
- **v1-DE-Reuse:** 23.7 GB dedupliziertes deutsches Pretraining-Material auf NAS (~4.7 B Tokens, 8.87 M Docs), 9:35 Min.
- **EN-Downloads:** Wikipedia EN (12 GB), FineWeb-Edu sample-10BT (40 GB), OpenMathInstruct-2 (8 GB) — zusammen ~15 B Tokens.
- **Code-Downloads:** StarCoderData 9-Sprachen-Subset (3.5 GB) + open-web-math (0.88 GB) — zusammen ~1.25 B Tokens.
- **Nicht geladen** (Dataset-HTTP-404 oder `datasets` v4+ script-ban): SlimPajama, Dolma, Proof-Pile-2. Gesamt-Deckung Phase 1 = **~21 B / 25 B = 84 %**, Lücke für Phase 2 reserviert.
- **Tokenizer-Korpus:** 15.5 GB Mix (50 EN / 40 DE / 10 Code), NUL-bereinigt.
- **Tokenizer-Training (SentencePiece Unigram, 200 k Vocab, 32 Threads):** 14.6 Min.
- **Quality-Report PASS:** EN 123 tok/100 w (Ziel ≤135), DE 133.8 (≤150, v1 war ~220), Code 313.6 tok/KB (≤350), Unknown 0 %, **Chat-Template-Roundtrip byte-exakt**.
- **Neue Lessons L-007..L-012** in `LESSONS.md`: SP-Normalisierung `identity` zwingend, NUL-Strip-Pflicht, `num_threads ≥ 1`, `input_sentence_size = 5 M` bei 32-GB-RAM, HF v4 `Dataset-scripts` verbot, Code-Metrik auf `tokens/KB` umgestellt.

## 2026-04-23 — Phase 0.5 abgeschlossen (Modell-Architektur)
- `src/auralis/model/` komplett implementiert: config dataclass, RMSNorm, SwiGLU-FFN, Mamba-2 (pure torch), Gated Linear Attention (pure torch), Sparse Attention mit Sliding-Window + Global Tokens, RoPE, Scaled-Normal Init, KV-Cache-Dataclass.
- `helix_model.py`: `HelixBlock` pre-norm style, `HelixModel` mit heterogenem Stack aus Config, `build_model(yaml)` Factory, tied-embedding LM-Head.
- Zwei Configs: **`helix_v2_100m.yaml`** (8 Layers, 134 M Params, Test-Modell für CPU) und **`helix_v2_1b.yaml`** (28 Layers, d=1280, ~954 M Params — trifft 1B-Ziel innerhalb 5 %).
- **50/50 Tests grün** in 2.7 s auf CPU: Config-Loading + Validation, alle Layer (Shapes, Backward, Causal-Masking, Window-Masking, Global-Tokens-Bypass, RoPE-Norm-Preservation), End-to-End Forward/Backward auf 100M-Modell.
- Forward auf frisch-initialisiertem 100M-Modell liefert Loss **12.37** — nahe dem Theorie-Wert von `ln(200000) = 12.20` für uniform-prior über das 200k-Vocab. Keine NaN/Inf.
- Pure-Python-Varianten von Mamba-2 und GLA liefern die Referenz-Semantik für CPU-Tests. Für echtes GPU-Pretraining wird Phase 1 zusätzlich `mamba_ssm` und `flash-linear-attention` einbinden (Config-Flag, Interface bleibt gleich).

## 2026-04-23 — Phase 1 Pretraining-Pipeline fertig (Launch-ready)
- `scripts/data/tokenize_for_pretraining.py`: batched SentencePiece-Encoding → uint32 .bin + int64 .idx pro Sprache, atomare Writes + Manifest, Resume-safe.
- `src/auralis/training/` komplett: `PretrainDataset` (memmap), `MixedDataLoader` mit largest-remainder-Partitionierung für Mix-Ratios, `build_optimizer` mit decay-Split für Normen/Biases, `build_scheduler` (cosine + constant_with_warmup), `PretrainTrainer` mit Gradient-Accumulation, Grad-Clip, Checkpoint-Rotation, NaN-Abbruch, Val-Loss-Alarm nach 3 Regressions.
- `scripts/pretrain/train_phase1.py`: CLI-Entry mit Preflight + Resume + Device-Override + `torch.compile`-Flag.
- `scripts/pretrain/smoke_test.py`: End-to-End-Validation in 30 s (134M-Modell, 20 Steps, synthetische Tokens, Checkpoint-Roundtrip).
- `configs/training/phase1_pretrain.yaml`: 80 k Steps × 128 effective Batch × 2048 Tokens = ~21 B Tokens (matched zur tatsächlichen Daten-Deckung).
- **64/64 Tests grün in 4 s** (+14 neue Training-Tests: `dataset`, `optimizer`, `trainer`).
- Tokenization der 88 GB Phase-1-Daten startete parallel (Background) — Durchsatz ~6 MB/s auf SMB, geschätzte Dauer ~4 h.
- **Launch-Guide** `docs/PHASE_1_LAUNCH.md` deckt RunPod-Setup, Preflight, Monitoring, Rollback-Prozeduren und Milestone-Erwartungen ab.
- **GPU-Launch selbst wurde NICHT automatisch gestartet** — das kostet $500-800 auf RunPod und braucht Michaels explizite Entscheidung + Account-Setup.

## 2026-04-23 — Blackwell-GPU-Validation auf Unraid
- Auralis-Docker-Container (Ubuntu 22.04, Python 3.11, torch 2.7.0+cu128) auf Unraid-Host mit RTX PRO 5000 Blackwell (47 GB VRAM) benutzt.
- V2-Datenverzeichnis via `mount --bind <repo> /data/NEWGPT/v2data` + `docker restart` in den Container eingehängt (SHFS propagiert laufende Mounts nicht).
- Libraries installiert: `flash-linear-attention`, `mamba-ssm 2.3.1`, `causal-conv1d 1.6.1`, `flash-attn 2.8.3` (alle cu128-kompatibel). Nach Triton 3.6-Upgrade kompilieren beide Triton-basierte Libraries (mamba-ssm + fla).
- **Library-Swap-Hooks eingebaut**: Pro Layer-Typ separat aktivierbar per Env-Var (`AURALIS_USE_CUDA_KERNELS`, `AURALIS_USE_MAMBA_KERNEL`, `AURALIS_USE_GLA_KERNEL`, `AURALIS_USE_FLASH_ATTN`). Interface für `HelixBlock` unverändert; default bleibt native pure-torch. GLA-Backend unterstützt gleiche Parameter-Shapes native/fla, Mamba-Backend nicht (Architektur-Unterschied).
- **Smoke-Test-Resultate (250M, bf16, batch=4):**
  - seq=256 native: 147 tok/s, 13.0 GB VRAM, Loss 12.16 → 11.59 (Δ+0.57, lernt)
  - seq=512 native: 82 tok/s, 24.85 GB VRAM
  - seq=512 gla-kernel: 88 tok/s, **21.27 GB VRAM (-14 %)**, numerisch identisch zu native
  - seq=512 gla+flash: identische Resultate (Sparse macht nur 3/12 Layer)
- **Blackwell-Erkenntnis:** Hauptvorteil der Kernels ist **VRAM-Ersparnis** (chunkwise statt materialisiertem [B,L,H,D,D] State), nicht primär tok/s. tok/s-Speedup wird bei seq=2048 Phase-1-Config deutlicher.
- **Mamba-Kernel auf Blackwell aktuell problematisch** — Triton compile bug in mamba_ssm selbst mit Triton 3.6. Für RunPod H100/H200 ok, für Blackwell Mamba native lassen.
- **(Später am selben Tag) Blackwell-Fix gefunden: `TRITON_OVERRIDE_ARCH=sm89`** — emuliert Ada (Compute-Capability 8.9) auf Blackwell (sm_120). Weder sm90 (WGMMA-Intrinsics fehlen) noch default (sm_120 unbekannt) funktionieren; sm89 ist das erste abwärtskompatible Target das Blackwell akzeptiert. Alle drei Kernels (mamba_ssm + fla + flash-attn) laufen jetzt gleichzeitig, numerisch identisch zur native-Referenz.
- **Final-Messungen Blackwell, 250M bf16, ALLE Kernels aktiv:**
  - seq=256 batch=4: 220 tok/s, 6.68 GB VRAM (-49 % vs native)
  - seq=512 batch=8: 1 928 tok/s, 16.36 GB (23× vs 3090 pure-python)
  - seq=1024 batch=4: 3 628 tok/s, 16.84 GB
  - **seq=2048 batch=2: 2 713 tok/s, 17.74 GB — Phase-1-Config validiert, Loss Δ +1.52 in 15 Steps**
- Docs (`docs/PHASE_1_LAUNCH.md`) um sm89-Workaround und per-Hardware Kernel-Setup erweitert.

---

## 2026-04-26 — Canary Runde 3 + Code-Review-Bugfixes + 1B-Sweep
- **Canary Runde 2 b16 abgeschlossen** (3 Mix-Varianten, baseline=10:34Z): val_loss baseline 3.286 / de_heavy 3.652 / code_heavy 3.912. Baseline (12/3/1) gewinnt overall, de_heavy holt bei DE auf (6.500 → 6.280), code_heavy nur bei Code minimal (5.541 → 5.451) bei deutlich schlechterem Gesamtbild.
- **Runde 3 `de_medium_b16` (Mix 70/25/5, 11/4/1)** als Zwischenstufen-Validierung gefahren. Eval @ step 5000: **val_loss 3.653, EN 2.380, DE 6.280, Code 5.592** — praktisch identisch zu de_heavy, kein erwarteter Gewinn aus der Zwischenstufe. Baseline bleibt für 1B-Hauptlauf der bessere Kandidat overall, aber DE-Lücke (6.5 → 6.28) ist real.
- **Disk-Crisis bei step ~1175 von Runde 3:** tok/s kollabierten von 33k auf 2-4k, data_wait stieg auf 90%+. Root cause: parallel laufender v1-Lessons-Audit mit `shuf -n N` auf den 56 GB EN-Trainings-Files blockierte sdc auf 100% util. Cleanup: alle `shuf` + parent-bash-Loops killed (PIDs 444973/465158/492246), Trainer erholte sich auf 27-29k tok/s, Run lief sauber durch. Lesson L-013 angelegt. Kollateralschaden: chained Sweep-Wrapper war im selben process-group → musste später manuell nachgestartet werden.
- **Code-Review-Pass über Trainer + Modell + Dataset** brachte 3 Findings:
  - **P1 ([trainer.py:359-360](src/auralis/training/trainer.py#L359))** — `_rotate_checkpoints()` parste `step_<n>_emergency.pt` positions-basiert → `ValueError`, crashte den Auto-Stop-Pfad direkt nach dem Emergency-Save. Fix: regex-basierte Step-Extraktion. → L-014
  - **P1 ([helix_model.py:108-117](src/auralis/model/helix_model.py#L108))** — Shared `RotaryEmbedding` wurde nur gebaut wenn ≥1 `sparse_attention`-Layer; reine `plain_attention`-Stacks mit `use_rope=true` liefen ohne Position-Encoding. Fix: Build-Condition prüft jetzt alle Layer-Typen. → L-015
  - **P3 ([dataset.py:71-74](src/auralis/training/dataset.py#L71))** — Sampler-Off-by-one durch exclusive upper bound von `Generator.integers()`, letztes legales Window wurde nie gezogen. Fix: `+1` bei der Obergrenze.
  - 3 Regressionstests ergänzt (`test_emergency_checkpoint_does_not_break_rotation`, `test_pretrain_dataset_can_sample_last_valid_window`, RoPE-plain-attention smoke). Server-Run: **100/100 grün** (97 alt + 3 neu).
- **1B Batch-Size-Sweep** manuell nachgestartet (PID 932424) — testet `helix_v2_1b` × batch [1,2,4,6,8,12] × seq [1024,2048] mit Pflicht-Env-Block. Output: `logs/batch_sweep_1b.log`. Ergebnis steuert 1B-Phase-1-Hauptlauf-Config.
- **L-014 Follow-up-Fix:** zweiter Review-Pass deckte auf, dass die Regex `r"step_(\d+)(?:_.*)?$"` zwar den `ValueError` beseitigt, aber Regular- und Emergency-Ckpts am gleichen Step unter denselben Sort-Key fallen — bei Same-Step-Kollision wurde ein Snapshot je nach Glob-Reihenfolge stillschweigend rotiert (Emergency konnte verschwinden). Fix: Rotation auf strikt `r"step_(\d+)$"` verengt, Emergency-Snapshots sind jetzt komplett rotation-exempt. Regression-Test `test_emergency_checkpoint_survives_same_step_rotation` ergänzt. Server: **101/101 grün**.
- **2 verbleibende P2-Findings in STATUS.md → Bekannte technische Schulden** dokumentiert (gradient_checkpointing Override-Contract + Sweep `--no-grad-ckpt` ineffektiv) — beide ohne Production-Impact für aktuelle Configs.
- **P2-Backlog aufgelöst:** Beide Findings in einem zweiten Pass gefixt durch Extraktion der gemeinsamen Logik nach [src/auralis/training/utils.py](src/auralis/training/utils.py) (`resolve_gradient_checkpointing()` als echter Override + `apply_gradient_checkpointing()` für expliziten enable/disable). [train_phase1.py](scripts/pretrain/train_phase1.py) und [batch_size_sweep.py](scripts/utils/batch_size_sweep.py) nutzen jetzt den gemeinsamen Pfad. 4 neue Regression-Tests in [test_utils.py](tests/training/test_utils.py). Server: **105/105 grün**.
- **1B Batch-Size-Sweep abgeschlossen** (`helix_v2_1b`, mix 70/25/5):
  - seq=1024: max batch=12 OK, top tok/s bei batch=8 (3.4k tok/s, 19 GB)
  - seq=2048: max batch=8 OK, batch=12 → OOM
  - **Top throughput: seq=2048 batch=4 → 11.3k tok/s @ 23.3 GB peak** (23.7 GB Reserve auf 47 GB Budget)
  - Bemerkenswert: batch=8 holt nur +6% tok/s über batch=4, kostet aber +84% VRAM → **batch=4 ist der Sweet Spot**.
  - Empfehlung für 1B Phase-1-Hauptlauf-Config: **seq=2048, batch=4**, gradient_checkpointing=on. Volle Tabelle in `logs/batch_sweep_results.json`.
- **Zombie-Wrapper aufgeräumt:** PID 931991 (alter chained `wait && start_sweep` Wrapper vom vorigen Chat) hatte einen self-referential Bug — die wait-Condition `pgrep -f "train_phase.*runde3"` matchte den eigenen command-line-String (das python-Argument enthält beide Strings) und wäre nie terminiert. Hat den shuf-Cleanup überlebt aber nie den Sweep getriggert; mein manueller Sweep war glücklicherweise der einzige der lief. Lesson: Wait-Wrapper-Pattern muss `pgrep` so formulieren dass die eigene command-line nicht matcht (z.B. `pgrep -f "[t]rain_phase1.py.*runde3"` oder per PID-File statt pgrep).
- **Pre-Hauptlauf Config-Audit (3 Findings adressiert):**
  - **Kritisch:** [phase1_pretrain.yaml:74](configs/training/phase1_pretrain.yaml#L74) hatte noch das alte `monitoring.alert_on:` Schema mit Feldern (`val_loss_increase`, `grad_norm_explosion`, `nan_in_loss`), die der Trainer (`trainer.py:181` liest `monitoring.health.<feld>`) komplett ignoriert hätte — d.h. die User-Thresholds für den 1B-Hauptlauf wären stillschweigend gedroppt worden, nur HealthConfig-Defaults aktiv. Fix: kompletter rewrite auf echte HealthConfig-Schema (`monitoring.health.grad_explosion_threshold: 100.0`, `monitoring.health.val_regression_stop_k: 3`). Verifiziert per `HealthConfig(**cfg)` Roundtrip auf Server.
  - **Sweep-Result eingearbeitet:** [phase1_pretrain.yaml:44-45](configs/training/phase1_pretrain.yaml#L44) `batch_size_per_device: 8 → 4`, `gradient_accumulation: 16 → 32`. Effective batch bleibt 128, aber Microbatch nutzt jetzt den 1B-Sweep-Sweet-Spot (11.3k tok/s @ 23 GB statt suboptimaler 8er-Batch).
  - **Doku-Drift:** [docs/PHASE_1_LAUNCH.md](docs/PHASE_1_LAUNCH.md) — Install-Befehl `pip install -e ".[train]"` (extra existiert nicht in pyproject.toml) → `.[all-linux]` (zieht pretrain + posttrain + lora + inference + dev inkl. mamba-ssm/flash-attn/fla via pretrain-extra). Plus stale "64/64 Tests" → "105/105", stale "Tokenization läuft noch" → "Tokenization fertig (~21B Tokens)".
- **Verbleibende offene Empfehlung (nicht ausgeführt):** Token-Bins vor 1B-Hauptlauf per `rsync` von SMB nach lokaler NVMe im Container kopieren — größter operativer Stabilitätsgewinn nach der shuf/SMB-Crisis. Pending Entscheidung.
- **NVMe-Stage durchgezogen:** 69 GB Token-Bins per rsync von `disk6` (HDD) nach `/mnt/cache/auralis_tokens_local/curated_40b/` (NVMe btrfs). SHA256-verify byte-identisch, host-side `mount --bind` über `<repo>/tokenized/curated_40b`, Container restart, im Container als `/dev/nvme0n1p1 on /workspace/v2data/tokenized/curated_40b type btrfs` sichtbar. Sequential read benchmark: **NVMe 1.3 GB/s vs HDD 30.5 MB/s = 42× speedup**. Reboot-Persistenz bewusst ausgesetzt (User Scripts entry später nach erfolgreichem Hauptlauf). Kollateral: anderer Chat hat währenddessen `OpenText`+ReadLine-loops auf den 87 GB Roh-Textfiles parallel laufen lassen (zweiter L-013-Replay an einem Tag) — rsync-Throughput von 124 MB/s auf 17 MB/s eingebrochen, sofort gestoppt nach User-Notice, rsync erholte sich auf 124 MB/s. Memory `feedback_disk_diagnosis.md` geschärft mit aktiver Cross-Chat-Koordinations-Regel.
- **Pre-Hauptlauf Smoke-Pass (50 Steps, production-equivalent settings):** Neue [phase1_smoke.yaml](configs/training/phase1_smoke.yaml) als Fork von phase1_pretrain.yaml mit nur `total_steps: 50`, eval/save/wandb deaktiviert, separates `output_dir`. Smoke v1 deckte einen versteckten Mix-Bug auf: bei micro-batch=4 und code=5% rundete `_partition_rows()` (largest-remainder) auf 0 code-rows pro micro-batch → über 80k×32 micro-batches **nie code gesehen**. Anderer Chat hat den Fix gemacht: stratifiziert kleine Mix-Anteile fair über mehrere micro-batches statt pro micro-batch hart abzurunden, mit Erwartungswert-Logging im Trainer (`train expected rows/batch per language`). 27 Tests + Server-Suite 106/106 grün. Smoke v2 mit gepatchtem DataLoader: **tok/s 13.0k (vs Sweep 11.3k), data_wait 0.3-0.5%, VRAM 17.7 GB peak, Loss 12.41→8.85 in 50 Steps** — alle Health-Guards ruhig, code-rows kommen jetzt im erwarteten Anteil im Mix vor.
- **3 weitere Pre-Hauptlauf Config-Findings im Smoke-Vorbereitungspass:**
  - **Kritisch:** `data.data_dir` zeigte auf `<DATA-ROOT>` (SMB-Pfad UND alter Vorgänger-Subdir!) — der echte Hauptlauf wäre sofort gecrasht. Fix: `/workspace/v2data/tokenized/curated_40b` (Container-Pfad, via NVMe-bind).
  - **Mix-Inkonsistenz:** `mix_ratios` waren 75/20/5 statt Sieger 70/25/5 — Fix.
  - **`external_backup.path`** zeigte auf SMB-Pfad `//<HOST>/...checkpoints/phase1` (gleiches Pattern). Backup wäre wirkungslos durchgelaufen (Trainer fängt Backup-Errors ab). Fix: `/checkpoints/phase1_pretrain_backup` (Container-mount auf disk6 array, write-test verifiziert).
- **🚀 1B Phase-1 Hauptlauf gestartet** (~18:18 lokal): PID 225 im Container, detached. Config: helix_v2_1b (0.90B params), seq=2048 batch=4 grad_accum=32 (effective 128), mix 70/25/5, 80k steps (~21B Tokens), gc=on, torch_compile=on, --no-wandb (wandb war im Container nicht authentifiziert verifiziert, sicherer Pfad gewählt). Token-Reads von NVMe-cache, Checkpoints auf disk6 HDD (cache reserviert für Tokens). Erwartete Wall-Clock: **~12-19 Tage** auf RTX PRO 5000 Blackwell. Health-Thresholds aktiv (grad_explosion=100, val_regression_stop_k=3). Logs: `logs/phase1_pretrain.log`, primary ckpts: `checkpoints/phase1_pretrain/`, backups alle 10k steps nach `/checkpoints/phase1_pretrain_backup/`.

---

## 2026-04-29 — Phase-3 SFT-Daten-Pipeline + WSL-Inference-Setup
- **Trainer-Trajektorie sehr gesund:** 80k-Hauptlauf bei step 8340 (10.4%), val_loss-Trajektorie nach plan: 1k=3.44, 2k=2.37, 3k=2.05, 4k=1.92, 5k=1.84, 6k=1.76, 7k=1.74, 8k=1.68. Speziell EN val_loss unter 0.9 (perplexity ~2.46) — Modell lernt EN auf ~50% top-1 next-token-accuracy. tok/s steady 12.9k, data_wait 1-2%, kein Health-Trigger.
- **OpenRouter + DeepSeek V4 Flash/Pro Pipeline** für Phase-3-SFT-Daten-Generierung gebaut: [scripts/data/synth/deepseek_v4_client.py](scripts/data/synth/deepseek_v4_client.py) async client mit task-type-basiertem Pro/Flash-Routing. Pro für code-engineering (idiomatic patterns), Flash für tutorial/explainer (preference-confirmed via A/B-Test 2026-04-28). Resume-safe, cost-tracking, optional reasoning_content extract.
- **Phase-3 Batch1** (980 examples, $0.37): Smoke-quality validation across 11 task_types. Quality-audit: 100% korrekt auf code/math, 70% auto-clean refusals (Rest auch ok bei Manuel-Sichtung), keine offensichtlichen Halluzinationen. Erkenntnis: step_by_step_reason avg 5421 tokens — viel zu lang.
- **Phase-3 Batch2** (5600 examples, $1.96 + $0.04 retry, alle errors transient 504): Mit max_tokens=1500-cap auf step_by_step_reason (Verbosity 5421→1305 tok, -76%) plus DE-deep-Topics (Recht, Geschichte, Literatur, Sprache, DACH, ~1500 zusätzliche prompts). 132 transient errors via retry komplett aufgelöst, Dedup-Pass behielt nur 5600 saubere unique IDs. Aber: 2 echte Halluzinationen in honest_refusal entdeckt — beide auf "Wer entwarf Goethes Bürostuhl?"-Prompt (Funk/Bertuch konfabuliert). 7/9 Samples derselben Prompt korrekt refused.
- **Anti-Halluzinations-A/B-Test** (310 examples, $0.024): NEUES System-Prompt für honest_refusal mit (a) explizit verbotenen Spekulations-Markern (vermutlich/wahrscheinlich/soll/angeblich), (b) Few-Shot Good-vs-Bad-Beispielen, (c) erlaubtem verifizierbaren Kontext-Debunk. **Resultat: 0% Halluzinations-Rate** (vs ~3% Baseline), 91% explizite Refusal-Marker, avg-out 143 tok statt 241 (konziser durch verbotenes Filler-Geschwafel). → Lesson L-017 dokumentiert. Neues Prompt im [generate_phase3_inputs_v2.py](scripts/data/synth/generate_phase3_inputs_v2.py) adoptiert.
- **WSL2 + RTX 3090 Inference-Setup:** Komplette Linux-Inference-Umgebung lokal für regelmäßige Phase-1/2/3-Iterationen ohne Server-Disruption. Stack: WSL2 Ubuntu 24.04, Python 3.12, torch 2.11.0+cu128, mamba_ssm 2.3.1, causal-conv1d 1.6.1, flash-linear-attention 0.5.0 (alle Linux-only-Libs ohne Probleme). flash-attn nicht installiert (Server-Trainer nutzt sparse_attention:native, brauchen wir nicht). best.pt via scp lokal kopiert. Inference-Script lädt 0.90B in 5.6s, generiert mit **30-40 tok/s auf 3090**, peak GPU-mem 2.10 GB. Erste Outputs bei step 7000: classic Pretrain-state Verhalten — DE/EN-Grammatik perfekt, Faktenwissen unzuverlässig, Topic-Drift bei längerer Generation. Erwartet bei nur 8.75% des Hauptlaufs.
- **Cost-Bilanz Phase-3-Daten heute:** $2.37 von ~$11 OpenRouter-Budget, 7188 SFT-Examples in Pipeline (980 + 5598 + 300 v1 + 310 v2 honest_refusal), production-quality validiert.
- **Doku-Updates:** [LESSONS.md](LESSONS.md) L-017 (Helpful-Elaboration-Trap), [STATUS.md](STATUS.md) Phase-3-Daten-Status pending.

## 2026-04-30 → 2026-05-30 — Zwischenstand (rekonstruiert aus git-Log + `reports/`)

> Diese ~29 Tage waren nicht im HISTORY erfasst. **Rekonstruiert** aus dem git-Commit-Log (bis HEAD `0cfe26f`, 2026-05-04 — danach wurde nichts mehr committet) und den datierten Dateien in `reports/` und `docs/` (05-24 .. 05-30). Keine Erfindung — wo Lauf-Metriken fehlen, steht nur das, was die Artefakte belegen.

**~04-24 → 05-04 — Phase-2-Prep + industrielle Daten-Pipeline (git-belegt):**
- `feat: Phase-2 prep, DE-Chain, Politik-Korpus, infra hardening` + drei Codex-Review-Pässe (P1-P3, 6 Findings, 2 Edge-Cases).
- **Industry-standard Daten- + Eval-Pipeline** verdrahtet, **Track-2-Benchmark-Suite** (`feat(eval)`), opt-in Speed-Knobs für Phase-2+ (`feat(perf)`).
- Referenz-Docs: Attention-Varianten + Positional Encoding, MoRA-Integrationsplan, Daten-Pipeline/Framework-Evaluation (Tier-1 getestet), `data_pipeline_v1.md`.
- **Ask-LLM-Quality-Scorer** (Nemotron-CC-Stil): `rewrite_low_quality.py`, `ask_llm_code.py`, `ask_llm_local.py` (direct-HTTP für LocalAI/vLLM), chunked streaming + `--resume`, `--min/--max-chars`-Prefilter.
- **Scorer-Modell-Debugging:** DeepSeek-Garbage-Loop → temperature 0→0.05 → Default `llama-3.3-70b` → später `qwen3.6-35b-a3b` (lokales <host>-Modell). Deckt sich mit der späteren Judge-Lektion **L-019**.
- **RunPod:** phase1-resume-Config + Pod-Setup-Script.

**~05-24 → 05-26 — 500M-v5-Forensik + v6-Datenplan:**
- `500m_step6000_sft_gate` (05-24); `pretrain_v5_500m_a100_root_cause` (05-26) — Ursachenanalyse des schwachen v5-500M.
- `pretrain_v6_data_eval_plan`, `candidates_audit` / `manifest_combined`, `canary_500m_<host>` (05-26): neuer v6-Datenkandidaten-Pool definiert + auditiert.

**~05-27 — v6-Daten-Mixes gebaut + kontaminationsgeprüft:**
- Gutenberg-Books clean_v2/v1 (contamination + manifests + tokenized manifests), `books_augmented` / `books_lowratio` / `expanded_test`-500M-Mixes, `strict_mix`, `instruction_pool` / `instruction_de_strict`, extra candidates.
- Erste `sft_response_fix_de` v1/v2 (microfit, curriculum-mixed, diag eos/weighted, guard/core-only, stabilize-from-core).

**~05-28 — großer SFT-Reparatur-Sweep (500M):**
- `sft_response_fix_de` **v3 → v9** mit dutzenden Varianten (anchor, balance/guard-patch, family-from-v4best, bridge, bonn_photo, stable-from-v6/v8) gegen die Interferenz-Achsen (Bonn/Berlin, Photosynthese, Faust/Goethe).
- Semantic-Gate-Sweeps (a2, v3..v9, contrastive/balanced/strong-probe-tune) über die `sft_response_fix_chat_gate`-v2..v6-Holdouts.

**~05-29 — Pivot auf 1B-Readiness + Frozen/Live-Gates:**
- `1b_readiness_plan`, `staged_training_plan`, `codex_handoff_1b_readiness_v2`; **1B-Preflight** (v2 + curated_40b_v2) → `ready_to_launch: False`.
- **Frozen-Response-Gate v2** über die 500M-Checkpoints (v8_safe, hybrid_v1_40, hybrid_v12_bridge_60, hybrid_v12_repair_v2_80) + Leak-Checks → **kein Checkpoint promotable** (Target schwach, Retention bricht).
- **Adaptive Live-Bridge** (Frozen-Gate live im Training), `learning_neuro_map` + `learning_trace_system`, v10 (source-disjoint) / v11 (contrastive) SFT.
- STATUS-Snapshot 05-29: 500M nicht produktionsnah; Diagnose = **Interferenz**, Weg = sauber gewichteter 1B-Mix statt weiterer 500M-Mini-Patches.

**~05-30 — curated_40b-Canary / 1B-Mix-A/B (Übergang zu dieser Session):**
- `1b_language_tree_plan`, `1b_lr2e4_mix_ab_smoke`, `1b_readiness_preflight_curated_40b_canary`, `frozen_response_gate sft_response_v2 curated_40b_canary`.
- Übergang in den bilingualen **1B-Ramp (de55/en45)**, der in dieser Session bei Step ~3400 analysiert wurde (siehe nächster Eintrag).

## 2026-05-30/31 — Deutsch-Edu-Filter (FineWeb-Edu-Methodik) + Multi-GPU/DDP

- **Ausgangslage:** Bilingualer 1B-Ramp (de55/en45) bis Step ~3400 (best.pt), Lernverhalten enttäuschend. **Saubere Diagnose:** nicht die Eval (Qwen-2.5 auf den Probes 37/50 = sinnvoll, nicht kaputt), nicht die Architektur (All-Plain-Attention-Kontrolle ~ gleichauf mit Helix bis Step 300), sondern **Under-Training** (~3.4B Tok ~16% Chinchilla) **+ qualitäts-invertierter DE-Mix** (schwächste Quelle = größtes Budget).
- **Deutsch-Daten-Audit:** Heuristik-Refilter sinnlos (Daten nicht vermüllt, ~0.01% Drops). Echter Hebel = **Bildungswert-Filter** wie fineweb_edu (Englisch hatte Edu-Score, Deutsch nie).
- **Edu-Annotation gebaut** (`scripts/data/score_german_edu.py`, OpenAI-kompatibel): erst **gemini-3.5-flash** getestet → **€24 Kosten-Schock** (Thinking-Tokens fressen `max_tokens` + werden als teurer Output berechnet) → Lauf gekillt. Umgestellt auf **`qwen/qwen3-235b-a22b-2507` via OpenRouter** (non-thinking, ~40× billiger, **strenger UND genauer** auf Web-Text — Gemini war zu lasch auf EuroParl-Fragmenten). 12k Labels, **~€1**.
- **Verteilungen (Qwen, ≥3):** wikipedia_de 85 % · fineweb2_de 25 % · german_commons **4.8 %** (fast nur Parlaments-/OCR-Fragmente).
- **Cheap Klassifikator** (`edu_embed.py` frozen multilingual-e5-large + `train_edu_classifier.py` Ridge-Kopf + Schwellen-Kalibrierung): val **Pearson 0.866, Keep-F1 0.872**; reproduziert das LLM-Urteil auf Held-out (294 docs/s). `score_corpus_edu.py` filtert den Vollkorpus.
- **German-v2:** fineweb2_de @≥2.0 (~38 % Keep) + wikipedia_de ganz, **german_commons gedroppt** → ~2.0B hochwertige DE-Tokens (`configs/data_paths.curated_v2_german.yaml`). Reicht für den ~1.8B-DE-Bedarf des Foundation-Runs ohne Wiederholung.
- **Multi-GPU/DDP** in den Trainer eingebaut (`trainer.py`, `train_phase1.py`, `scripts/ops/run_pretrain_multigpu.sh`), strikt `WORLD_SIZE>1`-gated → Single-GPU bit-identisch (verifiziert). DDP-agnostische Checkpoints, `no_sync`, Rank-0-Eval+Barrier, globaler Stop. **Gemessen: 12.9k tok/s/GPU** → volles 1B (~20B Tok) ~18 Tage (1 GPU) / ~5 Tage (4 GPU). Noch nicht auf echter Multi-GPU validiert (Testbox = 1 GPU).
- **Dataset-Review** (vier User-Vorschläge): RedPajama-V2-de = echter Modern-DE-Skalierungshebel (3T, Quality-Signals); **german-commons verworfen** (Stream front-loaded mit OCR-Historik — BLBooks/DiBiLit/GermanPD, ppl 500-1000+; verstärkt L-004 → L-020); babylm-german zu klein; **multitask_german_32k** als SFT-Daten nach `raw/sft/` gesichert.
- **Infra:** Colab vs RunPod analysiert → RunPod billiger + praxistauglicher für Mehrtage-Läufe (Spot dank Resume); Colab ungeeignet. 1B-Foundation läuft gratis auf <HOST> (1 GPU).
- **Versionierung:** zwei fokussierte Commits auf `feat/multigpu-ddp` (DDP `eb4f833`, Edu-Pipeline `95d71ba`), gepusht, **PR #1** offen.
- **Neue Lessons:** L-018..L-022.
