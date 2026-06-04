# SFT Teacher Pipeline — Grobentwurf (nach Phase 1)

Status: **Grobentwurf**. Detail + Qwen-Prompt-Templates + Judge-Kalibrations-
Set werden nach Canary Runde 2 in `INVESTIGATIVE_SYNTH_DATA.md` ausgearbeitet.

## Leitgedanke

Zwei getrennte SFT-Ströme, beide aus denselben Seed-Passagen gespeist:

1. **Content-SFT** — lehrt Domänenwissen  
   Input: Frage über Thema X  
   Output: faktenpräzise Antwort, mit Quellenanker wenn der Seed-Text einen bot

2. **Structural-SFT** — lehrt **Prozess** statt Fakt ("learn to learn")  
   Input: Rohinhalt aus Seed  
   Output: strukturierte Zerlegung (Q&A-Liste, Causal-Chain, Bullet-Summary,
   Confidence-markierte Unterfragen)

Michaels Kernformulierung: „das Modell soll mit seinen vorhandenen Daten
lernen zu lernen". Strom 2 ist exakt das.

## Pipeline-Phasen

```
cleaned/*.filtered.txt
       ↓
[ seed_collector.py ]           <- CPU, jetzt implementiert
       ↓
seeds/YYYY-MM-DD/{technical,factual,procedural,opinion,narrative,general}.jsonl
       ↓
[ qwen_synth_sft.py ]           <- nach Phase 1; Qwen 3.6 35B Apex via LocalAI
       ↓                           (zwei Output-Pfade pro Seed)
sft_synth/YYYY-MM-DD/
    content/*.jsonl             ← OSS-Instruct Output
    structural/*.jsonl          ← "zerlege diesen Text"-Output
       ↓
[ qwen_evol_instruct.py ]       <- optional, iterative Komplexitätssteigerung
       ↓
sft_synth_evolved/YYYY-MM-DD/...
       ↓
[ qwen_judge.py ]               <- Judge-LLM gates
       ↓
sft_curated/YYYY-MM-DD/...      ← endgültige SFT-Trainingsdaten
```

## Warum OSS-Instruct + Evol-Instruct kombiniert?

- **OSS-Instruct** (WizardCoder / Magicoder): Seed-Dokument inspiriert
  realistische User-Aufgaben. Verhindert „KI-imitiert-Benchmark"-Sound.
- **Evol-Instruct** (WizardLM): einfache Aufgabe → komplexer durch
  Constraints / Reasoning-Tiefe / Breite. Löst das Problem dass reine
  OSS-Instruct-Aufgaben meist zu einfach sind.

Kombiniert: OSS gibt Realismus + thematische Verankerung, Evol gibt
Schwierigkeitsgrade. Curriculum-Effekt in den Trainingsdaten.

## Judge-LLM-Gates (Pflicht für alle Samples)

Aus `DATA_PIPELINE_V2.md` §4:

1. Dedup gegen bestehende Train-Samples
2. Dedup innerhalb Batch (near-dup MinHash)
3. Widerspruch zu `facts.yaml` → drop (für Topic-LoRAs später)
4. Length-Gate (20 ≤ answer_len ≤ 800 chars)
5. Stil-Konsistenz (Cosine-Sim vs. Referenz-Stil)

**Zusätzlich für factual / wissenschafts-nahe Kategorien** (aus
Modus-B-Vereinbarung 2026-04-24):

6. Ground-Truth-Anker Pflicht (Quelle zitiert, Judge prüft Existenz)
7. Konsens-Kalibrationsset-Check: widerspricht Sample einer der ~100
   Konsens-Positionen → drop oder nur mit peer-review-Quelle
8. Keine „cui bono"-Narrative in Modus-B-Themen (laufende Wissenschaft)

## Zwei konkrete Prompt-Typen (Detail in INVESTIGATIVE_SYNTH_DATA.md)

### (A) Content-SFT Prompt an Qwen

```
SYSTEM: Du bist Lehrer-Assistent. Aus dem folgenden Textausschnitt
erzeuge 3 realistische Nutzerfragen und ihre präzisen Antworten.
Jede Antwort muss aus dem Textausschnitt belegbar sein — keine
Halluzinationen, kein extern hinzugefügtes Wissen.
Format: JSON {frage, antwort, quellen_zitat}.

USER: [SEED-PASSAGE hier]
```

### (B) Structural-SFT Prompt an Qwen

```
SYSTEM: Du bist Wissens-Strukturator. Der folgende Rohtext soll in
eine lernbare Form zerlegt werden. Erzeuge:
  1. zentrale_aussage (1 Satz)
  2. unter_aussagen (Liste, je 1 Satz)
  3. nachfragen (Liste, je ein Q&A-Paar)
  4. offene_fragen (was der Text NICHT beantwortet)
  5. confidence_pro_aussage (high / medium / low / uncertain)
Kein externer Kontext. Nur das was der Text hergibt.

USER: [SEED-PASSAGE hier]
```

## Wichtige Nicht-Ziele

- **Kein** endloser Crawl. Die Seed-Zahlen pro Source sind deckelbar;
  Masse bringt ohne Qualität nichts.
- **Keine** synthetische Generierung *während* Phase 1 läuft —
  GPU-Konflikt mit Canary-Runs + Pretraining wäre tödlich.
- **Kein** separater „kreativer Generator"-Modus. Wir bleiben bei Seed →
  Transformation. Rein generativ (Qwen erfindet Thema + Text) ohne Seed
  neigt zu Benchmark-Sound und Stil-Kollaps.

## Nächste konkrete Schritte

Jetzt:
- [x] `scripts/data/seed_collector.py` existiert
- [x] `configs/data/seed_collection.yaml` existiert
- [ ] Collector im Container laufen lassen (CPU-only, parallel zur Pipeline)

Nach Canary Runde 2 fertig:
- [ ] `INVESTIGATIVE_SYNTH_DATA.md` mit vollen Prompts (Modus A / Modus B)
- [ ] `eval/scientific_consensus.yaml` mit 100 Kalibrations-Fragen
- [ ] `scripts/data/qwen_client.py` — OpenAI-kompatibler LocalAI-Wrapper
- [ ] `scripts/data/qwen_synth_sft.py`
- [ ] `scripts/data/qwen_judge.py`

Nach Phase 1 Pretraining:
- [ ] Erste volle SFT-Generation-Runde auf ~300k Seeds
- [ ] QC-Statistiken, Sample-Review, Judge-Kalibrierung
- [ ] Phase 3 SFT-Training mit TRL auf den kuratierten Samples

## §7. Creative Writing — Songs / Texte (vier Säulen, keine Lyrics-Reproduktion)

Ziel: Modell kann eigenständig Texte schreiben (Songs, Gedichte, Prosa) und
versteht **warum** bestimmte Texte Menschen berühren. Das letzte Stück ist
das wichtigere — Struktur allein macht keinen Song funktionieren, die
**emotionale Resonanz** tut es.

Vier Säulen, die parallel gespeist werden:

| Säule | Datenquelle | Was gelernt wird |
|---|---|---|
| **A — Theorie** | Songwriting-Bücher, Blogs, Music-Theory-Papers (S2ORC-Subset) | Handwerk: Form, Reim, Metrik, Hook-Prinzipien |
| **B — Public-Domain-Texte** | Gutenberg-Volkslieder, Kirchenlieder vor 1925, klass. Gedichte (Goethe, Heine, Shakespeare) | Wie konkrete Texte in der Form aussehen |
| **C — Qwen-Generation** | Teacher erzeugt originale Texte nach expliziten Struktur-Vorgaben | Anwendung + Variation |
| **D — Reception-Discourse** | Music-Reviews, Genius-Annotations (Analyse-Text, NICHT Lyrics), Songwriter-Interviews, Music-Psychology-Papers, Reddit r/Music-Diskussionen | **Warum** Menschen Songs mögen |

**Urheberrechts-Regel, nicht verhandelbar**: Kein Training auf urheberrechtlich
geschützten Songtexten, auch nicht als KI-Paraphrase. Paraphrase ist
rechtlich eine Bearbeitung (UrhG § 23) und benötigt Zustimmung. Säule C
(Qwen generiert komplett neu) liefert die "hat-einen-Text"-Fähigkeit; Säule
D (Diskurs über Songs) liefert die "weiß-was-berührt"-Fähigkeit. Zusammen
erreichen sie das Ziel ohne Lizenzrisiko.

**Emotional-Resonance-Patterns** (Unterkategorie von Säule D):
- Spezifität → Universalität ("deine schwarze Katze schlief auf dem
  Kissen" statt abstrakt "Trauer") — konkrete Bilder triggern geteilte
  Erinnerungen
- Anker an den großen Sechs: Liebe, Verlust, Verlangen, Trotz, Verachtung,
  Verbundenheit
- Wahrheit > Konformität — warum Cash' "Hurt"-Cover härter trifft als
  das Original
- Kontext-Sensitivität: welcher Stil zu welcher Emotion passt
  (Battle-Rap ≠ Wiegenlied, beide gültig in ihrem Kontext)

Reception-Lizenzmatrix (Säule D):

| Quelle | Lizenz | Status |
|---|---|:-:|
| Genius-Annotations (via API) | CC-BY-SA (analytischer Text) | ✓ |
| Music-Psychology-Papers via OpenAlex | Open Access / CC-Varianten | ✓ |
| Pitchfork / Rolling Stone / laut.de Reviews | Pressezitatrecht, Einzelzitate OK, Massen-Crawl ToS-heikel | ⚠ manuell kuratieren |
| Reddit r/Music, r/hiphopheads etc. | ältere Pushshift-Dumps auf HF, neue API dicht | ⚠ Snapshot-basiert |
| Songwriter-Interviews (NPR Tiny Desk, Song Exploder) | Pressezitat, Transcripts teils frei | ✓ einzelne Folgen |
| Songfacts / SongMeanings | User-Content, Einzelzitat OK | ⚠ |

Download-Pfad in der Pipeline: W1.3+W1.4 heute (Theorie + Public Domain),
W1.5 erst nach Einzel-Lizenz-Check (Reception-Quellen).

## §8. Troubleshooting / Problem-Solving (Windows, Hardware, Handy, Netzwerk)

Ziel: Modell kann strukturiert diagnostizieren. Das gleiche Meta-Pattern wie
code-write+test aus §1 — nur auf System-/Hardware-Ebene.

**Training-Pattern** (jedes SFT-Sample hat diese Struktur):

```
1. Problem-Beschreibung (oft vage / emotional: "mein pc ist langsam")
2. Clarification-Runde (OS-Version? Was wurde schon versucht? Seit wann?)
3. Diagnose-Chain (systematisch: Kabel → Treiber → Config → Software → Hardware)
4. Lösung (konkrete Schritte, einer nach dem anderen)
5. Verifikation (woran erkennst du dass es jetzt geht?)
6. Escalation (wenn nicht: was als nächstes? Wer kann noch helfen?)
```

Das Modell lernt: **erst klären, dann lösen**. Wichtiger Unterschied zu
"Google-Suchergebnis wiedergeben" — echte User beschreiben Probleme ungenau;
die KI muss gezielt nachfragen bevor sie vermutet.

**Datenquellen (W1.6, läuft gerade):**

- `HuggingFaceH4/stack-exchange-preferences` — Q&A mit Upvote-basierten
  Präferenzen. Enthält Stack Overflow + superuser + serverfault +
  askubuntu + apple + android + electronics + networkengineering.
  CC-BY-SA. Perfekt für Phase 4 ORPO-Preference-Paare zusätzlich.

**Später (W2):**
- Vollständige per-Site-Dumps pro Stack-Exchange-Subdomain, wo wir nach
  "superuser", "serverfault", "askubuntu" zielgenau filtern können
- Arch Wiki, Ubuntu Wiki, Gentoo Wiki als Linux-Troubleshooting-Kern
- Subset aus `bigcode/the-stack-github-issues` für Bug-Report → Fix-PR-Paare

**Später (W3, Qwen-Stage nach Phase 1):**
- **Deutsche Übersetzung + Adaption** der englischen Stack-Exchange-Seeds
  durch Qwen. 90 % der Troubleshooting-Welt ist englisch, aber ein
  deutscher User sucht deutsch — wir schließen die Lücke synthetisch.
- Evol-Instruct auf Problem-Beschreibungen: einfach ("Maus geht nicht")
  → komplex ("Maus macht komische Dinge bei USB-Hub mit Drucker + anderen
  Geräten auf Win11 24H2 nach letztem Update")
- **Emotions-Kalibrierung**: viele Troubleshooting-Anfragen kommen genervt
  ("seit 3 Stunden versuch ich..."). Modell soll: kurz acknowledgen
  (nicht übertrieben), dann strukturiert helfen. Stack-Exchange-Ton ist
  die Referenz: knapp, empathisch, sachlich.
