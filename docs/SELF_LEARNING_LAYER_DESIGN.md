# Self-Learning Layer — Design (verifizierte aktive Lernschleife)

> **Zweck:** Architektur-Entwurf für einen *Selbst-Lern-Layer*, der in Helix
> eingebaut wird und zur Laufzeit neues Wissen erwirbt — nicht als großer
> Offline-Lauf, sondern als kleine, vom Modell selbst ausgelöste, **extern
> verifizierte** Online-Updates auf einem Adapter.
>
> **Status:** **ENTWURF / noch nicht implementiert.** Dieses Dokument ist die
> Design-Grundlage *vor* dem Code. Es baut bewusst additiv auf dem schon
> vorhandenen `src/auralis/adaptive/`-Modul auf. Es macht keine Versprechen über
> Qualität — die Schwellen sind alle noch zu kalibrieren (wie bei der
> Curriculum-YAML, siehe `adaptive/README.md`).
>
> **Experiment-Regel (aus `DOCS_INDEX.md`):** Nichts hieraus geht in den echten
> Pretraining-/Adapter-Stand, bevor eine Ablation ein klares Signal zeigt.

---

## 0) Was „selbst lernen" hier konkret heißt

Ehrlich vorweg, damit keine falsche Erwartung entsteht: Es gibt **keinen Trick**,
mit dem ein neuronales Netz ohne Gradienten dauerhaft neues Wissen *in seine
Gewichte* schreibt. „Wirklich lernen wie Training/Finetune" heißt zwangsläufig:
**es findet Training statt** — nur eben *online*, klein, kontrolliert und vom
Modell selbst ausgelöst. Der Unterschied zu einem normalen Trainingslauf ist
nicht *ob* trainiert wird, sondern **wann, wie oft, auf welchem Teil der
Gewichte und mit welchem Sicherheitsnetz.**

Drei Stufen, die Leute meinen, wenn sie „selbst lernen" sagen:

| Stufe | Mechanismus | Bleibt das Wissen? | In diesem Design |
|---|---|---|---|
| **1. Kontext-Gedächtnis** | Retrieval / Memory-Layer | nur im Prompt, weg nach Session | als **Arbeitsspeicher** genutzt |
| **2. Adapter-Online-Update** | LoRA/DoRA-Gewichte zur Laufzeit per Gradient | ja, in den Adapter-Gewichten | **das eigentliche Lernen** |
| **3. Voll-Backprop live ins 954M-Basismodell** | gesamtes Netz online updaten | ja, aber zerstörerisch | **bewusst ausgeschlossen** |

Stufe 3 führt zu *catastrophic forgetting* (jedes Update drückt altes Wissen
raus) und ist zu teuer. Dieses Design kombiniert **Stufe 1 als Kurzzeit-Puffer**
mit **Stufe 2 als Langzeit-Konsolidierung** — genau die „Beides kombiniert"-
Entscheidung.

---

## 1) Grundprinzip: eingefrorene Basis + ein einziger lernender Layer

Das zentrale Risiko jedes Online-Lerners ist *catastrophic forgetting*. Die
Lösung ist exakt das Designprinzip aus der README („eine breite, eingefrorene
Universal-Basis + Wissen/Skills als DoRA/LoRA-Adapter obendrauf"):

- **Basis (954M, 28L Hybrid) bleibt eingefroren.** Sie ist das stabile
  Weltwissen und kann nicht verlernt werden.
- **Ein kleiner, trainierbarer DoRA-Adapter** (Größenordnung 0,1–1 % der
  Parameter) ist das **einzige**, was sich zur Laufzeit ändert.
- Geht ein Update schief, wirft man den Adapter-Delta weg — die Basis ist
  unberührt. Rollback ist damit trivial und billig.

Genau diese Asymmetrie macht Online-Lernen überhaupt erst sicher.

---

## 2) Die verifizierte aktive Lernschleife (Kern dieses Designs)

Das gewünschte Verhalten ist kein passives „lerne aus jedem Input", sondern eine
**aktive, extern verifizierte Recherche-Schleife**. Beispielauftrag:
*„Informiere dich gründlich über Blumen."*

```
        ┌──────────────────────────────────────────────────────────┐
        │   VERIFIED ACTIVE LEARNING LOOP                           │
        │                                                          │
        │   (A) TASK            "Lerne gründlich über X"           │
        │         │                                                │
        │         ▼                                                │
        │   (B) RESEARCH        Helix sammelt Kandidaten-Fakten    │
        │         │             (eigenes Wissen + Retrieval/Tools) │
        │         ▼                                                │
        │   (C) DRAFT           strukturierte Claim-Liste über X   │
        │         │             je Claim: Aussage + Quelle + Conf  │
        │         ▼                                                │
        │   (D) VERIFY  ───►  externes Modell (z.B. GPT) prüft     │
        │         │           jeden Claim unabhängig.              │
        │         │           Konsens? Widerspruch? Lücke?         │
        │    ┌────┴─────┐                                          │
        │    ▼          ▼                                          │
        │  AGREE      DISAGREE / GAP                               │
        │    │          │                                          │
        │    │          ▼                                          │
        │    │     (E) GAP-FILL   gezielt fehlende/falsche         │
        │    │          │         Claims nachrecherchieren ──┐     │
        │    │          └──────────── zurück zu (B) ◄────────┘     │
        │    ▼                                                     │
        │  (F) CONSOLIDATE   nur verifizierte Claims → Adapter     │
        │         │          Mini-Online-Update (1–N Schritte)     │
        │         ▼                                                │
        │  (G) GUARD         frozen-gate: altes Wissen noch da?    │
        │         │          nein → ROLLBACK, ja → COMMIT          │
        └─────────┴────────────────────────────────────────────────┘
```

### Die Schritte im Detail

**(A) Task.** Ein Auftrag „lerne über X". Definiert das Lern-*Ziel* und die
Abbruchbedingung (genug verifizierte Claims / kein neuer Gewinn mehr).

**(B) Research.** Helix erzeugt Kandidaten-Wissen über X. Quellen, in
Prioritätsreihenfolge: (1) eigenes parametrisches Wissen, (2) Retrieval aus einem
lokalen Korpus / `<recall>`-Memory (siehe `docs/experimental/memory_kernel.md`),
(3) optional externe Tools/Suche. Output sind **rohe Claims**, noch ungeprüft.

**(C) Draft.** Die Claims werden in eine **prüfbare Struktur** gebracht — pro
Claim: Aussage, (vermutete) Quelle, Selbst-Confidence. Das ist absichtlich
dieselbe Form wie die Margin-Probes (`prompt + korrekte + falsche Fortsetzung`),
damit Verifikation *und* spätere Konsolidierung dasselbe Format teilen.

**(D) Verify (externer Prüfer).** Hier kommt das zweite Modell (z. B. GPT) ins
Spiel. Jeder Claim wird **unabhängig** geprüft. Drei mögliche Ergebnisse:
- **AGREE** — Prüfer bestätigt den Claim (idealerweise mehrere Prüfer/Sampling →
  Konsens-Score). → Kandidat fürs Lernen.
- **DISAGREE** — Prüfer widerspricht. → Claim verwerfen *oder* als „zu klären"
  markieren.
- **GAP** — Prüfer nennt einen wichtigen Aspekt von X, den Helix gar nicht
  abgedeckt hat. → gezielter Nachrecherche-Auftrag.

Wichtig (Lehre aus L-018/L-019): **der billigere/strengere Judge ist oft der
bessere.** Der Prüfer ist Telemetrie *und* Gate, nicht die Trainingsquelle selbst
— sonst lernt Helix nur „klinge wie GPT" statt Fakten.

**(E) Gap-Fill.** DISAGREE- und GAP-Claims erzeugen *neue, gezielte*
Research-Aufträge (zurück zu B). Das ist der Teil, der die Schleife „intelligent"
macht: sie sucht aktiv das, was fehlt, statt blind weiterzusammeln.

**(F) Consolidate.** **Nur Claims, die den Konsens-Score überschreiten**, werden
zu Trainingsbeispielen und in **1–N Gradientenschritten auf dem DoRA-Adapter**
gelernt. Das ist das „echte" Lernen — Stufe 2. Gegen Überanpassung an die letzten
Claims wird **Rehearsal** beigemischt (siehe §3).

**(G) Guard.** Nach jedem Konsolidierungs-Update läuft der **frozen-gate**
(`FrozenGateLiveEvaluator`, schon vorhanden): Retention-Probes prüfen, ob altes
Wissen noch sitzt. Regression → **Rollback** des Adapter-Deltas. Kein Update darf
das Modell netto verschlechtern.

---

## 3) Zwei-Stufen-Gedächtnis („Beides kombiniert")

Persistenz-Entscheidung war **Session-Puffer + persistente Konsolidierung**.
Modell dafür ist die menschliche Trennung Arbeits-/Langzeitgedächtnis:

| | **Working Memory (Session)** | **Consolidated Memory (persistent)** |
|---|---|---|
| **Wo** | Kontext / Retrieval-Store (Stufe 1) | DoRA-Adapter-Gewichte (Stufe 2) |
| **Schreiben** | sofort, jeder verifizierte Claim | periodisch (Sleep/Consolidate) |
| **Kosten** | billig, kein Gradient | Gradientenschritte + Guard |
| **Überlebt Neustart?** | nein | ja |
| **Risiko** | keins (read-only Basis) | forgetting → durch Guard abgesichert |

**Ablauf:** Verifizierte Claims landen sofort im **Working Memory** (sofort
nutzbar in derselben Session, kein Trainings-Risiko). Periodisch — ein
„**Consolidation-/Sleep-Pass**" — werden die *stabil bewährten* Inhalte aus dem
Working Memory in den **persistenten Adapter** trainiert (Schritt F+G). So ist
nichts sofort in die Gewichte verdrahtet (sicher), aber Bewährtes wird dauerhaft
(echtes Lernen). Das spiegelt die schon geparkte `<memory>`/`<recall>`-Idee aus
`docs/experimental/knowledge_dna_v2.md`, nur mit Konsolidierungs-Pfad in Gewichte.

---

## 4) Mapping auf das vorhandene `adaptive/`-Modul

Drei der vier Bausteine existieren schon. Wiederverwendung statt Neubau:

| Baustein im Loop | Vorhandenes Modul | Anpassung |
|---|---|---|
| Lern-Auslöser / „lohnt sich Lernen?" | `signals.py` (`is_plateaued`, `detect_regression`, `relative_improvement`) — torch-frei, getestet | als Trigger-Logik wiederverwenden statt Stage-Wechsel |
| Was wurde gelernt? (Telemetrie) | `monitor.py` + `learning_trace.jsonl`, Margin-Probes (`probes.py`, `scoring.py`) | pro Claim eine Margin-Probe → „Neuro-Map" |
| **Wächter / Rollback-Gate** | `frozen_gate.py` (`FrozenGateLiveEvaluator`, `summarize_frozen_results`) | direkt als Post-Update-Guard |
| Entscheidungs-Statemachine | `controller.py` (`CurriculumController`, `Decision`, `DecisionKind`) | von „advance/hold/stop Stage" auf „consolidate/gapfill/rollback/stop Loop" erweitern |
| Update-Mechanik | `trainer.py` | von „großer Lauf" auf „1–N Schritte auf Mini-Batch (Adapter-only)" |

**Designregel beibehalten** (aus `adaptive/__init__.py`): Entscheidungslogik
bleibt **pure-Python / torch-frei** (unit-testbar ohne GPU), alles Torch-Berührende
bleibt isoliert. Das hat sich bewährt (15 Tests in `tests/adaptive/`) und gilt
für die neuen Loop-Komponenten genauso.

---

## 5) Neue Module (was wirklich fehlt)

Nur diese vier Dinge sind echt neu — der Rest ist Verdrahtung:

1. **`src/auralis/lora/`** — aktuell **leer**. Hier muss der physische
   DoRA/LoRA-Layer rein: trainierbare Low-Rank-Matrizen parallel zu den
   eingefrorenen GLA-/Sparse-Attention-/FFN-Projektionen, plus
   merge/unmerge/snapshot/rollback. *Das ist der Ort, an dem gelernt wird.*
   (Vgl. Phase-5-LoRA-Spec und `Doc/REFERENCES/mora_integration.md`.)

2. **`src/auralis/inference/`** — aktuell **leer**. Hier kommt der
   **Test-Time-Learning-Loop** rein: Task → Research → Draft → Verify → (GapFill) →
   Consolidate → Guard → Commit/Rollback.

3. **Verifier-Brücke** — Anbindung des externen Prüfmodells (z. B. GPT via
   OpenRouter, dieselbe Infra wie der Edu-Judge / SFT-Teacher-Pipeline). Liefert
   pro Claim einen Konsens-/Trust-Score. Strikt getrennt vom Trainingssignal.

4. **Memory-Store** — der Working-Memory-Puffer (Retrieval) + der Replay-Buffer
   fürs Rehearsal. Kann auf den `<recall>`-Experimenten aufsetzen.

---

## 6) Catastrophic Forgetting — die drei Schutzschichten

Online-Lernen ohne diese drei Schichten zerstört das Modell. Reihenfolge der
Wichtigkeit:

1. **Frozen Base.** Nur der Adapter lernt. Die 954M-Basis ist read-only. Der
   größtmögliche Schaden ist ein schlechter Adapter, kein zerstörtes Modell.
2. **Rehearsal / Replay.** Jeder Konsolidierungs-Batch mischt N alte, bekannte
   Beispiele (aus dem Replay-Buffer + Retention-Probe-Quellen) unter die neuen
   Claims. Billigstes, wirksamstes Mittel gegen Vergessen.
3. **Frozen-Gate Guard + Rollback.** `FrozenGateLiveEvaluator` misst nach jedem
   Update `frozen_retention_pass`. Fällt es unter den Peak (`detect_regression`),
   wird das Adapter-Delta verworfen. Ein Update wird nur committed, wenn es
   `frozen_promotable` ist. → **Monotonie-Garantie: kein Netto-Rückschritt.**

---

## 7) Datenfluss / Pseudocode (Loop-Skizze)

```python
def learn_about(topic, helix, verifier, memory, adapter, guard):
    queue = [ResearchTask(topic)]
    learned = []
    while queue and not budget_exhausted():
        task = queue.pop()

        # (B/C) Helix recherchiert -> strukturierte Claims
        claims = helix.research_to_claims(task, memory)

        # (D) externer Prüfer, unabhängig, je Claim
        for c in claims:
            verdict = verifier.check(c)          # AGREE / DISAGREE / GAP
            if verdict.agree and verdict.score >= TRUST_THRESHOLD:
                memory.add_working(c)            # Stufe 1: sofort nutzbar
                learned.append(c)
            elif verdict.gap:
                queue.append(ResearchTask(verdict.missing_aspect))  # (E)
            # DISAGREE -> verwerfen oder als 'unsicher' parken

    # (F) Consolidation/Sleep-Pass: Bewährtes in den Adapter trainieren
    batch = build_batch(learned) + memory.sample_rehearsal(N)   # §6.2
    snapshot = adapter.snapshot()
    adapter.online_update(batch, steps=K)        # Stufe 2: echtes Lernen

    # (G) Guard
    if guard.regressed():                        # frozen_retention_pass < Peak
        adapter.restore(snapshot)                # ROLLBACK
        return Result.rolled_back(learned)
    return Result.committed(learned)             # COMMIT (persistent)
```

---

## 8) Risiken & ehrliche Caveats

- **Verifier-Bias / Modell-Kollaps.** Wenn Helix faktisch nur lernt, „GPT
  zuzustimmen", erbt es dessen Fehler und Stil statt Wissen. Gegenmittel:
  mehrere/sampling-basierte Prüfer für echten *Konsens*, Prüfer ≠ Trainingsquelle,
  Stil aus dem Trainingsziel heraushalten.
- **Selbstverstärkung von Fehlern.** Lernt der Loop aus eigenem Output ohne
  starken Guard, zementiert er Halluzinationen. Der frozen-gate-Guard und der
  Trust-Threshold sind genau dagegen. (Vgl. Anti-Halluzinations-Lehre L-017.)
- **Drift über viele Updates.** Viele kleine Adapter-Updates können langsam
  wegdriften, auch wenn jedes einzelne den Guard besteht. Gegenmittel:
  periodische Voll-Retention-Eval (`LORA_EVAL_STANDARD.md`), Adapter-Versionierung,
  „Re-Base" wenn ein Adapter zu groß/alt wird.
- **Verifikation kostet Geld/Latenz.** Jeder Claim = externer Call. Nur über dem
  Self-Confidence-Trigger verifizieren, Ergebnisse cachen.
- **Memory ≠ Lernen.** Stufe 1 fühlt sich nach „Lernen" an, ist aber nur
  Nachschlagen. Echtes Lernen passiert erst im Consolidation-Pass — nicht
  verwechseln (das ist L-002: „LoRA lernte Patterns, keine Fakten").
- **Alles noch unkalibriert.** `TRUST_THRESHOLD`, `K`, Rehearsal-`N`,
  Guard-Schwellen sind Platzhalter. Erst auf einem 500M/1B-Checkpoint messen, was
  „verifiziert" und „gemastert" in nats/pass-rate heißt (wie bei der
  Curriculum-YAML).

---

## 9) Implementierungs-Phasen (Vorschlag)

Klein anfangen, jede Phase liefert etwas Messbares:

- **P0 — DoRA-Layer** (`lora/`): trainierbarer Adapter auf der eingefrorenen
  Basis, mit snapshot/rollback. Test: Adapter lernt 1 Fakt, Basis unverändert.
- **P1 — Offline-Loop**: Research→Draft→Consolidate→Guard *ohne* externen Prüfer,
  auf einem Mini-Topic. Test: frozen-gate verhindert nachweislich einen
  forgetting-Schritt (Rollback greift).
- **P2 — Verifier-Brücke**: externer Prüfer (GPT/OpenRouter) liefert
  Trust-Scores; AGREE/DISAGREE/GAP fließt in den `controller`. Test: GAP erzeugt
  korrekt einen Nachrecherche-Task.
- **P3 — Zwei-Stufen-Gedächtnis**: Working-Memory + periodischer
  Consolidation-Pass + Rehearsal. Test: Wissen überlebt Neustart und Retention
  bleibt grün.
- **P4 — Kalibrierung & Ablation**: Schwellen auf Checkpoint kalibrieren, eine
  saubere Ablation „mit/ohne Selbst-Lern-Layer" auf einem Probe-Set.

Erst wenn P1 zeigt, dass der Guard echtes Forgetting stoppt, lohnt sich P2+.

---

## 10) Offene Fragen (vor dem Code zu klären)

1. **Prüfer-Setup:** ein externes Modell oder Konsens aus mehreren? Welcher
   Trust-Score zählt als „gelernt-würdig"?
2. **Adapter-Granularität:** ein globaler Adapter, oder ein Adapter *pro
   Thema/Skill* (à la „Wissen als ladbare Module")? Letzteres reduziert
   Interferenz, erhöht aber Routing-Komplexität.
3. **Research-Quellen:** nur parametrisch + lokales Retrieval, oder echte
   Web-Suche als Tool? (Bestimmt, wie „neu" das Wissen sein kann.)
4. **Consolidation-Trigger:** nach Zeit, nach N verifizierten Claims, oder erst
   wenn Working-Memory einen Stabilitäts-Plateau erreicht?

---

---

## 11) Sekundäre Option: Lernen direkt im Basismodell (ohne Adapter)

> **Einordnung vorweg:** Das ist eine **dokumentierte Alternative, nicht der
> empfohlene Weg.** Der Adapter-Weg (§1–§10) bleibt die Primärentscheidung,
> **genau weil die eingefrorene Basis nichts vergessen oder überschreiben kann** —
> ein schlechtes Update beschädigt höchstens den Adapter, nie das Weltwissen, und
> Rollback ist trivial (Delta wegwerfen). Diese Option wird hier festgehalten,
> falls man später *direkt in die echten Gewichte* lernen will (näher an „wie ein
> Mensch dazulernt"), mit vollem Bewusstsein für die Kosten.

### Die Idee

„Ohne LoRA, direkt im laufenden Modell" heißt technisch: die **echten
Basis-Gewichte** werden online verändert, statt ein separates Adapter-Modul.
Möglich ist das — aber genau das ist die in §0 als zerstörerisch markierte
**Stufe 3**. Dass es überhaupt funktionieren kann, liegt am selben Grund, warum
ein Mensch dabei nicht alles vergisst: das Gehirn trennt **schnelles**
(Hippocampus) von **langsamem** Lernen (Neokortex, Konsolidierung im Schlaf) und
schützt wichtige Synapsen (*synaptische Konsolidierung*). Naives Voll-Backprop
aufs ganze Netz hat diesen Schutz **nicht** → catastrophic forgetting. Eine
adapterfreie Variante muss diese Schutzmechanismen also künstlich nachbauen.

### Die Methoden (geordnet von chirurgisch zu roh)

| Methode | Was sie tut | Stärke | Risiko / Grenze |
|---|---|---|---|
| **Knowledge Editing** (ROME / MEMIT / MEND) | schreibt einen Fakt per geschlossenem Rang-1-Update direkt in eine **FFN-Matrix** | lokal, fast kein Forgetting, sofort; trifft Helix' **SwiGLU-FFN** | nur für *Fakten*, nicht für Fähigkeiten/Stil; viele Edits driften |
| **Online-Finetune + EWC** | updatet echte Gewichte, schützt wichtige per Fisher-Strafterm (= „steife Synapse") | lernt auch *Fähigkeiten*; direkter Hirn-Nachbau | Fisher-Matrix kostet Speicher/Rechnung; Schwellen heikel |
| **Rehearsal / Replay** | mischt alte bekannte Beispiele in jedes Update | billigste, wirksamste Forgetting-Bremse | braucht gepflegten Replay-Buffer |
| **Sparse / selektive Updates** | ändert nur wenige aktive FFN-Zeilen | LoRA-Effekt *ohne* Zusatzmatrix, in-place | Auswahl der Zeilen nicht trivial |
| **Fast Weights / Mamba-State** | nutzt Helix' rekurrenten **Mamba-2-State** als flüchtiges Kurzzeitgedächtnis | sofort, kein Gewichts-Update; „Hippocampus" | flüchtig — überlebt Session nicht ohne Konsolidierung |

### „Ohne LoRA" ≠ „ohne Struktur"

Der Kernpunkt: **reines Voll-Backprop live aufs ganze Modell zerstört es** — auch
das Gehirn macht das nicht so. Jede Methode oben führt eine *Schutzstruktur*
wieder ein (Lokalität, Importance-Regularisierung, Replay, Sparsity, schnell/
langsam-Trennung). LoRA ist nur **eine** dieser Strukturen — die bequemste, weil
Rollback trivial ist. Lässt man LoRA weg, braucht man zwingend eine der anderen.

### Wie es in dieses Design einklinken würde

Der Loop (§2) und das zweistufige Gedächtnis (§3) bleiben **identisch**. Es ändert
sich nur Schritt **(F) Consolidate**:

- statt `adapter.online_update(...)` → **MEMIT-Edit** der SwiGLU-FFN für
  verifizierte *Fakten* und/oder **Mini-Finetune mit EWC + Replay** für
  *Fähigkeiten*;
- statt `adapter.snapshot()/restore()` → **Gewichts-Snapshot/Restore** (teurer,
  aber funktional) als Rollback-Netz;
- der **frozen-gate-Guard (§6.3) bleibt unverändert** der harte Wächter.

Mamba-State würde als Working-Memory der Stufe 1 dienen (sofortiges „Erinnern"),
die FFN-Edits/EWC als Langzeit-Konsolidierung (Stufe 2).

### Direkter Vergleich (Entscheidungsgrundlage)

| | **Adapter (§1–§10, empfohlen)** | **Direkt im Basismodell (diese Option)** |
|---|---|---|
| Wo landet Wissen | Zusatz-Adapter | echte Gewichte (FFN etc.) |
| „Wie ein Mensch"? | eher Notizbuch obendrauf | näher dran (Kortex selbst) |
| **Vergessen/Löschen** | **praktisch ausgeschlossen** (Basis read-only) | **möglich** → braucht EWC/Replay/Edit-Lokalität |
| Rollback | trivial (Delta weg) | teuer (Voll-Snapshot) |
| Beste Methode | Online-Finetune des Adapters | MEMIT (Fakten) + EWC/Replay (Skills) |

**Fazit / Empfehlung:** Primär den Adapter-Weg umsetzen — er erfüllt die
ausdrückliche Anforderung „**es darf nichts vergessen oder löschen**" am
saubersten. Die adapterfreie Variante bleibt als spätere Option offen, am ehesten
als *MEMIT-Fakten-Edit* in die FFN (lokal, forgetting-arm), falls man Wissen
wirklich *in* die Basis und nicht *neben* sie schreiben will.

---

*Dieses Dokument ist Entwurf. Empfohlener nächster Schritt laut Plan: P0 (DoRA-Layer
in `lora/`) als kleinster lauffähiger Baustein, sobald das Design bestätigt ist.
Die adapterfreie Variante (§11) ist als spätere Option dokumentiert, nicht der
Primärpfad.*
