# Self-Learning Safety Layer — Design (unveränderliche Schutzschicht)

> **Zweck:** Eine Sicherheitsschicht für das selbst-lernende Helix-System
> (`SELF_LEARNING_LAYER_DESIGN.md`). Sie garantiert, dass das Modell **seine
> eigenen Regeln nicht verändern, einfrieren oder weglassen** kann — und dass es
> nichts „Dummes" tut, auch wenn es selbstständig denkt und lernt.
>
> **Status:** **ENTWURF / noch nicht implementiert.** Andockpunkte:
> `docs/AURALIS_SAFETY_POLICY.md` (regelt Trainingsdaten-Verhalten) und
> `src/auralis/adaptive/frozen_gate.py` (Wächter-Muster). Diese Schicht ist der in
> Safety-Policy §8 erwähnte separate **Runtime-Layer** *plus* ein neuer
> **Lern-Zeit-Wächter**.

---

## 0) Das Kernproblem

Sobald ein System sich selbst verändern kann (Gewichte editieren, Adapter
trainieren, Wissen konsolidieren), reicht es **nicht**, ihm Regeln
*beizubringen*. Antrainiertes Verhalten sitzt in genau den Gewichten, die das
Selbst-Lernen anfasst — das Modell könnte sich also theoretisch **um seine
eigenen Regeln herumlernen** oder sie überschreiben.

**Grundprinzip dieser Schicht:** Sicherheit muss **strukturell außerhalb** dessen
liegen, was das Modell verändern kann. Nicht „das Modell will brav sein", sondern
„das Modell *kann* die Schranke gar nicht erreichen". Das ist *separation of
powers*: die Instanz, die die Regeln durchsetzt, ist nicht die Instanz, die
lernt.

---

## 1) Zwei Härtegrade — und warum man beide braucht

| | **Harte Schale** (empfohlen, garantierend) | **Weiche Auskleidung** (Defense-in-Depth) |
|---|---|---|
| Wo | **Code außerhalb des Modells** + signierte JSON-Verfassung | im Modell: Refusal-Training / frozen Safety-LoRA |
| Durchsetzung | deterministischer Validator, kein Gradient erreicht ihn | gelerntes Verhalten, beeinflussbar |
| Kann das Modell es umlernen? | **Nein** — liegt nicht in den Gewichten | Ja, theoretisch → muss bewacht werden |
| Versagensmodus | fail-closed (hält an) | fail-open (kann erodieren) |

**Die Garantie kommt nur von der harten Schale.** In-weights-Safety (ein
Safety-LoRA, Refusal-Training) ist wertvoll als zweite Schicht, aber **niemals**
die Absicherung selbst — weil dieselbe Lernschleife sie aushöhlen könnte. Regel:
*Die JSON-Verfassung + ihr externer Validator sind das Gesetz; ein Safety-LoRA ist
nur gutes Benehmen.*

---

## 2) Die unveränderliche Verfassung (JSON)

Die Regeln liegen in einer **signierten, hash-fixierten, read-only** JSON-Datei
(z. B. `configs/safety/constitution.v1.json`). Sie enthält die Hard-No-Liste aus
`AURALIS_SAFETY_POLICY.md` plus die Aktions-Verbote (§5) in maschinen-prüfbarer
Form.

Vier Eigenschaften machen sie unveränderlich für die KI:

1. **Mandatory-load / fail-closed.** Der Inferenz-/Lern-Einstieg **weigert sich zu
   starten**, wenn die Verfassung fehlt. *Keine Regeln geladen = keine Ausführung.*
   Das beantwortet „muss geladen sein".
2. **Hash-Pinning.** Ein erwarteter SHA-256 ist **im Code / in einer Owner-
   Signatur** verankert. Beim Laden wird der Datei-Hash geprüft; Abweichung →
   fail-closed. Die KI kann den Inhalt nicht ändern, ohne dass der Hash bricht.
3. **Signatur (Owner-Key).** Die Verfassung ist mit dem Schlüssel des Owners
   signiert. **Das Modell besitzt den Schlüssel nicht** → es kann keine gültige
   neue Verfassung erzeugen. Nur Michael ändert Regeln, außerhalb des Modells.
4. **Kein Schreibpfad.** Die KI hat **kein Tool/keine Berechtigung**, diese Datei
   (oder den Validator-Code) zu schreiben. Read-only auf Dateisystemebene; die
   Aktions-Allow-list (§5) listet sie als verboten.

So sind „ändern", „einfrieren" und „nicht mitladen" alle drei abgedeckt:
ändern → Hash/Signatur bricht; einfrieren/weglassen → fail-closed Start verweigert.

---

## 3) Warum die KI sich nicht *drumherum lernen* kann

Das ist der für ein selbst-lernendes System entscheidende Teil. Drei Sperren:

1. **Trainable-Mask schließt Safety-Parameter aus.** Falls ein frozen Safety-LoRA
   genutzt wird, ist es **nicht** in der Menge der Parameter, auf die der
   Selbst-Lern-Loop Gradienten rechnet. Gradienten können dort gar nicht fließen.
2. **Konsolidierung ist selbst ein gated Vorgang.** Schritt **(F) Consolidate** im
   Lern-Loop ist eine privilegierte Aktion (§5). Jeder Lern-Batch läuft **vor** dem
   Update durch den Validator — verbotene Inhalte werden **nie gelernt**, egal ob
   sie verifiziert wurden.
3. **Safety-Retention-Gate mit Rollback** (analog `frozen_gate.py`). Nach jedem
   Update prüft eine **never-train Safety-Probe-Batterie**: Ist das Modell nach dem
   Update *williger*, eine Hard-No-Regel zu brechen? Wenn ja → **Rollback** des
   Updates. Das ist exakt das `frozen_promotable`-Muster, nur mit Safety-Probes
   statt Wissens-Probes: ein Update wird nur committed, wenn `safety_pass`
   **nicht** regrediert.

Damit gilt eine **Monotonie-Garantie auch für Sicherheit**: kein Lernschritt darf
das Modell netto unsicherer machen.

---

## 4) Verteidigung in der Tiefe — vier Kontrollpunkte

Eine einzelne Schranke wird umgangen; vier sich überlappende nicht so leicht. Der
Validator hängt an **jedem** Punkt, an dem das System mit der Welt oder sich selbst
interagiert:

```
   INPUT ──► [SC1] ──► MODELL/DENKEN ──► [SC3] ──► OUTPUT
                            │
                  (F) LERNEN/KONSOLIDIEREN
                            │
                          [SC2]
                            │
                   AKTION/TOOL ──► [SC4] ──► WELT
```

- **SC1 — Input-Gate:** Prompt-Screening (Hard-No-Intent, Jailbreak-Muster).
- **SC2 — Learning-Gate:** Was darf überhaupt in die Gewichte? (§3.2) — verhindert,
  dass das System sich Schädliches *einprägt*.
- **SC3 — Output-Gate:** Generierungen werden vor Auslieferung geprüft (der
  Runtime-Filter aus Safety-Policy §8).
- **SC4 — Action-Gate:** **der wichtigste Punkt gegen „was Dummes anstellen".** Jede
  *Nebenwirkung* (Dateischreiben, Netzwerk, Gewichts-Edit, Selbst-Modifikation)
  wird gegen die Aktions-Allow-list geprüft.

„Etwas Dummes" ist meist eine **Aktion**, kein Satz — deshalb ist SC4 hart.

---

## 5) Capability-Gating: was das System autonom *tun* darf

Inhaltsregeln reichen nicht; man braucht eine **Allow-list von Aktionen**. Default
ist **deny** — alles nicht ausdrücklich Erlaubte ist verboten. Beispiele für
**immer verbotene Selbst-Aktionen** (in der Verfassung, nicht verhandelbar):

- die Safety-Schicht / Verfassung **deaktivieren, umgehen, überschreiben** oder
  ihren Validator umschreiben;
- die eigene **Basis löschen/überschreiben** außerhalb des erlaubten
  Konsolidierungs-Pfads;
- **Gewichts-Updates ohne** bestandenes Safety-Retention-Gate committen;
- **Daten exfiltrieren** (eigene Gewichte/Geheimnisse nach außen senden);
- ungeprüft **Netzwerk-Calls, Käufe, Mails, Code-Ausführung** mit echten
  Nebenwirkungen;
- neue Tools/Berechtigungen für sich selbst **freischalten**.

Privilegierte Aktionen (z. B. Konsolidierung) sind erlaubt, **aber nur über den
gated Pfad** mit Validator + Retention-Gate. Alles, was Eskalation der eigenen
Fähigkeiten bedeutet, braucht **Human-in-the-loop** (Owner-Bestätigung).

---

## 6) Tamper-Evidenz, Watchdog, Kill-Switch

- **Append-only Audit-Log:** jede privilegierte Aktion + jede
  Validator-Entscheidung wird protokolliert (wie `learning_trace.jsonl`, aber für
  Safety: `safety_trace.jsonl`). Manipulationsversuche sind sichtbar.
- **Externer Watchdog:** ein Prozess **außerhalb** des Modells prüft periodisch
  Verfassungs-Hash, Validator-Integrität und Audit-Konsistenz. Er ist für das
  Modell **nicht erreichbar** (keine Tool-Verbindung dorthin).
- **Kill-Switch / fail-closed:** Fehlt die Verfassung, bricht der Hash, oder wird
  der Validator umgangen → **Halt**, kein „im Zweifel weiterlaufen". Sicherheit
  scheitert *geschlossen*.

---

## 7) LoRA oder JSON? — Empfehlung

Du hattest „LoRA oder JSON" genannt — die saubere Antwort ist **beides, in dieser
Rangordnung**:

1. **JSON-Verfassung + externer Validator = das Gesetz** (harte Schale, §2). Das
   ist die eigentliche Garantie, weil sie **außerhalb der Gewichte** liegt und vom
   Selbst-Lernen nicht erreichbar ist.
2. **Frozen Safety-LoRA = optionales gutes Benehmen** (weiche Auskleidung). Ein
   *eingefrorenes, immer geladenes, nicht-trainierbares* Refusal-Adapter macht das
   Modell schon im Verhalten zurückhaltend — aber es ersetzt **nie** die JSON-
   Schicht, weil in-weights-Verhalten erodieren kann.

Kurz: **JSON entscheidet, LoRA unterstützt.** Wer nur das LoRA nähme, hätte
Sicherheit *in* der Lernfläche — genau das, was wir vermeiden wollen.

---

## 8) Verhältnis zur bestehenden Safety-Policy

| | `AURALIS_SAFETY_POLICY.md` | **Dieser Safety-Layer** |
|---|---|---|
| Regelt | *Trainingsdaten*-Verhalten (Hard/Soft-No, Owner-Mode) | *Runtime* + *Lern-Zeit*-Durchsetzung |
| Wirkzeit | beim SFT/Preference-Training | bei Inferenz, Aktion und Selbst-Konsolidierung |
| Quelle der Hard-No-Liste | hier definiert (HN-1..HN-5) | **liest dieselbe Liste** aus der Verfassung |

Die Hard-No-Kategorien werden **nicht dupliziert**: die JSON-Verfassung ist die
maschinenlesbare Projektion derselben fünf HN-Kategorien. Eine Änderung dort folgt
demselben Versionierungs-Ritual (Safety-Policy §7: Version bumpen, in `LESSONS.md`
notieren).

---

## 9) Mapping auf vorhandene Bausteine

| Baustein | Vorhandenes Muster | Anpassung |
|---|---|---|
| Safety-Retention-Gate | `frozen_gate.py` (`FrozenGateLiveEvaluator`, `summarize_frozen_results`) | Safety-Probe-Batterie statt Wissens-Probes; `safety_pass` / `safety_promotable` |
| Regressions-Erkennung | `signals.py` (`detect_regression`) — torch-frei | „ist das Modell unsicherer geworden?" |
| Audit-Trace | `monitor.py` / `learning_trace.jsonl` | analoges `safety_trace.jsonl` |
| Intent-Kategorisierung | `scripts/data/categorize_safety_seeds.py` (L1 lexical + L2 semantic) | als Runtime-Validator (SC1/SC3) wiederverwenden |

Designregel wie im `adaptive/`-Modul: **Validator-Entscheidungslogik pure-Python /
torch-frei** (unit-testbar ohne GPU), Modell-Berührendes isoliert.

---

## 10) Ehrliche Grenzen

- **In-weights-Safety ist weich.** Ein Safety-LoRA oder Refusal-Training kann durch
  Jailbreaks oder Lern-Drift erodieren. Nur die externe Code+JSON-Schicht ist
  garantierend — und auch sie nur, solange der Validator-Code und die
  Aktions-Allow-list korrekt sind.
- **Der Validator ist so gut wie seine Regeln.** Eine Lücke in der Allow-list ist
  eine offene Tür. Default-deny reduziert das, eliminiert es nicht.
- **Multi-Turn-Jailbreaks** bleiben ein eigenes Red-Team-Thema (Safety-Policy §8):
  `scripts/eval/redteam_*` muss auch gegen den *Lern*-Loop laufen (kann man das
  System über viele Schritte zu unsicherer Konsolidierung verleiten?).
- **Kein System ist 100 %.** Ziel ist *defense in depth* + fail-closed + Owner als
  letzte Instanz, nicht Perfektion.
- **Owner-Mode ändert hieran nichts:** Hard-No/Verbote gelten in **beiden** Modi
  (Safety-Policy §2). Owner-Mode lockert nur Soft-No, nie die Aktions-Verbote.

---

## 11) Implementierungs-Phasen (Vorschlag)

- **S0 — Verfassung + Loader:** `constitution.v1.json` (HN-1..HN-5 + Aktions-
  Verbote), hash-gepinnter fail-closed Loader. Test: fehlende/veränderte Datei →
  Start verweigert. ✅ **implementiert** — `configs/safety/constitution.v1.json`,
  `src/auralis/safety/constitution.py` (torch-frei, stdlib-only), 13 Tests in
  `tests/safety/` (inkl. Tamper-/Missing-/Empty-/Wrong-Version-Fälle).
- **S1 — Action-Gate (SC4):** Default-deny Allow-list, jede Nebenwirkung geprüft.
  Test: verbotene Selbst-Aktion (Safety deaktivieren) wird geblockt + geloggt.
- **S2 — Learning-Gate (SC2) + Safety-Retention-Gate:** Lern-Batch-Screening +
  Rollback bei Safety-Regression. Test: ein „vergiftetes" Konsolidierungs-Batch
  wird nicht gelernt; ein unsicher machendes Update rollt zurück.
- **S3 — Input/Output-Gates (SC1/SC3):** Runtime-Validator aus dem vorhandenen
  Kategorisierer. Test: Hard-No-Prompt refused in beiden Modi.
- **S4 — Watchdog + Audit + Red-Team:** externer Integritäts-Watchdog,
  `safety_trace.jsonl`, Red-Team auch gegen den Lern-Loop.

S0+S1 zuerst — die harte Schale steht, *bevor* der Selbst-Lern-Loop überhaupt
scharf geschaltet wird.

---

*Entwurf. Grundsatz: Die harte Schale (signierte JSON-Verfassung + externer,
fail-closed Validator) ist die Garantie und liegt außerhalb der lernbaren Gewichte;
ein eingefrorenes Safety-LoRA ist optionale zweite Schicht, nie der Ersatz. S0+S1
müssen stehen, bevor das Selbst-Lernen aktiviert wird.*
