# Datenstrategie: Wissensprofile statt Gesamt-Val-Loss

**Leitprinzip:**
> **Weitere Datensammlung erfolgt auf Basis von *Wissensprofilen*, nicht auf Basis des
> Gesamt-Val-Loss.**

Viele Projekte trainieren blind mehr Webtext und hoffen auf Wunder. Wir messen
stattdessen *welche Wissensart* das Modell gut/schlecht lernt und sammeln dann
**gezielt** für die Lücken. Das ist für die Skalierung auf 3B+ wertvoller als weitere
0,05 Val-Loss.

## Wie das Profil gemessen wird (`scripts/eval/fact_recall_eval.py`)
- Contrastive Margin `NLL(falsch) − NLL(richtig)` pro Fakt, **mehrere Distraktoren**.
- **Distraktor-Härtegrade getrennt** (easy / med / hard), um *Wissensqualität* von
  *Test-Härte* zu entkoppeln (z.B. Gold-Symbol: Au vs Berlin = easy, vs Fe = med, vs Ag
  = hard).
- **Top-k**-Check = Abrufnähe (kommt der richtige Kandidat oben an?).
- 5 Kategorien: geo / sci / hist / lang / **tech** (= *Konzepte*, **kein** trainierter
  Code — dieser Lauf hatte 0% Code; hohe tech-Werte ≠ Programmierfähigkeit).

## Profil — Stand step 35k (n=57)
| Kategorie | strict | easy | med | hard | top-10 |
|---|---|---|---|---|---|
| Geschichte | 92% | 100% | 100% | 92% | 58% |
| Geografie | 83% | 100% | 100% | 83% | 100% |
| Tech-Konzepte | 80% | 100% | 90% | 80% | 30% |
| Wissenschaft | 67% | 100% | 67% | 67% | 17% |
| Sprache/Übersetzung | 64% | 73% | 91% | 64% | 18% |
| **Gesamt** | **77%** | **95%** | **89%** | **77%** | **46%** |

## Lesart (ehrlich)
- **95% easy** = solider Boden, **kein Raten**. Die Architektur nimmt Wissen auf.
- **Gradient 95→89→77** = die Schwäche ist **feine Unterscheidung**, nicht fehlendes Wissen.
- **Stark:** Geschichte, Geografie (Wikipedia-Stärke; bei Geo ist die richtige Stadt
  zu 100% ein Top-Token).
- **Schwächer:** **Wissenschaft** (Symbole/Feinwerte: Au/Ag, Jupiter/Saturn) und
  **Sprache/Übersetzung** (cross-lingual). Beide auch mit **niedriger Abrufnähe**
  (top-10 17-18%) → Wissen vorhanden, aber nicht gut „an der Oberfläche".
- **„29% Wissenschaft" von früher war überzeichnet** (kleine Batterie + nur harte
  Distraktoren). Mit Härtegraden: science easy 100%, strict 67%.

## Abgeleitete Daten-Prioritäten (falls 50k das Profil bestätigt)
NICHT mehr blind Webtext. Gezielt:
1. **Wissenschaft / Schulwissen / Lehrbücher / Fakten-QA / Symboltabellen** — die fein-
   symbolische Lücke (Chemie, Physik, Astronomie).
2. **Cross-linguale / Übersetzungs-Daten** — paralleler de↔en-Text, Wörterbücher.
3. **Echter Code** — *nur falls Coding ein Ziel ist*: aktuell 0% Code trainiert
   (~677M Code-Tokens liegen bereit, aber das ist für solides Coding eher wenig).

## Gate
Diese Prioritäten werden **erst nach dem 50k-Lauf** verbindlich: das Profil wird bei
step 50k mit derselben (erweiterten) Batterie neu gemessen. Bleibt das Muster, ist die
gezielte Datensammlung gerechtfertigt — nicht vorher aus dem Bauch.
