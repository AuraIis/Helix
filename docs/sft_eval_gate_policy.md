# Auralis SFT Eval Gate Policy

## Disjunkte Evals sind Pflicht

Ein besserer SFT-Score ist nur vertrauenswuerdig, wenn die Eval-Daten disjunkt zu den
Trainingsdaten sind. Disjunkt heisst hier: keine gleichen oder fast gleichen Prompts,
Antworten, Generator-Templates, Seeds, Quellen-Snippets oder Antwortmuster.

Grund: Ein Modell kann sonst oberflaechliche Muster lernen und im Score gut aussehen,
ohne die Faehigkeit wirklich zu koennen. Beim Code-SFT kann das zum Beispiel bedeuten,
dass `def`, `return`, Klammern und Keywords belohnt werden, obwohl der Code nicht korrekt
funktioniert.

## Harte Regel

- Vor jedem groesseren SFT zuerst eine disjunkte Eval bauen.
- Trainingsdaten und Evaldaten strikt trennen.
- Keine Eval aus derselben Datei, demselben Generator-Template oder denselben Seed-Fragen wie SFT.
- Prompt- und Antwort-Aehnlichkeit per Hash und fuzzy matching pruefen.
- Code-Evals muessen ausfuehrbar geprueft werden: Syntaxcheck, Unit Tests, erwartete Ausgabe.
- Fakten-Evals muessen aus anderen QA-Seeds/Quellen stammen als die Trainingsdaten.
- Halluzinations-Evals muessen Fallen enthalten, die nicht exakt im Training vorkamen.
- Automatischer Score allein reicht nicht: manuelle Stichprobe bleibt Pflicht.

## Entscheidungsgate

Ein SFT-Run darf erst als besser gelten, wenn alle drei Punkte stimmen:

1. Disjunkte Eval verbessert sich.
2. Manuelle Stichprobe wirkt besser, nicht nur formaler.
3. Keine Regression bei Refusal, Deutsch, Fakten und Code-Basics.
