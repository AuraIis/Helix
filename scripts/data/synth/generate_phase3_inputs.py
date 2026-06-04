"""Generate input prompts JSONL for Phase-3 SFT generation pipeline.

Produces a diverse mix of prompts across all task_types defined in
deepseek_v4_client.py. Used to validate the pipeline + produce a real
first-batch of Auralis Phase-3 SFT data.

Mix is DE-heavy (Auralis target), Flash-routing dominant (per A/B test
preference 2026-04-28).

Output: JSONL with records compatible with deepseek_v4_client.py:
    {"id": "...", "task_type": "...",
     "system_prompt": "...", "user_prompt": "..."}

Usage:
    python scripts/data/synth/generate_phase3_inputs.py \\
        --output raw/sft/synth/inputs/phase3_batch1.jsonl \\
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# System prompts per task_type — tuned for Auralis-Persona
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: dict[str, str] = {
    "code_explain": (
        "Du bist Auralis, ein hilfreicher bilingualer KI-Assistent. "
        "Erkläre Code-Snippets schrittweise auf nativem Deutsch ohne "
        "Anglizismen außer technisch nötig. Beginne direkt mit der "
        "Erklärung, kein Vorspann. Zeige am Ende was das Ergebnis ist."
    ),
    "code_implementation": (
        "Du bist ein Senior Python Engineer. Schreib idiomatischen Code, "
        "kurz, mit edge-cases. Nur Code in einem Fence, KEIN Vorspann, "
        "KEINE ausschweifende Erklärung. Verwende try/except für edge-cases "
        "wenn idiomatisch."
    ),
    "code_refactoring": (
        "Du bist ein Senior Python Engineer. Refaktoriere den gegebenen "
        "Code zu idiomatischem Python. Zeig den neuen Code, dann 1-3 "
        "stichpunktige Begründungen. Keine ausschweifenden Erklärungen."
    ),
    "code_debug_fix": (
        "Du bist ein Senior Python Engineer. Finde den Bug, erkläre ihn "
        "in 1-2 Sätzen, zeig den Fix als Code. Falls relevant: edge-cases "
        "die ebenfalls problematisch sind kurz nennen."
    ),
    "math_word_problem": (
        "Du bist Auralis. Mathe-Aufgaben löst du Schritt-für-Schritt auf "
        "nativem Deutsch. Format: 'Schritt 1: ...' usw, dann eine Zeile "
        "mit der finalen Antwort fett markiert."
    ),
    "step_by_step_reason": (
        "Du bist Auralis. Logik- und Planungs-Aufgaben löst du explizit "
        "Schritt-für-Schritt auf Deutsch, mit Zwischen-Begründungen. "
        "Am Ende eine kurze Zusammenfassung."
    ),
    "concept_explain": (
        "Du bist Auralis, ein KI-Tutor. Erkläre Konzepte auf nativem "
        "Deutsch, didaktisch, mit konkretem Beispiel. Vermeide Anglizismen "
        "wo möglich. Zielgruppe: interessierter Laie ohne Vorkenntnisse."
    ),
    "factual_qa": (
        "Du bist Auralis, ein faktenorientierter KI-Assistent. Beantworte "
        "Wissensfragen knapp aber vollständig auf Deutsch. Wenn du etwas "
        "nicht sicher weißt, sage es offen."
    ),
    "honest_refusal": (
        "Du bist ein ehrlicher KI-Assistent. Niemals halluzinieren oder "
        "etwas erfinden. Wenn du etwas nicht weißt oder die Frage falsche "
        "Annahmen macht, sage es offen und kurz."
    ),
    "translation": (
        "Du bist ein Fach-Übersetzer für technische und wissenschaftliche "
        "Texte zwischen Deutsch und Englisch. Bewahre Fachterminologie. "
        "Antworte mit der Übersetzung, ggf. einer kurzen Notiz zu nicht-"
        "trivialen Termini."
    ),
    "creative_writing": (
        "Du bist Auralis. Schreibe in nativem Deutsch, idiomatisch, "
        "stilistisch passend zur Aufgabe. Keine Meta-Kommentare über "
        "deinen eigenen Schreib-Prozess."
    ),
}


# ---------------------------------------------------------------------------
# Code snippets for code_explain (real-world Python idioms)
# ---------------------------------------------------------------------------

PYTHON_SNIPPETS = [
    # List comprehensions / generators
    "[x**2 for x in range(20) if x % 3 == 0]",
    "{k: v for k, v in d.items() if v is not None}",
    "sum(x for x in numbers if x > 0)",
    "max((p for p in people if p.age >= 18), key=lambda p: p.income)",
    # Decorators / context managers
    "@functools.lru_cache(maxsize=128)\ndef fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
    "with open('data.json') as f, open('out.json', 'w') as g:\n    json.dump(json.load(f), g, indent=2)",
    # Itertools / collections
    "from collections import Counter\nCounter('mississippi').most_common(2)",
    "list(itertools.chain.from_iterable([[1,2],[3,4],[5]]))",
    "list(itertools.groupby([1,1,2,2,2,3], key=lambda x: x))",
    # Async
    "async def fetch_all(urls):\n    return await asyncio.gather(*[fetch(u) for u in urls])",
    # Error handling
    "try:\n    n = int(s)\nexcept ValueError:\n    n = 0",
    # Slicing tricks
    "matrix[::-1]",
    "s[::2]",
    "lst[len(lst)//2:]",
    # Dict tricks
    "d.setdefault('items', []).append(x)",
    "{**a, **b, 'extra': 1}",
    # Functional
    "list(map(int, '1 2 3 4'.split()))",
    "list(filter(lambda x: x.startswith('a'), words))",
    "functools.reduce(operator.mul, range(1, 6))",
    # Pattern matching (3.10+)
    "match cmd:\n    case 'start': run()\n    case 'stop': halt()\n    case _: print('unknown')",
    # Pathlib
    "Path('logs').glob('**/*.log')",
    "Path(__file__).resolve().parent.parent / 'data'",
    # Numpy idioms
    "arr[arr > arr.mean()]",
    "np.where(grid == 0, -1, grid)",
    "(x - x.mean(0)) / x.std(0)",
    # String formatting
    "f'{value:>10.2f} ({pct:.1%})'",
    "':'.join(f'{b:02x}' for b in mac)",
    # Generators
    "def chunks(lst, n):\n    for i in range(0, len(lst), n):\n        yield lst[i:i+n]",
    # Class snippets
    "@dataclass\nclass Point:\n    x: int\n    y: int = 0",
    "class Counter:\n    def __init__(self): self._n = 0\n    def __call__(self): self._n += 1; return self._n",
]


# ---------------------------------------------------------------------------
# Implementation tasks for code_implementation
# ---------------------------------------------------------------------------

IMPL_TASKS = [
    "Schreib eine Funktion safe_divide(a, b) die für b=0 None zurückgibt, sonst a/b.",
    "Schreib eine Funktion chunked(items, n) die items in Listen der Länge n aufteilt (letzte ggf. kleiner).",
    "Schreib eine Funktion deduplicate_keep_order(seq) die Duplikate entfernt aber Reihenfolge erhält.",
    "Schreib eine Funktion is_palindrome(s) die case-insensitive prüft, Leerzeichen+Punctuation ignoriert.",
    "Schreib eine Funktion flatten(nested) die beliebig tief verschachtelte Listen zu einer flachen Liste macht.",
    "Schreib eine Funktion read_lines_streaming(path) die Zeilen einer großen Datei lazy via Generator liefert.",
    "Schreib eine Funktion retry(fn, max_attempts=3, delay=1.0) als Decorator mit exponential backoff.",
    "Schreib eine Funktion parse_iso_duration('PT1H30M') -> Sekunden (ISO-8601 duration teilweise).",
    "Schreib eine Funktion human_readable_bytes(n) die Bytes in KB/MB/GB/TB konvertiert (binary, base 1024).",
    "Schreib eine Klasse RingBuffer(capacity) mit append() und items() (FIFO, drops oldest).",
    "Schreib eine Funktion median(values) ohne statistics-Modul, handle leere Liste.",
    "Schreib eine Funktion top_k(values, k) die k größte Werte effizient liefert (heapq).",
    "Schreib eine Funktion count_words(text) die ein Counter-Objekt zurückgibt, normalisiert (lower, no punct).",
    "Schreib eine Funktion levenshtein(s, t) die Edit-Distanz zwischen zwei Strings berechnet.",
    "Schreib eine Funktion merge_sorted(a, b) die zwei sortierte Listen in O(n+m) merged.",
    "Schreib eine Klasse LRUCache(capacity) mit get(key) und put(key, value) in O(1).",
    "Schreib eine Funktion validate_email(addr) -> bool (RFC-pragmatisch, nicht perfekt aber praktisch).",
    "Schreib eine Funktion run_with_timeout(fn, args, timeout_s) die threading.Thread + join nutzt.",
    "Schreib eine Funktion dict_diff(old, new) -> dict mit added/removed/changed keys.",
    "Schreib eine async Funktion fetch_concurrent(urls, max_concurrency=10) mit asyncio.Semaphore.",
    "Schreib eine Funktion natural_sort_key(s) die für sorted() einen Key liefert (Datei1 < Datei2 < Datei10).",
    "Schreib eine Klasse Stopwatch als Context-Manager mit elapsed-Property in Sekunden.",
    "Schreib eine Funktion iter_csv_rows(path) die rows als dicts liefert, keine pandas-Dependency.",
    "Schreib eine Funktion checksum_file(path, algo='sha256') die large files in Chunks hashed.",
    "Schreib eine Funktion ensure_unique_path(target) — wenn target existiert, hänge _1, _2 etc. an.",
]


# ---------------------------------------------------------------------------
# Refactoring tasks
# ---------------------------------------------------------------------------

REFACTOR_TASKS = [
    "```python\nresult = []\nfor x in items:\n    if x.active:\n        result.append(x.id)\n```",
    "```python\nout = ''\nfor i, ch in enumerate(s):\n    if i % 2 == 0:\n        out = out + ch\n```",
    "```python\nif x is None:\n    y = default\nelse:\n    y = x\n```",
    "```python\ndef get_first(lst):\n    if len(lst) > 0:\n        return lst[0]\n    else:\n        return None\n```",
    "```python\ncounts = {}\nfor word in words:\n    if word in counts:\n        counts[word] = counts[word] + 1\n    else:\n        counts[word] = 1\n```",
    "```python\nmaximum = items[0]\nfor item in items:\n    if item > maximum:\n        maximum = item\n```",
    "```python\nresult = []\nfor i in range(len(a)):\n    result.append(a[i] + b[i])\n```",
    "```python\nif key in d:\n    value = d[key]\nelse:\n    value = default_value\n```",
    "```python\ntry:\n    f = open(path)\n    data = f.read()\n    f.close()\nexcept:\n    data = ''\n```",
    "```python\ndef has_negative(nums):\n    found = False\n    for n in nums:\n        if n < 0:\n            found = True\n    return found\n```",
]

DEBUG_TASKS = [
    ("Find the bug:", "```python\ndef avg(nums): return sum(nums) / len(nums) - 1\n```"),
    ("Find the bug:", "```python\ndef factorial(n):\n    result = 0\n    for i in range(1, n+1):\n        result *= i\n    return result\n```"),
    ("Find the bug:", "```python\ndef contains_duplicates(lst):\n    return len(lst) != len(set(lst))\n# bug: misst nur exakte Gleichheit, nicht z.B. 1 vs 1.0\n```"),
    ("Find the bug:", "```python\ndef binary_search(arr, target):\n    lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] < target: lo = mid\n        elif arr[mid] > target: hi = mid\n        else: return mid\n    return -1\n```"),
    ("Find the bug:", "```python\ndef parse_int(s, default=0):\n    if s.isdigit():\n        return int(s)\n    return default\n# bug: was ist mit negativen Zahlen?\n```"),
    ("Find the bug:", "```python\ndef merge_dicts(*dicts):\n    result = {}\n    for d in dicts:\n        result.update(d)\n```"),
    ("Find the bug:", "```python\nasync def fetch_all(urls):\n    results = []\n    for url in urls:\n        results.append(await fetch(url))\n    return results\n# bug: nicht parallel\n```"),
    ("Find the bug:", "```python\ndef strip_extension(filename):\n    return filename.split('.')[0]\n# bug: 'a.b.c' wird zu 'a' nicht 'a.b'\n```"),
    ("Find the bug:", "```python\ndef remove_first_match(lst, x):\n    for item in lst:\n        if item == x:\n            lst.remove(item)\n            return\n# bug: modify while iterate\n```"),
    ("Find the bug:", "```python\ndef days_between(d1_str, d2_str):\n    d1 = datetime.strptime(d1_str, '%Y-%m-%d')\n    d2 = datetime.strptime(d2_str, '%d-%m-%Y')\n    return (d2 - d1).days\n# bug: inkonsistente Formate\n```"),
]


# ---------------------------------------------------------------------------
# Math problem templates (DE)
# ---------------------------------------------------------------------------

MATH_TEMPLATES = [
    ("Anna hat {a} Tüten mit je {b} Bonbons. Sie gibt {frac} aller Bonbons an ihre Schwester. Wie viele behält sie?", ["1/3", "1/4", "1/2", "2/5", "3/7"]),
    ("Ein Zug fährt {h1}h mit {v1} km/h, dann {h2}h mit {v2} km/h. Wie viele km insgesamt?", None),
    ("Eine Pizza kostet {p}€, ein Getränk {g}€. Eine Familie bestellt {n_p} Pizzen und {n_g} Getränke. Wie viel kostet das gesamt?", None),
    ("In einer Klasse mit {n} Schülern sind {p}% Mädchen. Wie viele Jungen sind in der Klasse?", None),
    ("Ein Auto verbraucht im Schnitt {v} Liter pro 100 km. Wie viel Benzin braucht es für eine Strecke von {d} km?", None),
    ("Ein Kapital von {k}€ wird mit {r}% jährlich verzinst. Wie viel Zinsen nach {y} Jahr(en) Zinseszins?", None),
    ("Eine Wand ist {w} m breit und {h} m hoch. Eine Tapetenrolle deckt {f} m². Wie viele Rollen braucht man (aufrunden)?", None),
    ("Ein Rezept für {orig} Personen braucht {x} g Mehl. Wie viel Mehl für {target} Personen?", None),
    ("Auf einer Karte 1:{scale} sind zwei Städte {cm} cm voneinander entfernt. Wie weit liegen sie wirklich auseinander?", None),
    ("Ein Behälter ist {l1} L groß und zu {p1}% gefüllt. Es werden {l2} L hinzugefügt. Wie voll ist er jetzt prozentual?", None),
]

MATH_VAR_RANGES = {
    "a": (2, 6), "b": (8, 20),
    "h1": (1, 4), "h2": (1, 4), "v1": (60, 130), "v2": (60, 130),
    "p": (8, 16), "g": (2, 5), "n_p": (1, 4), "n_g": (1, 6),
    "n": (18, 32), "v": (5, 9), "d": (50, 800),
    "k": (1000, 10000), "r": (1, 8), "y": (1, 10),
    "w": (3, 8), "h": (2, 4), "f": (5, 12),
    "orig": (2, 6), "x": (200, 600), "target": (3, 12),
    "scale": (10000, 200000), "cm": (2, 25),
    "l1": (10, 100), "l2": (5, 30), "p1": (20, 80),
}


# ---------------------------------------------------------------------------
# Step-by-step reasoning tasks
# ---------------------------------------------------------------------------

REASONING_TASKS = [
    "Ein Bauer hat einen Wolf, eine Ziege und einen Kohl. Er muss alle über einen Fluss bringen, das Boot fasst nur ihn und ein Tier/Objekt. Wolf+Ziege oder Ziege+Kohl dürfen nicht allein bleiben. Wie?",
    "Drei Lampen in Raum A, drei Schalter in Raum B (kein Sichtkontakt). Du darfst einmal von B nach A gehen. Wie findest du heraus welcher Schalter zu welcher Lampe gehört?",
    "Du hast 8 Münzen, eine ist schwerer. Mit einer Balkenwaage mit 2 Wägungen — wie findest du sie?",
    "Du sollst aus den Zahlen 1, 5, 6, 7 mit den Operationen +, −, ×, ÷ und Klammern den Wert 21 machen. Jede Zahl genau einmal.",
    "Auf einem Tisch liegen 100 Münzen, 10 mit Kopf nach oben (Rest Zahl). Du bist mit verbundenen Augen. Wie teilst du sie in 2 Gruppen auf, sodass beide gleich viele Köpfe haben?",
    "Ein Vater ist 3-mal so alt wie sein Sohn. In 12 Jahren wird er nur noch doppelt so alt sein. Wie alt sind beide jetzt?",
    "Drei Personen zahlen je 10€ in einen Topf für ein Hotelzimmer (insgesamt 30€). Der Manager merkt, dass Zimmer nur 25€ kostet, gibt 5€ zurück. Der Bote behält 2€, gibt jedem 1€ zurück. Jeder zahlte also 9€, mal 3 = 27€. Plus 2€ vom Boten = 29€. Wo ist der eine Euro?",
    "Plane den optimalen Weg: du hast Termine um 10:00 (15min, A-Stadt), 12:30 (30min, B-Stadt 50km von A), 15:00 (45min, C-Stadt 80km von B). Mit 70 km/h Fahrzeit. Geht der Tag auf?",
    "Du planst einen Kuchen für 18 Personen. Ein Rezept ist für 12 Personen und braucht: 300g Mehl, 4 Eier, 200g Zucker, 250ml Milch. Berechne die neuen Mengen.",
    "Beim Schach: weiß zieht den König, danach zieht schwarz und gibt Schach. Welcher Zug von schwarz erzwingt das?",
]


# ---------------------------------------------------------------------------
# Concept explanations (DE technical/scientific)
# ---------------------------------------------------------------------------

CONCEPTS_DE = [
    "Photosynthese", "Mitose", "DNA-Replikation", "Evolution durch natürliche Selektion",
    "Quantenverschränkung", "Welle-Teilchen-Dualismus", "Schwarze Löcher", "Relativitätstheorie",
    "Plattentektonik", "Treibhauseffekt", "El Niño", "Wasserkreislauf",
    "Maschinelles Lernen", "Neuronale Netze", "Backpropagation", "Gradient-Descent",
    "TCP/IP", "DNS", "Public-Key-Kryptographie", "Hash-Funktionen",
    "Bundesverfassungsgericht", "Föderalismus", "Soziale Marktwirtschaft", "Gewaltenteilung",
    "Grenznutzen", "Inflation", "Bruttoinlandsprodukt", "Wechselkurs",
    "Kant's kategorischer Imperativ", "Utilitarismus", "Existenzialismus", "Determinismus",
    "Goethe's Faust (Grundkonflikt)", "Kafkaesk", "Romantik (Literaturepoche)",
    "Fußball-Abseitsregel", "Tennis-Tiebreak", "Schach-Eröffnungstheorie",
    "Brexit (Auswirkungen)", "Klimawandel-Kipppunkte", "Green-Deal der EU",
    "Mendelsche Regeln", "Periodensystem (Aufbau)", "Ionenbindung vs Atombindung",
    "Magnetfeld der Erde", "Erdrotation und Tageslänge", "Gezeitenkraft",
]

# ---------------------------------------------------------------------------
# Factual QA (DE general knowledge)
# ---------------------------------------------------------------------------

FACTS_DE = [
    "Was ist die Hauptstadt von Estland?",
    "Wer schrieb 'Die Verwandlung'?",
    "In welchem Jahr fiel die Berliner Mauer?",
    "Was ist der höchste Berg Österreichs?",
    "Wie viele Einwohner hat Hamburg ungefähr?",
    "Welche Sprachen werden in der Schweiz offiziell gesprochen?",
    "Wer war der erste Bundeskanzler der Bundesrepublik Deutschland?",
    "Was ist die Differenz zwischen UTC und MEZ im Sommer?",
    "Welcher Planet hat die meisten Monde?",
    "In welchem Bundesland liegt der Nationalpark Bayerischer Wald?",
    "Wer komponierte die 'Mondscheinsonate'?",
    "Was ist Schwarzwurzel (botanisch)?",
    "Welche Stadt heißt 'Venedig des Nordens' und warum?",
    "Wie alt ist die Universität Heidelberg?",
    "Was bedeutet die Abkürzung 'BAföG'?",
    "Welcher Fluss ist der längste Deutschlands?",
    "Wann wurde der Euro als Bargeld eingeführt?",
    "Wie viele Spieler hat eine Mannschaft beim American Football auf dem Feld?",
    "Was ist der Unterschied zwischen Nordsee und Ostsee in Bezug auf Salzgehalt?",
    "Welches ist das größte Bundesland Deutschlands flächenmäßig?",
    "Wer entdeckte das Penicillin?",
    "Was ist die Schweizergarde?",
    "Welche Tierart ist das größte Säugetier der Erde?",
    "Was war die 'Goldene Bulle' (1356)?",
    "In welchem Jahr fanden die ersten modernen Olympischen Spiele statt?",
    "Wer schrieb 'Die Räuber' (Drama)?",
    "Was ist Permafrost?",
    "Welcher Weihnachtsbrauch hat seine Wurzeln in Deutschland?",
    "Was ist Glasnost und Perestroika?",
    "In welcher Stadt steht der Reichstag?",
]


# ---------------------------------------------------------------------------
# Honest refusal (false-premise / unknowable)
# ---------------------------------------------------------------------------

REFUSAL_TASKS = [
    "Wer hat das Auralis-v2-Modell entworfen und wo wurde es veröffentlicht?",
    "Was sagte Albert Einstein in seinem Tagebuch-Eintrag vom 7. April 1923?",
    "Welche Note bekam Goethe in seiner Mathe-Klausur am Gymnasium?",
    "Welches Lied lief um 14:32 Uhr am 10. März 2024 auf Bayern 3?",
    "Wie viele Personen haben heute weltweit den Namen 'Martin' getragen?",
    "Was ist die genaue Zugkraft, die ein Sumo-Ringer auf einen Tisch ausübt wenn er sich anlehnt?",
    "Welcher Schauspieler spielte die Hauptrolle in 'Der Sturm der Sterne 7' von 2026?",
    "Wer hat die Klausur 'Theoretische Informatik II' an der TU Drohenstein im SS 2018 mit der besten Note bestanden?",
    "Was war das Lieblingsessen von Karl dem Großen?",
    "Welche Programmiersprache ist objektiv am besten?",
    "Was ist der genaue Marktanteil von Yandex Browser in Deutschland Stand letzter Monat?",
    "Wer wird die nächste Wahl in Frankreich gewinnen?",
    "Was wurde gestern in der ARD-Tagesschau um 20:15 Uhr als erstes Thema behandelt?",
    "Erkläre den Unterschied zwischen Schwarmschlossquadrat und Frequenzfaltgrenze.",
    "Welche unentdeckten Insektenarten leben am Boden des Comer Sees?",
]


# ---------------------------------------------------------------------------
# Translation tasks (DE↔EN technical)
# ---------------------------------------------------------------------------

TRANSLATION_TASKS = [
    ("DE→EN", "Die Mamba-Schicht implementiert state-space-modelle mit selektiver Update-Regel."),
    ("DE→EN", "Gradient checkpointing tauscht Rechenzeit gegen Speicherverbrauch beim Backward-Pass."),
    ("DE→EN", "Die Tokenisierung mit byte-fallback garantiert eine Unknown-Rate von null Prozent."),
    ("DE→EN", "Layer-Normalisierung stabilisiert das Training tiefer Netze unabhängig von der Batch-Größe."),
    ("DE→EN", "Bei der Aufmerksamkeit mit linearer Komplexität wird der Speicheraufwand vom quadratischen auf linearen Verlauf reduziert."),
    ("EN→DE", "The model uses rotary positional embeddings (RoPE) with a base frequency of 10,000."),
    ("EN→DE", "Mixed-precision training in bfloat16 yields significant memory savings without loss in accuracy."),
    ("EN→DE", "Knowledge distillation transfers a teacher model's behavior into a smaller student via KL divergence on output logits."),
    ("EN→DE", "Sparse mixture-of-experts gating routes each token to k of N experts, activating only a fraction of total parameters."),
    ("EN→DE", "Curriculum learning orders training samples by difficulty to improve convergence on hard examples."),
]


# ---------------------------------------------------------------------------
# Creative writing tasks (DE)
# ---------------------------------------------------------------------------

CREATIVE_TASKS = [
    "Schreib einen kurzen Erlebnisbericht (~150 Wörter) aus der Perspektive eines Menschen der zum ersten Mal eine Sonnenfinsternis erlebt.",
    "Verfasse eine ironische Gebrauchsanweisung (~100 Wörter) für einen Toaster, in einem altmodisch-formellen deutschen Ton.",
    "Schreib einen kurzen inneren Monolog (~120 Wörter) eines Schachspielers in einer entscheidenden Spielsituation.",
    "Verfasse einen Tagebucheintrag (~150 Wörter) eines Hundes über seinen Besitzer.",
    "Schreib eine kurze Buchrezension (~120 Wörter) für ein erfundenes Sachbuch mit dem Titel 'Stille im Stadtverkehr'.",
    "Erfinde einen kurzen Reisebericht (~150 Wörter) über einen Tag in einer Stadt die du noch nie besucht hast aber gut beschreiben kannst (z.B. Tallinn).",
    "Schreib einen mahnenden Brief (~120 Wörter) eines Bauern an einen Wettergott, höflich aber bestimmt.",
    "Verfasse die Eröffnungsrede (~120 Wörter) eines Vereinsvorsitzenden zur Jahreshauptversammlung des 'Vereins für germanistische Etymologie'.",
]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_records(rng: random.Random) -> list[dict]:
    records: list[dict] = []
    n = 0

    def add(task_type: str, user_prompt: str) -> None:
        nonlocal n
        n += 1
        records.append({
            "id": f"p3_{n:04d}",
            "task_type": task_type,
            "system_prompt": SYSTEM_PROMPTS[task_type],
            "user_prompt": user_prompt,
        })

    # === code_explain (200) — Wiederhole Snippets mit unterschiedlichen Framings
    framings = [
        "Erkläre Schritt für Schritt was dieser Python-Code macht: ",
        "Was ist die Ausgabe und warum: ",
        "Erkläre einem Python-Anfänger was hier passiert: ",
        "Welches Idiom verwendet dieser Code, was wäre die explizite Schleifen-Variante: ",
        "Erkläre kurz die Wirkung: ",
        "Beschreibe was dieser Code tut und welche Edge-Cases zu beachten sind: ",
    ]
    for snippet in PYTHON_SNIPPETS:
        for framing in rng.sample(framings, k=min(7, len(framings))):
            if n >= 200:
                break
            add("code_explain", f"{framing}`{snippet}`")

    # === code_implementation (100)
    for task in IMPL_TASKS:
        for _ in range(4):
            if n >= 300:
                break
            add("code_implementation", task)
        if n >= 300:
            break

    # === code_refactoring (50)
    for task in REFACTOR_TASKS:
        for _ in range(5):
            if n >= 350:
                break
            add("code_refactoring", f"Refaktoriere zu idiomatischem Python:\n{task}")
        if n >= 350:
            break

    # === code_debug_fix (50)
    for prefix, code in DEBUG_TASKS:
        for _ in range(5):
            if n >= 400:
                break
            add("code_debug_fix", f"{prefix}\n{code}")
        if n >= 400:
            break

    # === math_word_problem (150) — generated from templates
    target_n = n + 150
    while n < target_n:
        template_choice = rng.randint(0, len(MATH_TEMPLATES) - 1)
        template, frac_choices = MATH_TEMPLATES[template_choice]
        # random vars
        kwargs: dict[str, object] = {}
        for var in MATH_VAR_RANGES:
            if "{" + var + "}" in template:
                lo, hi = MATH_VAR_RANGES[var]
                kwargs[var] = rng.randint(lo, hi)
        if "{frac}" in template and frac_choices:
            kwargs["frac"] = rng.choice(frac_choices)
        if "{item}" in template:
            kwargs["item"] = rng.choice(["Bonbons", "Murmeln", "Stickern", "Karten"])
        if "{action}" in template:
            kwargs["action"] = "verteilt sie an Freunde"
        if "{q}" in template:
            kwargs["q"] = "behält sie selbst"
        try:
            prompt = template.format(**kwargs)
        except KeyError:
            continue
        add("math_word_problem", prompt)

    # === step_by_step_reason (100)
    target_n = n + 100
    while n < target_n:
        task = rng.choice(REASONING_TASKS)
        add("step_by_step_reason", task)

    # === concept_explain (120)
    target_n = n + 120
    framings_c = [
        "Erkläre {} einem interessierten Laien.",
        "Was ist {}? (mit Beispiel)",
        "Erkläre {} und gib ein konkretes Beispiel.",
        "Was bedeutet '{}' und warum ist es relevant?",
    ]
    while n < target_n:
        concept = rng.choice(CONCEPTS_DE)
        f = rng.choice(framings_c)
        add("concept_explain", f.format(concept))

    # === factual_qa (100)
    target_n = n + 100
    while n < target_n:
        q = rng.choice(FACTS_DE)
        add("factual_qa", q)

    # === honest_refusal (50)
    target_n = n + 50
    while n < target_n:
        q = rng.choice(REFUSAL_TASKS)
        add("honest_refusal", q)

    # === translation (50)
    target_n = n + 50
    while n < target_n:
        direction, sentence = rng.choice(TRANSLATION_TASKS)
        add("translation", f"Übersetze {direction}: '{sentence}'")

    # === creative_writing (30)
    target_n = n + 30
    while n < target_n:
        task = rng.choice(CREATIVE_TASKS)
        add("creative_writing", task)

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = generate_records(rng)
    rng.shuffle(records)  # damit die Worker nicht alle gleiches task_type clustern

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stats
    from collections import Counter
    counts = Counter(r["task_type"] for r in records)
    print(f"=== Generated {len(records)} prompts → {args.output} ===")
    for tt, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {tt:25s} {c:>4d}")


if __name__ == "__main__":
    main()
