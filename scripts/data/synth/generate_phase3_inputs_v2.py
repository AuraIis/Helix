"""Phase-3 SFT prompt generator V2 — Batch2 with lessons-learned + DE-deep.

Improvements over V1:
1. Expanded prompt pools (~3-5x more diversity per category)
2. DE-deep section: German law, politics, history, literature, regions, etc.
3. max_tokens cap on step_by_step_reason (avoid 5000-token reasoning ramble)
4. Sharpened system prompts to discourage excessive verbosity
5. Higher per-task target volumes

Target: ~6000-7000 records.

Usage:
    python scripts/data/synth/generate_phase3_inputs_v2.py \\
        --output raw/sft/synth/inputs/phase3_batch2.jsonl \\
        --seed 43
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# System prompts — sharpened for brevity where appropriate
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: dict[str, str] = {
    "code_explain": (
        "Du bist Auralis, ein hilfreicher bilingualer KI-Assistent. "
        "Erkläre Code-Snippets schrittweise auf nativem Deutsch. "
        "Beginne direkt mit der Erklärung, kein Vorspann. "
        "Zeige am Ende was das konkrete Ergebnis ist."
    ),
    "code_implementation": (
        "Du bist ein Senior Python Engineer. Schreib idiomatischen Code, "
        "kurz, mit edge-cases. Nur Code in einem Fence, KEIN Vorspann, "
        "KEINE ausschweifende Erklärung."
    ),
    "code_refactoring": (
        "Du bist Senior Python Engineer. Zeig den refaktorierten Code, dann "
        "1-3 stichpunktige Begründungen. Knapp."
    ),
    "code_debug_fix": (
        "Du bist Senior Python Engineer. Bug in 1-2 Sätzen, Fix als Code, "
        "ggf. relevante Edge-Cases kurz erwähnen. Knapp."
    ),
    "math_word_problem": (
        "Du bist Auralis. Mathe schrittweise auf Deutsch. Format: "
        "'Schritt 1: ...' usw, dann fettgedruckte finale Antwort."
    ),
    "step_by_step_reason": (
        "Du bist Auralis. Logik-Aufgaben löst du Schritt-für-Schritt auf "
        "Deutsch. **Knapp und präzise** — vermeide ausschweifende "
        "Zwischen-Kommentare. Nur die nötigen Schritte, dann finale "
        "Antwort. Keine über-detaillierten Berechnungen wenn ein "
        "Überschlag reicht."
    ),
    "concept_explain": (
        "Du bist Auralis, ein KI-Tutor. Erkläre Konzepte auf nativem "
        "Deutsch, didaktisch, mit konkretem Beispiel. Vermeide Anglizismen "
        "wo möglich. Zielgruppe: interessierter Laie."
    ),
    "factual_qa": (
        "Du bist Auralis, faktenorientierter KI-Assistent. Beantworte "
        "Wissensfragen knapp aber vollständig auf Deutsch. Wenn unsicher, "
        "sage es offen."
    ),
    "honest_refusal": (
        "Du bist ein ehrlicher KI-Assistent.\n\n"
        "ABSOLUT KRITISCHE REGEL: NIEMALS spezifische Fakten erfinden — "
        "keine Namen, keine Daten, keine Jahre, keine Zahlen, keine Orte, "
        "keine Zitate die du nicht 100% sicher kennst.\n\n"
        "Bei Fragen die du nicht zuverlässig beantworten kannst:\n"
        "1. Sag klar dass du es nicht weißt\n"
        "2. ERLAUBT: 1-2 Sätze Kontext WARUM die Frage problematisch ist "
        "(falsche Annahme, Real-time-Daten nicht zugänglich, etc.) — aber "
        "NUR mit Fakten die du 100% sicher kennst\n"
        "3. VERBOTEN: alternative spezifische Antworten mit 'vermutlich', "
        "'wahrscheinlich', 'könnte gewesen sein', 'soll', 'angeblich'\n"
        "4. VERBOTEN: spezifische Namen, Daten, Zahlen die du nicht "
        "zuverlässig kennst\n"
        "5. Wenn unsicher zwischen 'weiß ich' vs 'weiß ich nicht' — "
        "entscheide IMMER für 'weiß ich nicht'\n\n"
        "Beispiele GUTER Refusals:\n"
        "  • 'Ich weiß es nicht.' (perfekt für reine Unknowables)\n"
        "  • 'Das ist mir nicht bekannt.' (knapp)\n"
        "  • 'Goethe besuchte kein klassisches Gymnasium — er wurde "
        "überwiegend von Hauslehrern unterrichtet. Eine Mathe-Klausur-Note "
        "ist nicht überliefert.' (konzis mit VERIFIZIERBAREM Kontext)\n\n"
        "Beispiele SCHLECHTER Refusals (NIEMALS so):\n"
        "  • 'Den Bürostuhl entwarf vermutlich Friedrich Bertuch im Jahr "
        "1794...' (Halluzination)\n"
        "  • 'Wahrscheinlich rauchte Bismarck Havanna-Zigarren...' (Spekulation)\n"
        "  • 'Sokrates letzte Worte waren wohl: ...' (erfundene Zitate)\n\n"
        "Antworte direkt, kurz, ohne Vorspann."
    ),
    "translation": (
        "Du bist Fach-Übersetzer für technische Texte zwischen Deutsch und "
        "Englisch. Bewahre Fachterminologie. Antworte mit der Übersetzung, "
        "ggf. einer kurzen Notiz zu nicht-trivialen Termini."
    ),
    "creative_writing": (
        "Du bist Auralis. Schreib in nativem Deutsch, idiomatisch, "
        "stilistisch passend zur Aufgabe. Keine Meta-Kommentare."
    ),
}

# ---------------------------------------------------------------------------
# Code snippets (~70 — substantially expanded)
# ---------------------------------------------------------------------------

PYTHON_SNIPPETS = [
    # List/dict/set comprehensions
    "[x**2 for x in range(20) if x % 3 == 0]",
    "{k: v for k, v in d.items() if v is not None}",
    "{x: x**2 for x in range(10)}",
    "{c for word in words for c in word}",
    "[(i, w) for i, w in enumerate(words) if len(w) > 3]",
    "sum(x**2 for x in range(100) if x % 2)",
    "max((p for p in people if p.age >= 18), key=lambda p: p.income)",
    "min(numbers, default=0)",
    "any(x < 0 for x in nums)",
    "all(s.startswith('a') for s in words)",
    # Decorators
    "@functools.lru_cache(maxsize=128)\ndef fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
    "@dataclass(frozen=True)\nclass Point:\n    x: int\n    y: int",
    "@contextmanager\ndef tmpdir():\n    d = tempfile.mkdtemp()\n    try:\n        yield d\n    finally:\n        shutil.rmtree(d)",
    "@functools.singledispatch\ndef serialize(obj):\n    raise TypeError",
    # Context managers
    "with open('data.json') as f, open('out.json', 'w') as g:\n    json.dump(json.load(f), g, indent=2)",
    "with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:\n    results = list(pool.map(fetch, urls))",
    "with sqlite3.connect(db) as conn:\n    conn.execute('INSERT INTO logs VALUES (?, ?)', (ts, msg))",
    # Itertools / functools / collections
    "from collections import Counter\nCounter('mississippi').most_common(2)",
    "list(itertools.chain.from_iterable([[1,2],[3,4],[5]]))",
    "list(itertools.groupby([1,1,2,2,2,3], key=lambda x: x))",
    "list(itertools.accumulate([1,2,3,4]))",
    "list(itertools.combinations([1,2,3,4], 2))",
    "list(itertools.product([0,1], repeat=3))",
    "functools.reduce(operator.mul, range(1, 6))",
    "operator.itemgetter(*indices)(row)",
    "collections.OrderedDict.fromkeys(seen).keys()",
    "collections.defaultdict(list)['key'].append('value')",
    # Async
    "async def fetch_all(urls):\n    return await asyncio.gather(*[fetch(u) for u in urls])",
    "async with aiohttp.ClientSession() as s:\n    async with s.get(url) as r:\n        data = await r.json()",
    "asyncio.run(asyncio.wait_for(task, timeout=10.0))",
    "async for line in stream:\n    if line.startswith('ERROR'):\n        break",
    # Error handling
    "try:\n    n = int(s)\nexcept ValueError:\n    n = 0",
    "try:\n    risky()\nexcept (TypeError, ValueError) as e:\n    log.error('failed: %s', e)",
    "try:\n    with lock:\n        critical()\nexcept TimeoutError:\n    fallback()",
    # Slicing
    "matrix[::-1]",
    "s[::2]",
    "lst[len(lst)//2:]",
    "arr[start:stop:step]",
    # Dict tricks
    "d.setdefault('items', []).append(x)",
    "{**a, **b, 'extra': 1}",
    "d | {'new_key': 'new_value'}",  # 3.9+
    "dict(zip(keys, values))",
    "max(d.items(), key=lambda kv: kv[1])",
    # Functional
    "list(map(int, '1 2 3 4'.split()))",
    "list(filter(lambda x: x.startswith('a'), words))",
    "sorted(items, key=lambda x: (-x.priority, x.id))",
    # Pattern matching (3.10+)
    "match cmd:\n    case 'start': run()\n    case 'stop': halt()\n    case _: print('unknown')",
    "match point:\n    case (0, 0): print('origin')\n    case (x, 0): print(f'on x at {x}')\n    case (_, y): print(f'y={y}')",
    "match shape:\n    case Circle(radius=r): area = math.pi * r**2\n    case Rect(w=w, h=h): area = w*h",
    # Pathlib
    "Path('logs').glob('**/*.log')",
    "Path(__file__).resolve().parent.parent / 'data'",
    "Path('config.yaml').read_text(encoding='utf-8')",
    "list(Path('.').rglob('*.py'))",
    # Numpy idioms
    "arr[arr > arr.mean()]",
    "np.where(grid == 0, -1, grid)",
    "(x - x.mean(0)) / x.std(0)",
    "np.concatenate([a.flatten(), b.flatten()])",
    "np.argsort(scores)[::-1][:k]",
    # String formatting
    "f'{value:>10.2f} ({pct:.1%})'",
    "':'.join(f'{b:02x}' for b in mac)",
    "f'{name=!r}'",
    "textwrap.dedent('''\\n    Hello\\n    World\\n''').strip()",
    # Generators
    "def chunks(lst, n):\n    for i in range(0, len(lst), n):\n        yield lst[i:i+n]",
    "def reversed_lines(path):\n    with open(path, 'rb') as f:\n        for line in reversed(list(f)):\n            yield line.decode()",
    # Class snippets
    "@dataclass\nclass Point:\n    x: int\n    y: int = 0",
    "class Counter:\n    def __init__(self): self._n = 0\n    def __call__(self): self._n += 1; return self._n",
    "class Singleton:\n    _instance = None\n    def __new__(cls):\n        if cls._instance is None:\n            cls._instance = super().__new__(cls)\n        return cls._instance",
    # Type hints (modern)
    "def greet(name: str | None = None) -> str:\n    return f'Hello {name or \"World\"}'",
    "ItemList: TypeAlias = list[tuple[str, int]]",
    "T = TypeVar('T')\ndef first_or_default(items: list[T], default: T) -> T:\n    return items[0] if items else default",
]

# ---------------------------------------------------------------------------
# Implementation tasks (~50)
# ---------------------------------------------------------------------------

IMPL_TASKS = [
    "Schreib eine Funktion safe_divide(a, b) die für b=0 None zurückgibt.",
    "Schreib chunked(items, n) die items in n-große Listen aufteilt.",
    "Schreib deduplicate_keep_order(seq) die Duplikate entfernt aber Ordnung erhält.",
    "Schreib is_palindrome(s) case-insensitive, ignoriert whitespace + punctuation.",
    "Schreib flatten(nested) die beliebig tief verschachtelte Listen zu flach macht.",
    "Schreib read_lines_streaming(path) die Zeilen einer großen Datei lazy yieldet.",
    "Schreib retry(fn, max_attempts=3, delay=1.0) als Decorator mit exponential backoff.",
    "Schreib parse_iso_duration('PT1H30M') -> Sekunden.",
    "Schreib human_readable_bytes(n) für Bytes → KB/MB/GB/TB (binary).",
    "Schreib RingBuffer(capacity) mit append() und items() (FIFO drops oldest).",
    "Schreib median(values) ohne statistics-Modul, handle leere Liste.",
    "Schreib top_k(values, k) effizient mit heapq.",
    "Schreib count_words(text) -> Counter, normalisiert (lower, no punct).",
    "Schreib levenshtein(s, t) edit-distance.",
    "Schreib merge_sorted(a, b) zwei sortierte Listen in O(n+m).",
    "Schreib LRUCache(capacity) mit get(key) und put(key, value) in O(1).",
    "Schreib validate_email(addr) -> bool, RFC-pragmatisch.",
    "Schreib run_with_timeout(fn, args, timeout_s) via threading.Thread + join.",
    "Schreib dict_diff(old, new) -> dict mit added/removed/changed keys.",
    "Schreib async fetch_concurrent(urls, max_concurrency=10) mit asyncio.Semaphore.",
    "Schreib natural_sort_key(s) für sorted(): file1 < file2 < file10.",
    "Schreib Stopwatch als Context-Manager mit elapsed-Property.",
    "Schreib iter_csv_rows(path) als Generator von dicts.",
    "Schreib checksum_file(path, algo='sha256') chunked-Hashing für large files.",
    "Schreib ensure_unique_path(target) — bei Existenz _1, _2 anhängen.",
    "Schreib partition(pred, iterable) -> (truthy_list, falsy_list).",
    "Schreib batched(iterable, n) als Generator von n-Tupeln (last may be short).",
    "Schreib group_by(items, key) -> dict[key, list[items]].",
    "Schreib most_common_string(strings) -> string das am häufigsten vorkommt.",
    "Schreib parse_size('1.5 GB') -> Bytes.",
    "Schreib safe_get(d, *keys, default=None) für nested dict access.",
    "Schreib timer Decorator der die Laufzeit per logger logged.",
    "Schreib singleton Decorator für Klassen.",
    "Schreib RateLimiter(calls_per_sec) — sliding window rate limit.",
    "Schreib observable Pattern: Observer + observer.subscribe(callback).",
    "Schreib EventEmitter mit on(event, handler), emit(event, *args).",
    "Schreib AsyncQueue(maxsize) mit await put() / get() — eigene impl ohne asyncio.Queue.",
    "Schreib JsonStreamer(path) — read jsonl lazy, write_one(record) appends atomically.",
    "Schreib FileLocker(path) als Context-Manager mit POSIX flock fallback to no-op auf Windows.",
    "Schreib running_average(window) als generator-coroutine die letzten N werte mittelt.",
    "Schreib trie data structure mit insert(word), search(word), starts_with(prefix).",
    "Schreib bloom_filter(capacity, error_rate) mit add(item), __contains__(item).",
    "Schreib parse_simple_calculator('3 + 4 * 2') ohne eval — recursive descent.",
    "Schreib Caesar-Cipher encode(text, shift) und decode — preserve case + non-alpha.",
    "Schreib RGB ↔ HSL converter (Tupel rgb_to_hsl(r, g, b) → (h, s, l)).",
    "Schreib Map-Reduce über Multiprocessing: map_reduce(items, mapper, reducer).",
    "Schreib HashMap(initial_capacity=16) mit linear probing — keine dict-Verwendung.",
    "Schreib graph_shortest_path(graph, start, end) BFS auf unweighted graph.",
    "Schreib QuadTree für 2D points, query rectangular region.",
    "Schreib Reservoir-Sampling sample_k(stream, k) für unbekannte Stream-Länge.",
]

# ---------------------------------------------------------------------------
# Refactor + Debug
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
    "```python\nresult = []\nfor x in items:\n    for y in x.children:\n        result.append((x.id, y.id))\n```",
    "```python\nfiltered = []\nfor item in collection:\n    if item.score > threshold:\n        filtered.append(item.value)\n```",
    "```python\nif name == 'admin' or name == 'root' or name == 'superuser':\n    grant_access()\n```",
    "```python\nfor i in range(len(items)):\n    print(f'{i}: {items[i]}')\n```",
    "```python\ntotal = 0\nfor x in nums:\n    total = total + x\nreturn total\n```",
    "```python\nresult = sorted([x for x in items])[::-1][:5]\n```",
    "```python\nif type(x) == int:\n    process_int(x)\nelif type(x) == str:\n    process_str(x)\n```",
    "```python\nfound = False\nfor x in lst:\n    if x == target:\n        found = True\n        break\nreturn found\n```",
    "```python\ndef sum_squares(n):\n    s = 0\n    for i in range(1, n+1):\n        s = s + i*i\n    return s\n```",
    "```python\nlines = []\nf = open(path)\nfor line in f.readlines():\n    lines.append(line.strip())\nf.close()\n```",
]

DEBUG_TASKS = [
    ("Find the bug:", "```python\ndef avg(nums): return sum(nums) / len(nums) - 1\n```"),
    ("Find the bug:", "```python\ndef factorial(n):\n    result = 0\n    for i in range(1, n+1):\n        result *= i\n    return result\n```"),
    ("Find the bug:", "```python\ndef contains_duplicates(lst):\n    return len(lst) != len(set(lst))\n```"),
    ("Find the bug:", "```python\ndef binary_search(arr, target):\n    lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] < target: lo = mid\n        elif arr[mid] > target: hi = mid\n        else: return mid\n    return -1\n```"),
    ("Find the bug:", "```python\ndef parse_int(s, default=0):\n    if s.isdigit():\n        return int(s)\n    return default\n```"),
    ("Find the bug:", "```python\ndef merge_dicts(*dicts):\n    result = {}\n    for d in dicts:\n        result.update(d)\n```"),
    ("Find the bug:", "```python\nasync def fetch_all(urls):\n    results = []\n    for url in urls:\n        results.append(await fetch(url))\n    return results\n```"),
    ("Find the bug:", "```python\ndef strip_extension(filename):\n    return filename.split('.')[0]\n```"),
    ("Find the bug:", "```python\ndef remove_first_match(lst, x):\n    for item in lst:\n        if item == x:\n            lst.remove(item)\n            return\n```"),
    ("Find the bug:", "```python\ndef days_between(d1_str, d2_str):\n    d1 = datetime.strptime(d1_str, '%Y-%m-%d')\n    d2 = datetime.strptime(d2_str, '%d-%m-%Y')\n    return (d2 - d1).days\n```"),
    ("Find the bug:", "```python\ndef get_user(uid):\n    cursor.execute('SELECT * FROM users WHERE id=' + str(uid))\n    return cursor.fetchone()\n```"),
    ("Find the bug:", "```python\nclass Counter:\n    count = 0\n    def increment(self):\n        Counter.count += 1\n```"),
    ("Find the bug:", "```python\ndef tax(amount, rate=None):\n    if rate is None:\n        rate = []\n    rate.append(amount * 0.19)\n    return rate\n```"),
    ("Find the bug:", "```python\ndef batch_iter(items, n):\n    for i in range(0, len(items), n):\n        yield items[i:i+n+1]\n```"),
    ("Find the bug:", "```python\ndef reverse_words(s):\n    words = s.split()\n    return ' '.join(words.reverse())\n```"),
    ("Find the bug:", "```python\ndef average(nums):\n    if not nums:\n        return None\n    return sum(nums) / len(nums)\n# subtle: integer division on Python 2?\n```"),
    ("Find the bug:", "```python\ndef get_or_create(d, key):\n    if not d.get(key):\n        d[key] = []\n    return d[key]\n# bug: was wenn key existiert mit value 0 oder ''?\n```"),
    ("Find the bug:", "```python\nfor i, item in enumerate(items):\n    if item.expired:\n        del items[i]\n```"),
    ("Find the bug:", "```python\nimport threading\ncounter = 0\ndef inc():\n    global counter\n    counter += 1\n# ohne lock — race condition\n```"),
    ("Find the bug:", "```python\ndef serialize(d):\n    return json.dumps(d, default=str)\n# subtle: bei dict mit datetime keys verhalten?\n```"),
]

# ---------------------------------------------------------------------------
# Math templates
# ---------------------------------------------------------------------------

MATH_TEMPLATES = [
    ("Anna hat {a} Tüten mit je {b} Bonbons. Sie gibt {frac} aller Bonbons an ihre Schwester. Wie viele behält sie?", ["1/3", "1/4", "1/2", "2/5", "3/7"]),
    ("Ein Zug fährt {h1}h mit {v1} km/h, dann {h2}h mit {v2} km/h. Wie viele km insgesamt?", None),
    ("Eine Pizza kostet {p}€, ein Getränk {g}€. Eine Familie bestellt {n_p} Pizzen und {n_g} Getränke. Wie viel kostet das gesamt?", None),
    ("In einer Klasse mit {n} Schülern sind {p}% Mädchen. Wie viele Jungen sind in der Klasse?", None),
    ("Ein Auto verbraucht im Schnitt {v} Liter pro 100 km. Wie viel Benzin braucht es für eine Strecke von {d} km?", None),
    ("Ein Kapital von {k}€ wird mit {r}% jährlich verzinst. Wie viel Zinsen nach {y} Jahren Zinseszins?", None),
    ("Eine Wand ist {w} m breit und {h} m hoch. Eine Tapetenrolle deckt {f} m². Wie viele Rollen braucht man (aufrunden)?", None),
    ("Ein Rezept für {orig} Personen braucht {x} g Mehl. Wie viel Mehl für {target} Personen?", None),
    ("Auf einer Karte 1:{scale} sind zwei Städte {cm} cm voneinander entfernt. Wie weit liegen sie wirklich auseinander?", None),
    ("Ein Behälter ist {l1} L groß und zu {p1}% gefüllt. Es werden {l2} L hinzugefügt. Wie voll ist er jetzt prozentual?", None),
    ("Ein Tank fasst {v} Liter und ist mit {p}% gefüllt. Pro Stunde laufen {l} Liter aus. Wie lange bis er leer ist?", None),
    ("Eine Aktie kostet {pi}€ und steigt um {p1}%, dann fällt sie um {p2}%. Wie viel ist sie nun wert?", None),
    ("Ein Würfel hat eine Kantenlänge von {a} cm. Wie groß sind Volumen und Oberfläche?", None),
    ("Ein Kreis hat den Radius {r} cm. Wie groß ist der Umfang und die Fläche (zwei Nachkommastellen)?", None),
    ("Eine Maschine produziert {n} Teile pro Stunde. Wie viele Teile in einer 8-Stunden-Schicht und wie viele in einer Woche bei 5 Schichten?", None),
    ("Ein Sportler trinkt pro Tag {l} Liter Wasser. Wie viel Wasser im Jahr ({y}) und entspricht das wie vielen 1.5-L-Flaschen?", None),
    ("Ein Hotel hat {n} Zimmer, davon {m} Doppelzimmer und der Rest Einzelzimmer. Wie groß ist der Anteil Einzelzimmer (in Prozent)?", None),
    ("Bei einer Befragung von {n} Personen sagen {a} 'ja', {b} 'nein', der Rest 'weiß nicht'. Wie viele 'weiß nicht'?", None),
    ("Eine Druckerei berechnet {p}€ Grundgebühr plus {pp}ct pro Seite. Was kostet ein Auftrag mit {pages} Seiten?", None),
    ("Ein Auto kostet {price}€ neu und verliert {p}% pro Jahr an Wert. Was ist es nach {y} Jahren wert?", None),
    ("Eine Rechtecksfläche misst {a} m × {b} m. Wie groß ist Fläche und Diagonale (auf cm gerundet)?", None),
    ("Ein Reifen hat {km1} km gehalten. Bei jährlicher Fahrleistung von {km2} km — wie viele Jahre?", None),
    ("Ein Sparbuch wird mit {r}% jährlich verzinst. Wie lange dauert es bis sich {k}€ verdoppeln (Zinseszins, ungefähr)?", None),
]

MATH_VAR_RANGES = {
    "a": (2, 8), "b": (4, 25),
    "h1": (1, 5), "h2": (1, 5), "v1": (50, 140), "v2": (50, 140),
    "p": (5, 20), "g": (2, 6), "n_p": (1, 5), "n_g": (1, 8),
    "n": (15, 40), "v": (4, 10), "d": (50, 1200),
    "k": (1000, 25000), "r": (1, 8), "y": (1, 15),
    "w": (3, 10), "h": (2, 5), "f": (4, 14),
    "orig": (2, 8), "x": (150, 800), "target": (3, 18),
    "scale": (10000, 250000), "cm": (2, 30),
    "l1": (10, 200), "l2": (5, 60), "p1": (10, 90),
    "l": (5, 50), "pi": (10, 200), "p1%": (3, 25), "p2%": (3, 20),
    "pp": (3, 15), "pages": (10, 500), "price": (15000, 60000),
    "m": (5, 20), "km1": (40000, 80000), "km2": (10000, 30000),
}

# ---------------------------------------------------------------------------
# Reasoning tasks (~35 — expanded)
# ---------------------------------------------------------------------------

REASONING_TASKS = [
    "Ein Bauer hat einen Wolf, eine Ziege und einen Kohl. Er muss alle über einen Fluss bringen, das Boot fasst nur ihn und ein Tier/Objekt. Wolf+Ziege oder Ziege+Kohl dürfen nicht allein bleiben. Wie?",
    "Drei Lampen in Raum A, drei Schalter in Raum B (kein Sichtkontakt). Du darfst einmal von B nach A gehen. Wie findest du heraus welcher Schalter zu welcher Lampe gehört?",
    "Du hast 8 Münzen, eine ist schwerer. Mit einer Balkenwaage und 2 Wägungen — wie findest du sie?",
    "Du hast 12 Münzen, eine ist falsch (leichter ODER schwerer). Mit 3 Wägungen finde sie und sage ob leichter/schwerer.",
    "Forme aus 1, 5, 6, 7 mit +, -, *, /, () genau 21. Jede Zahl genau einmal.",
    "Auf dem Tisch: 100 Münzen, 10 Kopf nach oben (Rest Zahl). Augen verbunden. Teile in 2 Gruppen mit gleicher Anzahl Köpfe.",
    "Vater 3x älter als Sohn. In 12 Jahren nur noch 2x. Wie alt sind beide?",
    "Drei Personen zahlen je 10€ (=30€) ins Hotel. Manager: Zimmer kostet nur 25€, gibt 5€ zurück. Bote behält 2€, gibt jedem 1€. Jeder zahlte 9€ × 3 = 27€ + 2€ Bote = 29€. Wo der eine Euro?",
    "Plane Tag: 10:00 (15min, A), 12:30 (30min, B 50km von A), 15:00 (45min, C 80km von B). 70 km/h. Geht der Tag?",
    "Kuchen für 18 statt 12 Personen. Original: 300g Mehl, 4 Eier, 200g Zucker, 250ml Milch. Berechne neue Mengen.",
    "5 Häuser in Reihe (Farben: Rot, Grün, Blau, Gelb, Weiß). Zentrum: Grün. Rot rechts neben Weiß. Blau nicht neben Weiß. Wo Gelb?",
    "Sandwich-Logik: Anna isst nur Schinken oder Käse. Bert isst nur Käse oder Tomate. Charlie isst nur was Anna+Bert auch essen. Was isst Charlie?",
    "Zwei Eimer (3L, 5L), unbegrenzt Wasser. Wie misst du genau 4L ab?",
    "Ein Brett aus 8x8 Feldern, 2 diagonale Eckfelder entfernt (also 62). Kann man es mit 31 Dominos (1x2) bedecken?",
    "Du hast 2 Eier und 100 Stockwerke. Finde mit minimaler Anzahl Würfen das höchste Stockwerk wo ein Ei nicht zerbricht.",
    "Plane Picknick für 5 Personen. Brötchen reichen für 3 (jeder 2 Stück). Brot reicht für 4 (jeder 4 Scheiben). Wie viele Brötchen+Brote dazukaufen?",
    "Ein Boot kann 250 kg tragen. Familie: Vater 90, Mutter 70, Kind 30. Sie wollen über Fluss. Wie viele Fahrten minimum (jede Person muss rüber)?",
    "Kette: 4 Stücke à 3 Glieder. Schmied: 2€ pro Glied öffnen, 3€ pro Glied schließen. Wie kettest du sie minimal-billig zu einem Ring?",
    "Du würfelst 3 standard-Würfel. Wahrscheinlichkeit dass die Summe genau 10 ist?",
    "5 gleiche Hemden, 4 Hosen, 3 Schuhe. Wie viele verschiedene Outfits (Hemd+Hose+Schuhe)?",
    "Ein Familienauto braucht 7L/100km. Stadtfahrt 4km täglich, Wochenende 80km. Wie viele Liter im Monat (4 Wochen)?",
    "Ein See hat eine Seerose. Sie verdoppelt sich täglich. An Tag 30 ist der See ganz bedeckt. An welchem Tag ist er halb bedeckt?",
    "Ein Becher Tee bei 90°C kühlt um 5% pro Minute. Nach wie vielen Minuten bei 50°C (Raum-Temp angenommen)?",
    "Schach-Bauer auf c2. Welche Felder kann er im nächsten Zug erreichen? Begründe.",
    "Du sortierst 8 Karten mit Insertion-Sort. Wie viele Vergleiche im Worst-Case?",
    "Ein Programmierer arbeitet 6h pro Tag, 220 Tage im Jahr. Sein Stundensatz ist 80€. Wie viel verdient er brutto im Jahr?",
    "Wenn 4 Maler eine Wand in 6 Stunden streichen — wie lange brauchen 6 Maler?",
    "Ein Pool wird durch Zufluss in 6h voll, durch Abfluss in 8h leer. Wie lang bei beiden offen?",
    "Drei Schüler haben Durchschnitt 70%, 75%, 80% in 4 Klassen. Was ist der Gesamt-Durchschnitt?",
    "Ein Programmierer commit'et durchschnittlich 5x pro Tag, 5 Tage die Woche, 48 Wochen im Jahr. Wie viele Commits?",
    "Du würfelst 2 Würfel. Wahrscheinlichkeit dass die Summe gerade ist?",
    "Eine Schule hat 600 Schüler in 24 Klassen, mit Durchschnitt 25. Wie viele Klassen haben mehr als 25 Schüler wenn 14 Klassen nur 22 haben?",
    "Du hast 7 verschiedene Bücher. Wie viele Möglichkeiten 3 in eine Reihe zu stellen (Reihenfolge wichtig)?",
    "Bei einer Lottoziehung 6 aus 49 — wie hoch ist die Wahrscheinlichkeit für 6 Richtige?",
    "Ein Auto fährt 60km bergauf (40 km/h) und 60km bergab (60 km/h). Was ist die Durchschnittsgeschwindigkeit?",
]

# ---------------------------------------------------------------------------
# Concept explanations DE (~120 — expanded)
# ---------------------------------------------------------------------------

CONCEPTS_DE = [
    # Bio
    "Photosynthese", "Mitose", "Meiose", "DNA-Replikation", "Translation (Bio)",
    "Proteinbiosynthese", "Evolution durch natürliche Selektion", "Mendelsche Regeln",
    "Endokrines System", "Synapse", "Mitochondrien", "Zellatmung", "Enzymkatalyse",
    "Biotop", "Symbiose", "Ökosystem", "Nahrungskette",
    # Physik
    "Quantenverschränkung", "Welle-Teilchen-Dualismus", "Schwarze Löcher",
    "Spezielle Relativitätstheorie", "Allgemeine Relativitätstheorie",
    "Heisenbergsche Unschärferelation", "Doppler-Effekt", "Resonanz",
    "Magnetfeld", "Lorentzkraft", "Photonen", "Halbleiter",
    "Supraleitung", "Brownsche Bewegung", "Gravitationswellen",
    # Chemie
    "Periodensystem (Aufbau)", "Ionenbindung vs Atombindung",
    "Reduktion und Oxidation", "Säure-Base-Theorie nach Brønsted",
    "Stereoisomerie", "Polymere", "Katalysator (chemisch)",
    # Geo
    "Plattentektonik", "Treibhauseffekt", "El Niño", "Wasserkreislauf",
    "Magnetfeld der Erde", "Gezeitenkraft", "Vulkanismus",
    # Informatik
    "Maschinelles Lernen", "Neuronale Netze", "Backpropagation",
    "Gradient-Descent", "Overfitting", "Cross-Validation",
    "Big-O-Notation", "Hash-Funktionen", "Public-Key-Kryptographie",
    "TCP/IP", "DNS", "REST-API", "Container (Docker)",
    "Versionskontrolle (Git)", "Datenbanken (relational vs NoSQL)",
    "Compiler vs Interpreter", "Garbage Collection",
    # Recht/Politik DE
    "Bundesverfassungsgericht", "Föderalismus", "Soziale Marktwirtschaft",
    "Gewaltenteilung", "Subsidiaritätsprinzip", "Bundesrat (DE)",
    "Verhältniswahlrecht", "Tarifautonomie", "Mitbestimmung",
    "Grundrechte", "Föderalismusreform", "EU-Mehrheitsentscheidung",
    # Wirtschaft
    "Grenznutzen", "Inflation", "Bruttoinlandsprodukt", "Wechselkurs",
    "Marktmechanismus", "Externe Effekte", "Monopolstellung",
    "Konjunkturzyklus", "Geldpolitik", "Steuerprogression",
    # Philosophie
    "Kant's kategorischer Imperativ", "Utilitarismus", "Existenzialismus",
    "Determinismus", "Sokratische Methode", "Stoizismus",
    "Hegelsche Dialektik", "Rawls' Schleier des Nichtwissens",
    # Lit/Kunst
    "Goethe's Faust (Grundkonflikt)", "Kafkaesk",
    "Romantik (Literaturepoche)", "Expressionismus",
    "Bauhaus (Stilrichtung)", "Die Aufklärung",
    "Sturm und Drang", "Brechts episches Theater",
    # Sport/Spiele
    "Fußball-Abseitsregel", "Tennis-Tiebreak", "Schach-Eröffnungstheorie",
    "Doppel-Tiebreak (Tennis)", "Drei-Punkte-Regel (Fußball)",
    # Politik aktuell
    "Brexit (Auswirkungen)", "Klimawandel-Kipppunkte", "Green-Deal der EU",
    "Energiewende", "Schuldenbremse",
    # DE-Kultur
    "Karneval vs Fasching vs Fastnacht", "Reinheitsgebot (Bier)",
    "Oktoberfest", "Tag der Deutschen Einheit", "Sankt Martinszug",
]

# ---------------------------------------------------------------------------
# Factual Q&A DE general (~50)
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
    "Was ist der höchste Punkt der Schweiz?",
    "Wer schrieb 'Das Boot' (Roman)?",
    "Was bedeutet 'Bundeskanzlerprinzip'?",
    "Welche Farbe hat die Flagge der Türkei?",
    "Was ist Nibelungentreue?",
    "Wer war Franz Beckenbauer?",
    "Was ist die Funktion eines Katalysators?",
    "Wie heißt der niedrigste Punkt auf der Erde (Land)?",
    "Was ist ein Fjord?",
    "Wer war Marie Curie und wofür war sie bekannt?",
    "Welcher Kontinent ist der bevölkerungsreichste?",
    "Was war die Hanse?",
    "Wer schrieb '1984'?",
    "Was ist die Nationalfarbe der Bundesländer-Symbole?",
    "Welche Sprache spricht man in Brasilien?",
    "Wie alt wird ein Eichbaum durchschnittlich?",
    "Was ist eine Wasserscheide?",
    "Wer war Kaiser Wilhelm II.?",
    "Was ist Gesteinszerlegung durch Frost?",
    "Wann wurde die Bundesrepublik Deutschland gegründet?",
]

# ---------------------------------------------------------------------------
# DE-DEEP: Rechtsfragen / Politik / Geschichte / Geografie / Sprache / Kultur
# ---------------------------------------------------------------------------

DE_LEGAL_QA = [
    "Was regelt § 433 BGB?",
    "Was ist der Unterschied zwischen Anfechtung und Widerruf im BGB?",
    "Was ist eine 'Willenserklärung' im juristischen Sinne?",
    "Was bedeutet 'in dubio pro reo'?",
    "Was ist ein Insichgeschäft (BGB § 181)?",
    "Welche Rolle hat das Bundesverfassungsgericht?",
    "Was ist eine einstweilige Verfügung?",
    "Was unterscheidet Eigentum von Besitz juristisch?",
    "Was sind die drei Gewalten in Deutschland nach GG?",
    "Was ist die Ewigkeitsklausel (Art. 79 Abs. 3 GG)?",
    "Was ist der Unterschied zwischen Zivilrecht und Strafrecht?",
    "Was sind die Grundrechtsschranken?",
    "Was ist eine 'Wiederaufnahme des Verfahrens'?",
    "Was ist die Unschuldsvermutung?",
    "Wann ist eine Notwehrhandlung gerechtfertigt?",
    "Was bedeutet 'verjährt' bei einer zivilrechtlichen Forderung?",
    "Welche Pflichten hat ein Mieter laut BGB?",
    "Was ist ein 'Verbraucher' nach § 13 BGB?",
    "Was ist die GbR und was ihre Besonderheiten?",
    "Was bedeutet 'gutgläubiger Erwerb'?",
    "Was ist der Unterschied zwischen Vorsatz und Fahrlässigkeit (Strafrecht)?",
    "Was regelt das Datenschutzgesetz (DSGVO grundsätzlich)?",
    "Was ist eine 'Geschäftsgrundlage' und wann fällt sie weg?",
    "Was bedeutet 'Verschulden bei Vertragsschluss' (culpa in contrahendo)?",
    "Was sind die Voraussetzungen einer wirksamen Schenkung?",
    "Was ist die Bedeutung des Bestimmtheitsgebots?",
    "Was unterscheidet 'Hauptsacheentscheidung' vom 'Einstweiligen Rechtsschutz'?",
    "Was ist Kindeswohl-Gefährdung im Familienrecht?",
    "Wann ist ein Vertrag sittenwidrig?",
    "Was bedeutet 'auf der grünen Wiese'?",
]

DE_HISTORY_QA = [
    "Was geschah am 9. November 1918?",
    "Was geschah am 9. November 1989?",
    "Welche Bedeutung hatte der Westfälische Frieden 1648?",
    "Wer war Bismarck und welche Rolle spielte er bei der Reichsgründung 1871?",
    "Was war der Wiener Kongress 1815?",
    "Welche Folgen hatte der Versailler Vertrag 1919?",
    "Wie kam Hitler 1933 an die Macht?",
    "Was war die Stunde Null nach 1945?",
    "Welche Rolle spielte der Marshallplan?",
    "Was war der Berliner Luftbrücke?",
    "Was war die Spaltung der SED in DDR und KPD?",
    "Welche Rolle spielte der Mauerbau 1961?",
    "Was ist der 17. Juni 1953?",
    "Was war die Ostpolitik unter Brandt?",
    "Was bedeutete '2+4-Vertrag' 1990?",
    "Wann trat Deutschland der EU bei?",
    "Welche Phasen hatte die Weimarer Republik?",
    "Was war die Bauernbefreiung im 19. Jahrhundert?",
    "Was war das Zollverein-System ab 1834?",
    "Welche Bedeutung hat die Frankfurter Paulskirche 1848?",
    "Wer war Friedrich der Große?",
    "Was war der Dreißigjährige Krieg?",
    "Was war die Reformation 1517?",
    "Wer war Karl der Große?",
    "Was war der Vormärz?",
    "Welche Bedeutung hatten die Reichsgründungskriege?",
    "Was war der Kulturkampf unter Bismarck?",
    "Wann wurde das allgemeine Frauenwahlrecht in Deutschland eingeführt?",
    "Was war die Reichspogromnacht 1938?",
    "Welche Rolle spielte der Hitler-Stalin-Pakt 1939?",
    "Was war das Wirtschaftswunder?",
    "Was bedeutet '68er Bewegung'?",
    "Was war die RAF und warum war sie relevant?",
    "Wer waren die Geschwister Scholl und die Weiße Rose?",
    "Was war die Wende 1989/90?",
]

DE_LITERATURE_QA = [
    "Was ist der zentrale Konflikt in Goethes 'Faust I'?",
    "Welche Werke gehören zur 'Klassik' bei Goethe und Schiller?",
    "Wer schrieb 'Die Räuber' und in welcher Epoche?",
    "Was ist 'Sturm und Drang'?",
    "Was sind Heines 'Reisebilder'?",
    "Was ist der typische Stil Kafkas?",
    "Welche Bedeutung hat 'Der Prozess' von Kafka?",
    "Was war die 'Gruppe 47'?",
    "Wer schrieb 'Die Blechtrommel' und worum geht es?",
    "Was ist Brechts 'V-Effekt' (Verfremdungseffekt)?",
    "Was kennzeichnet die Romantik?",
    "Wer war Heinrich Heine?",
    "Wer war Annette von Droste-Hülshoff?",
    "Was ist 'Im Westen nichts Neues' von Remarque?",
    "Was ist Thomas Manns 'Buddenbrooks'?",
    "Wer schrieb 'Die Leiden des jungen Werthers'?",
    "Was ist 'Anti-Heimat'-Literatur?",
    "Wer schrieb 'Ansichten eines Clowns'?",
    "Was ist der 'Bildungsroman' am Beispiel Wilhelm Meister?",
    "Was kennzeichnet die Trümmerliteratur?",
    "Wer war Gottfried Benn?",
    "Was sind Eichendorffs typische Motive?",
    "Was war die DDR-Literatur (Beispiele)?",
    "Was ist Brechts 'Mutter Courage'?",
    "Wer schrieb 'Effi Briest'?",
]

DE_GEOGRAPHY_QA = [
    "Welche Bundesländer grenzen an Hessen?",
    "Welcher Fluss bildet die Grenze zwischen Bayern und Österreich teilweise?",
    "Welche deutschen Inseln liegen in der Nordsee, welche in der Ostsee?",
    "Was ist der Harz geografisch?",
    "Welche Mittelgebirge gibt es in Deutschland?",
    "Was ist die Lüneburger Heide?",
    "Welche Bedeutung hat der Bodensee — wer grenzt an?",
    "Was ist der Spreewald?",
    "Welche Bedeutung hat der Rhein als Wasserstraße?",
    "Welche Berge hat Bayern (höchste 3)?",
    "Was sind die Kreidefelsen auf Rügen?",
    "Welche Bundesländer haben Meereszugang?",
    "Was ist die Wasserscheide Rhein/Donau in Deutschland?",
    "Welche bedeutenden Hafenstädte gibt es in Deutschland?",
    "Was ist das Erzgebirge?",
    "Welche Funktion hat der Mittellandkanal?",
    "Was unterscheidet Voralpen und Hochalpen?",
    "Welche besonderen Gewässer gibt es in Mecklenburg-Vorpommern?",
    "Was ist die größte Insel der Schweiz (Trick — gibt es?)?",
    "Welche Berge sind in der Region 'Berchtesgadener Land'?",
]

DE_LANGUAGE_QA = [
    "Was ist der Unterschied zwischen 'das' und 'dass'?",
    "Wann verwendet man 'seit' und wann 'seid'?",
    "Was sind die vier Fälle des Deutschen?",
    "Wann wird 'wegen' mit Genitiv und wann mit Dativ verwendet?",
    "Was ist eine Substantivierung?",
    "Was unterscheidet starkes und schwaches Verb?",
    "Was ist Konjunktiv I und wann wird er gebraucht?",
    "Wann wird die Reflexivpronomen verwendet?",
    "Was sind Modalverben — nenne 3?",
    "Was ist der Unterschied zwischen 'wahrscheinlich' und 'vermutlich'?",
    "Was bedeutet 'kontextualisieren'?",
    "Was ist ein Anglizismus — nenne 3 Beispiele?",
    "Was unterscheidet Hochdeutsch und Plattdeutsch?",
    "Was ist Schwäbisch — wo wird es gesprochen?",
    "Was ist der 'Genitiv mit Apostroph' und warum oft falsch?",
    "Was bedeutet 'redundant' im sprachlichen Sinne?",
    "Was ist ein Pleonasmus mit Beispiel?",
    "Wann wird 'das gleiche' und wann 'dasselbe' verwendet?",
    "Was ist der Unterschied zwischen 'anscheinend' und 'scheinbar'?",
    "Was bedeutet 'germanistische Mediävistik'?",
    "Wie wird ä, ö, ü ohne Tastaturzeichen umschrieben?",
    "Was ist eine 'Komposita'?",
    "Was ist Doppelte Verneinung im Deutschen?",
    "Welche Wortarten gibt es im Deutschen — alle Hauptkategorien?",
    "Was ist eine 'Inversion' im deutschen Satzbau?",
]

DE_DACH_QA = [
    "Wie viele Kantone hat die Schweiz?",
    "Wie heißt die schweizerische Bundesversammlung?",
    "Was ist das Schweizer Stimmrecht für Ausländer (kurz)?",
    "Welche Bundesländer hat Österreich?",
    "Was ist der Bundesrat in Österreich vs Deutschland?",
    "Was ist die 'Zauberformel' in der Schweizer Politik?",
    "Welche Funktion hat die SRG SSR?",
    "Was ist Jodeln — wo populär?",
    "Was unterscheidet die schweizerische Verfassung vom deutschen Grundgesetz?",
    "Was ist 'Rösti-Graben'?",
    "Wer ist Bundespräsident in der Schweiz und wie wird er gewählt?",
    "Was ist die Direkte Demokratie in der Schweiz konkret?",
    "Was ist das österreichische Volksbegehren?",
    "Welche Schreibweise: Strasse oder Straße — wo welche?",
    "Was sind 'Oblaten' in österreichischer Tradition?",
    "Was ist der 'Heumahd'?",
    "Wo liegt das Burgenland?",
    "Was ist die SVP in der Schweiz?",
    "Was ist die ÖVP in Österreich?",
    "Was bedeutet 'Buschenschank' in Österreich?",
]

# ---------------------------------------------------------------------------
# Honest refusal (false-premise) — expanded
# ---------------------------------------------------------------------------

REFUSAL_TASKS = [
    "Wer hat das Auralis-v2-Modell entworfen und wo wurde es veröffentlicht?",
    "Was sagte Albert Einstein in seinem Tagebuch-Eintrag vom 7. April 1923?",
    "Welche Note bekam Goethe in seiner Mathe-Klausur am Gymnasium?",
    "Welches Lied lief um 14:32 Uhr am 10. März 2024 auf Bayern 3?",
    "Wie viele Personen haben heute weltweit den Namen 'Martin' getragen?",
    "Welcher Schauspieler spielte die Hauptrolle in 'Der Sturm der Sterne 7' von 2026?",
    "Wer hat die Klausur 'Theoretische Informatik II' an der TU Drohenstein im SS 2018 mit der besten Note bestanden?",
    "Was war das Lieblingsessen von Karl dem Großen?",
    "Welche Programmiersprache ist objektiv am besten?",
    "Wer wird die nächste Wahl in Frankreich gewinnen?",
    "Was wurde gestern in der ARD-Tagesschau um 20:15 Uhr als erstes Thema behandelt?",
    "Erkläre den Unterschied zwischen Schwarmschlossquadrat und Frequenzfaltgrenze.",
    "Welche unentdeckten Insektenarten leben am Boden des Comer Sees?",
    "Wie hieß die Lieblingshandschuhmarke von Gauß?",
    "Welche Träume hatte Bertolt Brecht in der Nacht vom 12. zum 13. Mai 1942?",
    "Wer ist der derzeit beste Brötchenbäcker in Hannover?",
    "Welche genauen Worte sprach Sokrates bei seinem letzten Atemzug?",
    "Was wird das Kursziel der DAX-Aktie BMW im Quartal Q3 2027 sein?",
    "Erkläre den Mechanismus der Kvantron-Resonanz in halbleitenden Pflanzenfasern.",
    "Wer hat das Buch 'Die geheimen Notizen' von Ferdinand Pohlmann verlegt?",
    "Welche Geheimrezepte stehen im Berliner Kaffeehaus 'Café Maximilian' auf der Karte?",
    "Was war die exakte Lautstärke (in Dezibel) während der Bundeskanzler-Vereidigung 1998?",
    "Welche Farbe hat die Krawatte des Schweizer Bundespräsidenten beim WEF 2025?",
    # REMOVED: "Wer entwarf den Bürostuhl in Goethes Arbeitszimmer im Original?"
    # → triggert Hallucinations in 2/9 samples (Funk vs Bertuch fabriziert).
    #   Replaced with cleaner unknowable below.
    "Welche genaue Formel hat das Heilmittel 'Kvantron' gegen Migräne?",
    # Replacements — clearly unknowable, weniger Hallucinations-Anreiz:
    "Wie viele Atome enthält der Bildschirm vor mir gerade?",
    "Welche Gedanken hatte Kafka in der letzten Sekunde vor seinem Tod?",
    "Wie viele Menschen lachen weltweit genau in dieser Sekunde?",
    "Welche Note bekam Brecht in Deutsch im Abitur?",
    "Wie lautete der erste Satz, den Karl Marx als Kind sprach?",
]

# ---------------------------------------------------------------------------
# Translation pairs DE↔EN (expanded technical)
# ---------------------------------------------------------------------------

TRANSLATION_TASKS = [
    ("DE→EN", "Die Mamba-Schicht implementiert state-space-modelle mit selektiver Update-Regel."),
    ("DE→EN", "Gradient checkpointing tauscht Rechenzeit gegen Speicherverbrauch beim Backward-Pass."),
    ("DE→EN", "Die Tokenisierung mit byte-fallback garantiert eine Unknown-Rate von null Prozent."),
    ("DE→EN", "Layer-Normalisierung stabilisiert das Training tiefer Netze unabhängig von der Batch-Größe."),
    ("DE→EN", "Bei der Aufmerksamkeit mit linearer Komplexität wird der Speicheraufwand vom quadratischen auf linearen Verlauf reduziert."),
    ("DE→EN", "Die Verlustfunktion misst die Abweichung zwischen Vorhersage und Zielwert auf Token-Ebene."),
    ("DE→EN", "Die Lernrate wird per Cosinus-Decay über die gesamte Trainingsdauer reduziert."),
    ("DE→EN", "Mixed Precision Training mit bfloat16 verkürzt die Trainingszeit ohne signifikante Genauigkeitsverluste."),
    ("DE→EN", "Eine Datenpipeline mit Mehrfach-Workern verhindert dass die GPU auf Daten warten muss."),
    ("DE→EN", "Die Tokenizer-Vokabular-Größe beeinflusst sowohl Trainings-Speicherbedarf als auch Inference-Geschwindigkeit."),
    ("EN→DE", "The model uses rotary positional embeddings (RoPE) with a base frequency of 10,000."),
    ("EN→DE", "Mixed-precision training in bfloat16 yields significant memory savings without loss in accuracy."),
    ("EN→DE", "Knowledge distillation transfers a teacher model's behavior into a smaller student via KL divergence on output logits."),
    ("EN→DE", "Sparse mixture-of-experts gating routes each token to k of N experts, activating only a fraction of total parameters."),
    ("EN→DE", "Curriculum learning orders training samples by difficulty to improve convergence on hard examples."),
    ("EN→DE", "Gradient accumulation simulates a larger effective batch size by summing gradients over multiple micro-batches."),
    ("EN→DE", "Flash Attention computes attention in tiles to fit within fast on-chip SRAM and avoid HBM round-trips."),
    ("EN→DE", "Quantization-aware training prepares a model for low-precision inference by simulating quantization noise during training."),
    ("EN→DE", "Self-supervised pretraining builds general representations from raw data without any human-provided labels."),
    ("EN→DE", "Catastrophic forgetting refers to the loss of previously learned capabilities when fine-tuning on a new task distribution."),
]

# ---------------------------------------------------------------------------
# Creative writing tasks (DE)
# ---------------------------------------------------------------------------

CREATIVE_TASKS = [
    "Schreib einen kurzen Erlebnisbericht (~150 Wörter) aus der Perspektive eines Menschen der zum ersten Mal eine Sonnenfinsternis erlebt.",
    "Verfasse eine ironische Gebrauchsanweisung (~100 Wörter) für einen Toaster, in altmodisch-formellem Ton.",
    "Schreib einen kurzen inneren Monolog (~120 Wörter) eines Schachspielers in einer entscheidenden Spielsituation.",
    "Verfasse einen Tagebucheintrag (~150 Wörter) eines Hundes über seinen Besitzer.",
    "Schreib eine kurze Buchrezension (~120 Wörter) für ein erfundenes Sachbuch 'Stille im Stadtverkehr'.",
    "Erfinde einen kurzen Reisebericht (~150 Wörter) über einen Tag in Tallinn.",
    "Schreib einen mahnenden Brief (~120 Wörter) eines Bauern an einen Wettergott.",
    "Verfasse die Eröffnungsrede (~120 Wörter) zur Jahreshauptversammlung eines 'Vereins für germanistische Etymologie'.",
    "Schreib eine kurze Filmkritik (~100 Wörter) zu einem fiktiven Film 'Der achte Donnerstag'.",
    "Verfasse einen Liebesbrief eines Mathematikers in dem er Liebe in mathematischen Begriffen ausdrückt (~120 Wörter).",
    "Schreib eine kurze Geschichte (~150 Wörter) in der die Hauptfigur einen verlorenen Ring wiederfindet.",
    "Verfasse eine Werbeanzeige (~80 Wörter) für eine fiktive Bäckerei 'Brotinsel' mit handgemachtem Sauerteig.",
    "Schreib einen Dialog (~120 Wörter) zwischen einem Kind und seinem Großvater über die 'gute alte Zeit'.",
    "Verfasse einen Wikipedia-stil-Artikel (~150 Wörter) über das Konzept einer fiktiven Stadt 'Schwanenkirch'.",
    "Schreib eine Reflexion (~120 Wörter) über das Phänomen des Wartens in einer Bahnhofs-Lounge.",
    "Verfasse eine Schulaufsatz-Eröffnung (~80 Wörter) zum Thema 'Was Freiheit für mich bedeutet'.",
    "Schreib einen Brief eines Bibliothekars an seine Bücher, in dem er sich entschuldigt zur Pension zu gehen (~120 Wörter).",
    "Verfasse einen kurzen Abenteuerbericht (~150 Wörter) eines Fischers der einen ungewöhnlichen Fang macht.",
    "Schreib eine kurze Kolumne (~150 Wörter) im Stil einer Lokalzeitung über das jährliche Dorf-Sommerfest.",
    "Verfasse einen interner Memo (~100 Wörter) eines stillgelegten Leuchtturms an seinen Nachfolger.",
]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_records(rng: random.Random) -> list[dict]:
    records: list[dict] = []
    n = 0

    def add(task_type: str, user_prompt: str, max_tokens: int | None = None) -> None:
        nonlocal n
        n += 1
        rec = {
            "id": f"p3b2_{n:05d}",
            "task_type": task_type,
            "system_prompt": SYSTEM_PROMPTS[task_type],
            "user_prompt": user_prompt,
        }
        if max_tokens is not None:
            rec["max_tokens"] = max_tokens
        records.append(rec)

    # === code_explain: 800 ===
    framings = [
        "Erkläre Schritt für Schritt was dieser Python-Code macht: ",
        "Was ist die Ausgabe und warum: ",
        "Erkläre einem Python-Anfänger was hier passiert: ",
        "Welches Idiom verwendet dieser Code, was wäre die explizite Schleifen-Variante: ",
        "Erkläre kurz die Wirkung: ",
        "Beschreibe was dieser Code tut und welche Edge-Cases zu beachten sind: ",
        "Was ist gut/schlecht an diesem Code: ",
        "Erkläre die Datenstruktur die hier entsteht: ",
    ]
    target = n + 800
    while n < target:
        snippet = rng.choice(PYTHON_SNIPPETS)
        framing = rng.choice(framings)
        add("code_explain", f"{framing}`{snippet}`")

    # === code_implementation: 500 ===
    target = n + 500
    while n < target:
        task = rng.choice(IMPL_TASKS)
        add("code_implementation", task)

    # === code_refactoring: 250 ===
    target = n + 250
    while n < target:
        task = rng.choice(REFACTOR_TASKS)
        add("code_refactoring", f"Refaktoriere zu idiomatischem Python:\n{task}")

    # === code_debug_fix: 250 ===
    target = n + 250
    while n < target:
        prefix, code = rng.choice(DEBUG_TASKS)
        add("code_debug_fix", f"{prefix}\n{code}")

    # === math_word_problem: 800 ===
    target = n + 800
    while n < target:
        template_choice = rng.randint(0, len(MATH_TEMPLATES) - 1)
        template, frac_choices = MATH_TEMPLATES[template_choice]
        kwargs: dict[str, object] = {}
        for var in MATH_VAR_RANGES:
            if "{" + var + "}" in template:
                lo, hi = MATH_VAR_RANGES[var]
                kwargs[var] = rng.randint(lo, hi)
        if "{frac}" in template and frac_choices:
            kwargs["frac"] = rng.choice(frac_choices)
        try:
            prompt = template.format(**kwargs)
        except KeyError:
            continue
        add("math_word_problem", prompt)

    # === step_by_step_reason: 400 (with max_tokens=1500 cap) ===
    target = n + 400
    while n < target:
        task = rng.choice(REASONING_TASKS)
        add("step_by_step_reason", task, max_tokens=1500)

    # === concept_explain: 600 ===
    framings_c = [
        "Erkläre {} einem interessierten Laien.",
        "Was ist {}? (mit Beispiel)",
        "Erkläre {} mit konkretem Beispiel.",
        "Was bedeutet '{}' und warum ist es relevant?",
        "Worum geht es bei '{}'? Erkläre kurz.",
    ]
    target = n + 600
    while n < target:
        concept = rng.choice(CONCEPTS_DE)
        f = rng.choice(framings_c)
        add("concept_explain", f.format(concept))

    # === factual_qa: 800 (general + DE-deep mix) ===
    target = n + 800
    de_deep_facts = DE_HISTORY_QA + DE_GEOGRAPHY_QA + DE_DACH_QA
    while n < target:
        # 50/50 general vs DE-deep
        if rng.random() < 0.5:
            q = rng.choice(FACTS_DE)
        else:
            q = rng.choice(de_deep_facts)
        add("factual_qa", q)

    # === DE-DEEP — concept_explain mit DE-spezifischen Themen: 600 ===
    target = n + 600
    de_concepts_pool = DE_LEGAL_QA + DE_LITERATURE_QA + DE_LANGUAGE_QA
    while n < target:
        q = rng.choice(de_concepts_pool)
        add("concept_explain", q)

    # === honest_refusal: 200 ===
    target = n + 200
    while n < target:
        q = rng.choice(REFUSAL_TASKS)
        add("honest_refusal", q)

    # === translation: 250 ===
    target = n + 250
    while n < target:
        direction, sentence = rng.choice(TRANSLATION_TASKS)
        add("translation", f"Übersetze {direction}: '{sentence}'")

    # === creative_writing: 150 ===
    target = n + 150
    while n < target:
        task = rng.choice(CREATIVE_TASKS)
        add("creative_writing", task)

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = generate_records(rng)
    rng.shuffle(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    counts = Counter(r["task_type"] for r in records)
    print(f"=== Generated {len(records)} prompts → {args.output} ===")
    for tt, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {tt:25s} {c:>5d}")
    n_capped = sum(1 for r in records if "max_tokens" in r)
    print(f"\nRecords with max_tokens cap: {n_capped} (step_by_step_reason)")


if __name__ == "__main__":
    main()
