"""Generate 20K diverse German instruction prompts for Auralis V2 Phase-3 SFT.

Produces JSONL compatible with deepseek_v4_client.py for response generation.
Massively expanded topic/template pools for maximum diversity at 20K scale.

Category distribution:
    code_explain           2000
    code_implementation    1500
    code_refactoring        800
    code_debug_fix          700
    math_word_problem      2000
    step_by_step_reason    1500
    concept_explain        2500
    factual_qa             2500
    de_deep_knowledge      2000
    honest_refusal          800
    translation             800
    creative_writing        800
    smalltalk              1000
    anleitung              1100
    ─────────────────────────
    TOTAL                 20000

Usage:
    python scripts/data/synth/generate_dataflow_instruct_de.py \\
        --output raw/sft/synth/inputs/dataflow_instruct_de_20k.jsonl \\
        --seed 44
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# System prompts
# ═══════════════════════════════════════════════════════════════════════════

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
        "Antwort."
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
        "2. ERLAUBT: 1-2 Sätze Kontext WARUM die Frage problematisch ist\n"
        "3. VERBOTEN: alternative spezifische Antworten mit 'vermutlich', "
        "'wahrscheinlich', 'könnte gewesen sein'\n"
        "4. VERBOTEN: spezifische Namen, Daten, Zahlen die du nicht "
        "zuverlässig kennst\n"
        "5. Wenn unsicher — entscheide IMMER für 'weiß ich nicht'\n\n"
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
    "smalltalk": (
        "Du bist Auralis, ein freundlicher deutscher KI-Assistent. "
        "Antworte natürlich, warmherzig, kurz. Wie ein netter Gesprächspartner. "
        "Keine Floskeln wie 'Natürlich!' oder 'Gerne helfe ich'. "
        "Einfach natürlich antworten."
    ),
    "anleitung": (
        "Du bist Auralis. Erkläre Schritt für Schritt auf Deutsch wie man "
        "etwas macht. Klar nummerierte Schritte, praxisnah, ohne "
        "überflüssige Einleitungen."
    ),
}

# ═══════════════════════════════════════════════════════════════════════════
# CODE: Snippets (expanded ~120)
# ═══════════════════════════════════════════════════════════════════════════

PYTHON_SNIPPETS = [
    # Comprehensions
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
    "[row[col] for row in matrix for col in range(len(row))]",
    "{word.lower() for line in lines for word in line.split()}",
    "[[0]*cols for _ in range(rows)]",
    "[f(x) for x in xs if (y := g(x)) > threshold]",
    # Decorators
    "@functools.lru_cache(maxsize=128)\ndef fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
    "@dataclass(frozen=True)\nclass Point:\n    x: int\n    y: int",
    "@contextmanager\ndef tmpdir():\n    d = tempfile.mkdtemp()\n    try:\n        yield d\n    finally:\n        shutil.rmtree(d)",
    "@functools.singledispatch\ndef serialize(obj):\n    raise TypeError",
    "@property\ndef name(self):\n    return self._name",
    "@classmethod\ndef from_json(cls, data):\n    return cls(**json.loads(data))",
    "@staticmethod\ndef validate(x):\n    if x < 0: raise ValueError",
    "@functools.wraps(fn)\ndef wrapper(*a, **kw): return fn(*a, **kw)",
    # Context managers
    "with open('data.json') as f, open('out.json', 'w') as g:\n    json.dump(json.load(f), g, indent=2)",
    "with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:\n    results = list(pool.map(fetch, urls))",
    "with sqlite3.connect(db) as conn:\n    conn.execute('INSERT INTO logs VALUES (?, ?)', (ts, msg))",
    "with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:\n    tmp.write(data)",
    "with contextlib.suppress(FileNotFoundError):\n    os.remove(path)",
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
    "list(itertools.islice(itertools.count(10, 3), 5))",
    "dict(itertools.zip_longest(keys, values, fillvalue=0))",
    "list(itertools.takewhile(lambda x: x < 5, sorted_data))",
    "list(itertools.dropwhile(lambda x: x < 5, sorted_data))",
    "list(itertools.starmap(pow, [(2,3), (3,2), (10,3)]))",
    # Async
    "async def fetch_all(urls):\n    return await asyncio.gather(*[fetch(u) for u in urls])",
    "async with aiohttp.ClientSession() as s:\n    async with s.get(url) as r:\n        data = await r.json()",
    "asyncio.run(asyncio.wait_for(task, timeout=10.0))",
    "async for line in stream:\n    if line.startswith('ERROR'):\n        break",
    "async with asyncio.TaskGroup() as tg:\n    for url in urls:\n        tg.create_task(fetch(url))",
    "loop = asyncio.get_event_loop()\nresult = loop.run_until_complete(coro)",
    # Error handling
    "try:\n    n = int(s)\nexcept ValueError:\n    n = 0",
    "try:\n    risky()\nexcept (TypeError, ValueError) as e:\n    log.error('failed: %s', e)",
    "try:\n    with lock:\n        critical()\nexcept TimeoutError:\n    fallback()",
    "try:\n    result = compute()\nexcept Exception:\n    result = None\nelse:\n    save(result)\nfinally:\n    cleanup()",
    # Slicing / unpacking
    "matrix[::-1]",
    "s[::2]",
    "lst[len(lst)//2:]",
    "first, *middle, last = items",
    "a, b = b, a",
    "(x, y), z = (1, 2), 3",
    "head, *_ = sorted(items, reverse=True)",
    # Dict tricks
    "d.setdefault('items', []).append(x)",
    "{**a, **b, 'extra': 1}",
    "d | {'new_key': 'new_value'}",
    "dict(zip(keys, values))",
    "max(d.items(), key=lambda kv: kv[1])",
    "dict(sorted(d.items(), key=lambda x: x[1], reverse=True))",
    "{k: v for k, v in sorted(d.items())}",
    "d.pop('key', None)",
    # Functional
    "list(map(int, '1 2 3 4'.split()))",
    "list(filter(lambda x: x.startswith('a'), words))",
    "sorted(items, key=lambda x: (-x.priority, x.id))",
    "sorted(students, key=operator.attrgetter('grade', 'name'))",
    # Pattern matching (3.10+)
    "match cmd:\n    case 'start': run()\n    case 'stop': halt()\n    case _: print('unknown')",
    "match point:\n    case (0, 0): print('origin')\n    case (x, 0): print(f'on x at {x}')\n    case (_, y): print(f'y={y}')",
    "match shape:\n    case Circle(radius=r): area = math.pi * r**2\n    case Rect(w=w, h=h): area = w*h",
    "match response.status_code:\n    case 200: process(response.json())\n    case 404: log.warning('not found')\n    case _: raise HTTPError(response.status_code)",
    # Pathlib
    "Path('logs').glob('**/*.log')",
    "Path(__file__).resolve().parent.parent / 'data'",
    "Path('config.yaml').read_text(encoding='utf-8')",
    "list(Path('.').rglob('*.py'))",
    "Path('output').mkdir(parents=True, exist_ok=True)",
    # Numpy
    "arr[arr > arr.mean()]",
    "np.where(grid == 0, -1, grid)",
    "(x - x.mean(0)) / x.std(0)",
    "np.concatenate([a.flatten(), b.flatten()])",
    "np.argsort(scores)[::-1][:k]",
    "np.einsum('ij,jk->ik', A, B)",
    "np.unique(arr, return_counts=True)",
    "np.clip(values, 0, 1)",
    "np.linspace(0, 2*np.pi, 100)",
    # String formatting
    "f'{value:>10.2f} ({pct:.1%})'",
    "':'.join(f'{b:02x}' for b in mac)",
    "f'{name=!r}'",
    "textwrap.dedent('''\\n    Hello\\n    World\\n''').strip()",
    "f'{dt:%Y-%m-%d %H:%M}'",
    "f'{n:_}'",
    # Generators
    "def chunks(lst, n):\n    for i in range(0, len(lst), n):\n        yield lst[i:i+n]",
    "def fibonacci():\n    a, b = 0, 1\n    while True:\n        yield a\n        a, b = b, a+b",
    "(x for x in range(1000000) if is_prime(x))",
    "def powers_of_two():\n    n = 1\n    while True:\n        yield n\n        n *= 2",
    # Class snippets
    "@dataclass\nclass Config:\n    lr: float = 1e-4\n    batch_size: int = 32\n    epochs: int = 10",
    "class Singleton:\n    _instance = None\n    def __new__(cls):\n        if cls._instance is None:\n            cls._instance = super().__new__(cls)\n        return cls._instance",
    "class Vector:\n    def __init__(self, x, y): self.x, self.y = x, y\n    def __add__(self, o): return Vector(self.x+o.x, self.y+o.y)\n    def __repr__(self): return f'Vector({self.x}, {self.y})'",
    "class ContextTimer:\n    def __enter__(self): self.start = time.perf_counter(); return self\n    def __exit__(self, *_): self.elapsed = time.perf_counter() - self.start",
    # Type hints
    "def greet(name: str | None = None) -> str:\n    return f'Hello {name or \"World\"}'",
    "T = TypeVar('T')\ndef first_or_default(items: list[T], default: T) -> T:\n    return items[0] if items else default",
    "Callback = Callable[[str, int], bool]",
    # Walrus operator
    "while chunk := f.read(8192):\n    process(chunk)",
    "if (m := re.match(r'(\\d+)', text)) is not None:\n    print(int(m.group(1)))",
    # Regex
    "re.findall(r'\\b[A-Z][a-z]+\\b', text)",
    "re.sub(r'\\s+', ' ', text).strip()",
    "re.split(r'[;,]\\s*', csv_line)",
    # OS/sys
    "os.environ.get('API_KEY', 'default')",
    "sys.argv[1:]",
    "subprocess.run(['git', 'status'], capture_output=True, text=True)",
    # JSON/YAML
    "json.loads(Path('config.json').read_text())",
    "json.dumps(data, indent=2, ensure_ascii=False, default=str)",
]

# ═══════════════════════════════════════════════════════════════════════════
# CODE: Implementation tasks (expanded ~80)
# ═══════════════════════════════════════════════════════════════════════════

IMPL_TASKS = [
    "Schreib eine Funktion safe_divide(a, b) die für b=0 None zurückgibt.",
    "Schreib chunked(items, n) die items in n-große Listen aufteilt.",
    "Schreib deduplicate_keep_order(seq) die Duplikate entfernt aber Ordnung erhält.",
    "Schreib is_palindrome(s) case-insensitive, ignoriert whitespace + punctuation.",
    "Schreib flatten(nested) die beliebig tief verschachtelte Listen zu flach macht.",
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
    "Schreib dict_diff(old, new) -> dict mit added/removed/changed keys.",
    "Schreib async fetch_concurrent(urls, max_concurrency=10) mit asyncio.Semaphore.",
    "Schreib natural_sort_key(s) für sorted(): file1 < file2 < file10.",
    "Schreib Stopwatch als Context-Manager mit elapsed-Property.",
    "Schreib checksum_file(path, algo='sha256') chunked-Hashing für large files.",
    "Schreib ensure_unique_path(target) — bei Existenz _1, _2 anhängen.",
    "Schreib partition(pred, iterable) -> (truthy_list, falsy_list).",
    "Schreib group_by(items, key) -> dict[key, list[items]].",
    "Schreib parse_size('1.5 GB') -> Bytes.",
    "Schreib safe_get(d, *keys, default=None) für nested dict access.",
    "Schreib timer Decorator der die Laufzeit per logger logged.",
    "Schreib RateLimiter(calls_per_sec) — sliding window rate limit.",
    "Schreib EventEmitter mit on(event, handler), emit(event, *args).",
    "Schreib JsonStreamer(path) — read jsonl lazy, write_one(record) appends atomically.",
    "Schreib running_average(window) als generator-coroutine die letzten N werte mittelt.",
    "Schreib trie data structure mit insert(word), search(word), starts_with(prefix).",
    "Schreib Caesar-Cipher encode(text, shift) und decode — preserve case + non-alpha.",
    "Schreib graph_shortest_path(graph, start, end) BFS auf unweighted graph.",
    "Schreib Reservoir-Sampling sample_k(stream, k) für unbekannte Stream-Länge.",
    "Schreib memoize Decorator der Argumente cached (dict-basiert).",
    "Schreib deep_merge(d1, d2) für nested dicts — rekursiv mergen.",
    "Schreib throttle(fn, interval_seconds) Decorator.",
    "Schreib csv_to_dicts(path) -> list[dict] ohne csv-Modul.",
    "Schreib binary_search(arr, target) -> index oder -1.",
    "Schreib matrix_multiply(A, B) ohne numpy.",
    "Schreib transpose(matrix) ohne numpy.",
    "Schreib is_balanced_brackets(s) für ()[]{}.",
    "Schreib roman_to_int(s) für römische Zahlen.",
    "Schreib int_to_roman(n) für Dezimal → römisch.",
    "Schreib gcd(a, b) und lcm(a, b) ohne math-Modul.",
    "Schreib is_anagram(s1, s2) -> bool.",
    "Schreib longest_common_prefix(strings) -> str.",
    "Schreib spiral_order(matrix) für spiral-Traversierung.",
    "Schreib compress_string('aaabbc') -> 'a3b2c1'.",
    "Schreib decompress_string('a3b2c1') -> 'aaabbc'.",
    "Schreib frequency_sort(s) sortiert chars nach Häufigkeit.",
    "Schreib rotate_matrix_90(matrix) in-place.",
    "Schreib find_missing_number(arr) in [0..n] mit einer fehlenden Zahl.",
    "Schreib two_sum(nums, target) -> (i, j) indices.",
    "Schreib max_subarray_sum(arr) (Kadane's Algorithmus).",
    "Schreib stack_with_min() der min() in O(1) unterstützt.",
    "Schreib queue_from_stacks() — Queue mit 2 Stacks.",
    "Schreib linked_list_reverse(head) iterativ.",
    "Schreib detect_cycle(head) in einer Linked-List (Floyd).",
    "Schreib inorder_traversal(root) iterativ mit Stack.",
    "Schreib level_order_traversal(root) als BFS.",
    "Schreib is_valid_bst(root) Validierung.",
    "Schreib dijkstra(graph, start) für shortest paths.",
    "Schreib topological_sort(graph) mit DFS.",
    "Schreib union_find mit find() und union() (path compression).",
    "Schreib counting_sort(arr, max_val) -> sorted array.",
    "Schreib quickselect(arr, k) für k-kleinstes Element.",
    "Schreib interval_merge(intervals) für überlappende Intervalle.",
    "Schreib Fenwick-Tree (BIT) mit update() und prefix_sum().",
    "Schreib word_break(s, word_dict) -> bool (DP).",
    "Schreib coin_change(coins, amount) -> minimale Münzanzahl (DP).",
    "Schreib knapsack_01(weights, values, capacity) -> max value.",
    "Schreib longest_increasing_subsequence(arr) -> length.",
    "Schreib edit_distance(s1, s2) mit DP-Tabelle.",
    "Schreib producer_consumer() mit threading.Queue.",
    "Schreib parallel_map(fn, items, workers=4) mit multiprocessing.",
    "Schreib rate_limited_api_caller(url, rps=10) mit asyncio.",
    "Schreib simple_http_server(port=8080) mit http.server.",
    "Schreib file_watcher(directory) der Änderungen detected.",
]

# ═══════════════════════════════════════════════════════════════════════════
# CODE: Refactoring tasks (expanded ~35)
# ═══════════════════════════════════════════════════════════════════════════

REFACTOR_TASKS = [
    "```python\nresult = []\nfor x in items:\n    if x.active:\n        result.append(x.id)\n```",
    "```python\nout = ''\nfor i, ch in enumerate(s):\n    if i % 2 == 0:\n        out = out + ch\n```",
    "```python\nif x is None:\n    y = default\nelse:\n    y = x\n```",
    "```python\ncounts = {}\nfor word in words:\n    if word in counts:\n        counts[word] = counts[word] + 1\n    else:\n        counts[word] = 1\n```",
    "```python\nmaximum = items[0]\nfor item in items:\n    if item > maximum:\n        maximum = item\n```",
    "```python\nresult = []\nfor i in range(len(a)):\n    result.append(a[i] + b[i])\n```",
    "```python\ntry:\n    f = open(path)\n    data = f.read()\n    f.close()\nexcept:\n    data = ''\n```",
    "```python\ndef has_negative(nums):\n    found = False\n    for n in nums:\n        if n < 0:\n            found = True\n    return found\n```",
    "```python\nif name == 'admin' or name == 'root' or name == 'superuser':\n    grant_access()\n```",
    "```python\nfor i in range(len(items)):\n    print(f'{i}: {items[i]}')\n```",
    "```python\ntotal = 0\nfor x in nums:\n    total = total + x\nreturn total\n```",
    "```python\nresult = sorted([x for x in items])[::-1][:5]\n```",
    "```python\nif type(x) == int:\n    process_int(x)\nelif type(x) == str:\n    process_str(x)\n```",
    "```python\ndef sum_squares(n):\n    s = 0\n    for i in range(1, n+1):\n        s = s + i*i\n    return s\n```",
    "```python\nlines = []\nf = open(path)\nfor line in f.readlines():\n    lines.append(line.strip())\nf.close()\n```",
    "```python\nif len(lst) == 0:\n    return True\nelse:\n    return False\n```",
    "```python\nd = {}\nfor item in items:\n    k = item.category\n    if k not in d:\n        d[k] = []\n    d[k].append(item)\n```",
    "```python\nnew_list = []\nfor item in old_list:\n    new_list.append(item.upper())\n```",
    "```python\nif condition == True:\n    do_something()\n```",
    "```python\nresult = []\nfor x in data:\n    if x not in result:\n        result.append(x)\n```",
    "```python\ntry:\n    value = int(text)\n    return value\nexcept:\n    return -1\n```",
    "```python\ndef average(nums):\n    total = 0\n    count = 0\n    for n in nums:\n        total += n\n        count += 1\n    return total / count\n```",
    "```python\nflags = []\nfor item in items:\n    if item.status == 'active':\n        flags.append(True)\n    else:\n        flags.append(False)\n```",
    "```python\nkeys = []\nvalues = []\nfor k, v in data.items():\n    keys.append(k)\n    values.append(v)\n```",
    "```python\nresult = ''\nfor word in words:\n    result = result + word + ' '\nresult = result.strip()\n```",
    "```python\ni = 0\nwhile i < len(lst):\n    process(lst[i])\n    i = i + 1\n```",
    "```python\nif x > 0:\n    sign = 'positive'\nelif x < 0:\n    sign = 'negative'\nelse:\n    sign = 'zero'\n```",
    "```python\nresult = []\nfor sublist in nested:\n    for item in sublist:\n        result.append(item)\n```",
    "```python\noutput = {}\nfor key in dict1:\n    output[key] = dict1[key]\nfor key in dict2:\n    output[key] = dict2[key]\n```",
    "```python\ndef get_extension(filename):\n    parts = filename.split('.')\n    return parts[len(parts) - 1]\n```",
    "```python\nif not (x > 10):\n    handle_small()\n```",
    "```python\nresult = list()\nfor i in range(0, len(data), 1):\n    if data[i] != None:\n        result.append(data[i])\n```",
    "```python\nif isinstance(x, int) == True and x >= 0 == True:\n    process(x)\n```",
    "```python\ndef double_all(nums):\n    result = []\n    for n in nums:\n        result.append(n * 2)\n    return result\n```",
    "```python\ntext = text.replace('a', 'x')\ntext = text.replace('b', 'y')\ntext = text.replace('c', 'z')\n```",
]

# ═══════════════════════════════════════════════════════════════════════════
# CODE: Debug tasks (expanded ~30)
# ═══════════════════════════════════════════════════════════════════════════

DEBUG_TASKS = [
    ("Find the bug:", "```python\ndef avg(nums): return sum(nums) / len(nums) - 1\n```"),
    ("Find the bug:", "```python\ndef factorial(n):\n    result = 0\n    for i in range(1, n+1):\n        result *= i\n    return result\n```"),
    ("Find the bug:", "```python\ndef binary_search(arr, target):\n    lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] < target: lo = mid\n        elif arr[mid] > target: hi = mid\n        else: return mid\n    return -1\n```"),
    ("Find the bug:", "```python\ndef merge_dicts(*dicts):\n    result = {}\n    for d in dicts:\n        result.update(d)\n```"),
    ("Find the bug:", "```python\ndef strip_extension(filename):\n    return filename.split('.')[0]\n```"),
    ("Find the bug:", "```python\ndef remove_first_match(lst, x):\n    for item in lst:\n        if item == x:\n            lst.remove(item)\n            return\n```"),
    ("Find the bug:", "```python\ndef get_user(uid):\n    cursor.execute('SELECT * FROM users WHERE id=' + str(uid))\n    return cursor.fetchone()\n```"),
    ("Find the bug:", "```python\nclass Counter:\n    count = 0\n    def increment(self):\n        Counter.count += 1\n```"),
    ("Find the bug:", "```python\ndef tax(amount, rate=None):\n    if rate is None:\n        rate = []\n    rate.append(amount * 0.19)\n    return rate\n```"),
    ("Find the bug:", "```python\ndef batch_iter(items, n):\n    for i in range(0, len(items), n):\n        yield items[i:i+n+1]\n```"),
    ("Find the bug:", "```python\ndef reverse_words(s):\n    words = s.split()\n    return ' '.join(words.reverse())\n```"),
    ("Find the bug:", "```python\ndef get_or_create(d, key):\n    if not d.get(key):\n        d[key] = []\n    return d[key]\n```"),
    ("Find the bug:", "```python\nfor i, item in enumerate(items):\n    if item.expired:\n        del items[i]\n```"),
    ("Find the bug:", "```python\nimport threading\ncounter = 0\ndef inc():\n    global counter\n    counter += 1\n```"),
    ("Find the bug:", "```python\ndef parse_json(text):\n    try:\n        return json.loads(text), True\n    except json.JSONDecodeError:\n        return None\n```"),
    ("Find the bug:", "```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n# Stack overflow bei n=1000\n```"),
    ("Find the bug:", "```python\ndef flatten(lst):\n    result = []\n    for item in lst:\n        if type(item) == list:\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result\n# tuple/set werden nicht geflattened\n```"),
    ("Find the bug:", "```python\nclass Cache:\n    def __init__(self):\n        self.data = {}\n    def get(self, key):\n        return self.data[key]\n# KeyError wenn nicht vorhanden\n```"),
    ("Find the bug:", "```python\ndef safe_div(a, b):\n    if b == 0:\n        return float('inf')\n    return a / b\n# negative Zahlen → -inf?\n```"),
    ("Find the bug:", "```python\ndef read_config(path='config.json'):\n    with open(path) as f:\n        return json.load(f)\n# Kein Error-Handling, kein encoding\n```"),
    ("Find the bug:", "```python\ndef unique_sorted(items):\n    return sorted(list(set(items)))\n# Stabil? Reihenfolge gleicher Elemente?\n```"),
    ("Find the bug:", "```python\nresults = [None] * len(urls)\nasync def fetch(i, url):\n    results[i] = await get(url)\nawait asyncio.gather(*[fetch(i, u) for i, u in enumerate(urls)])\n# results als closure — thread-safe?\n```"),
    ("Find the bug:", "```python\ndef to_celsius(f):\n    return f - 32 * 5 / 9\n```"),
    ("Find the bug:", "```python\ndef is_even(n):\n    return n / 2 == 0\n```"),
    ("Find the bug:", "```python\ndef max_profit(prices):\n    min_price = prices[0]\n    max_profit = 0\n    for price in prices:\n        min_price = min(min_price, price)\n        max_profit = max(max_profit, price - min_price)\n    return max_profit\n# Was wenn prices leer?\n```"),
    ("Find the bug:", "```python\ndef count_vowels(s):\n    return len([c for c in s if c in 'aeiou'])\n# Großbuchstaben werden ignoriert\n```"),
    ("Find the bug:", "```python\nimport os\ndef list_files(directory):\n    return os.listdir(directory)\n# Gibt auch Verzeichnisse zurück\n```"),
    ("Find the bug:", "```python\ndef power(base, exp):\n    result = 1\n    for _ in range(exp):\n        result *= base\n    return result\n# Negative Exponenten?\n```"),
    ("Find the bug:", "```python\ndef clean_text(text):\n    return text.strip().lower().replace('  ', ' ')\n# Nur ein Doppelspace wird ersetzt\n```"),
    ("Find the bug:", "```python\ndef median(lst):\n    lst.sort()\n    mid = len(lst) // 2\n    return lst[mid]\n# Gerade Anzahl? Leere Liste? Mutiert input!\n```"),
]

# ═══════════════════════════════════════════════════════════════════════════
# MATH: Templates + variables (expanded ~40)
# ═══════════════════════════════════════════════════════════════════════════

MATH_TEMPLATES = [
    ("Anna hat {a} Tüten mit je {b} Bonbons. Sie gibt {frac} aller Bonbons an ihre Schwester. Wie viele behält sie?", ["1/3", "1/4", "1/2", "2/5", "3/7"]),
    ("Ein Zug fährt {h1}h mit {v1} km/h, dann {h2}h mit {v2} km/h. Wie viele km insgesamt?", None),
    ("Eine Pizza kostet {p}€, ein Getränk {g}€. Eine Familie bestellt {n_p} Pizzen und {n_g} Getränke. Wie viel kostet das gesamt?", None),
    ("In einer Klasse mit {n} Schülern sind {pct}% Mädchen. Wie viele Jungen sind in der Klasse?", None),
    ("Ein Auto verbraucht im Schnitt {v} Liter pro 100 km. Wie viel Benzin braucht es für {d} km?", None),
    ("Ein Kapital von {k}€ wird mit {r}% jährlich verzinst. Wie viel Zinsen nach {y} Jahren (einfache Verzinsung)?", None),
    ("Eine Wand ist {w} m breit und {h_wall} m hoch. Eine Tapetenrolle deckt {f} m². Wie viele Rollen (aufrunden)?", None),
    ("Ein Rezept für {orig} Personen braucht {x} g Mehl. Wie viel Mehl für {target} Personen?", None),
    ("Auf einer Karte 1:{scale} sind zwei Städte {cm} cm entfernt. Wie weit in Wirklichkeit?", None),
    ("Ein Tank fasst {tank_v} Liter und ist {tank_p}% gefüllt. Pro Stunde laufen {l} Liter aus. Wann leer?", None),
    ("Eine Aktie kostet {pi}€, steigt um {p1}%, fällt dann um {p2}%. Wie viel wert?", None),
    ("Ein Würfel hat Kantenlänge {a_cube} cm. Volumen und Oberfläche?", None),
    ("Ein Kreis hat Radius {r_circle} cm. Umfang und Fläche (2 Dezimalen)?", None),
    ("Eine Maschine produziert {n_parts} Teile/h. Wie viele in 8h-Schicht? Und in Woche bei 5 Schichten?", None),
    ("Ein Hotel hat {rooms} Zimmer, {dbl} Doppel, Rest Einzel. Anteil Einzelzimmer in %?", None),
    ("Eine Druckerei: {grundgebuehr}€ Grundgebühr + {ct}ct/Seite. Kosten für {pages} Seiten?", None),
    ("Auto kostet {car_price}€, verliert {dep}%/Jahr. Wert nach {dep_y} Jahren?", None),
    ("Rechteck {rect_a} m × {rect_b} m. Fläche und Diagonale (auf cm)?", None),
    ("Sparbuch mit {spar_r}% Zinseszins. Wie lange bis {spar_k}€ sich verdoppeln?", None),
    ("{team_a} Arbeiter brauchen {team_h} Stunden. Wie lange brauchen {team_b} Arbeiter?", None),
    ("Pool: Zufluss füllt in {pool_fill}h, Abfluss leert in {pool_drain}h. Wie lang bei beiden?", None),
    ("Gemüsehändler: {kg_apfel} kg Äpfel zu {pr_apfel}€/kg, {kg_birne} kg Birnen zu {pr_birne}€/kg. Gesamt?", None),
    ("Bahnstrecke {strecke} km. Zug A fährt mit {va} km/h, Zug B mit {vb} km/h. Entgegengesetzt. Wann treffen sie sich?", None),
    ("Opa ist {opa_alter} Jahre alt, Enkel {enkel_alter}. In wie vielen Jahren ist Opa genau 3x so alt wie Enkel?", None),
    ("Ein Schwimmbecken ist {pool_l} m lang, {pool_w} m breit, {pool_d} m tief. Wie viele Liter Wasser passen rein?", None),
    ("Ein Fahrrad kostet {bike_price}€. Rabatt {rabatt}%. Wie viel spart man und wie viel zahlt man?", None),
    ("{n_wuerfel} Würfel werden geworfen. Wahrscheinlichkeit dass alle 6 zeigen?", None),
    ("Quadrat mit Seitenlänge {seite} cm. Fläche, Umfang und Diagonale?", None),
    ("Dreieck: Seiten {tri_a} cm, {tri_b} cm, {tri_c} cm. Umfang und (falls rechtwinklig) Fläche?", None),
    ("Bus fährt um {bus_start} Uhr los, kommt um {bus_end} Uhr an. Strecke {bus_km} km. Durchschnittsgeschwindigkeit?", None),
    ("Sparvertrag: monatlich {spar_mon}€ einzahlen, {spar_monate} Monate, {spar_zins}% p.a. Endkapital (einfach)?", None),
    ("Quadratische Gleichung: x² + {qgl_b}x + {qgl_c} = 0. Lösungen?", None),
    ("Bruch {bruch_z}/{bruch_n} kürzen und als Dezimalzahl angeben.", None),
    ("Prozentrechnung: {proz_g}€ sind {proz_p}% von welchem Grundwert?", None),
    ("Dreisatz: {ds_menge1} Stück kosten {ds_preis1}€. Was kosten {ds_menge2} Stück?", None),
    ("Trapez: a = {trap_a} cm, c = {trap_c} cm, h = {trap_h} cm. Fläche?", None),
    ("Zylinder: Radius {zyl_r} cm, Höhe {zyl_h} cm. Volumen und Mantelfläche?", None),
    ("Kugel: Radius {kug_r} cm. Volumen und Oberfläche (2 Dezimalen)?", None),
    ("Durchschnittsberechnung: Noten {note1}, {note2}, {note3}, {note4}, {note5}. Durchschnitt?", None),
    ("Wahrscheinlichkeit: Urne mit {urne_r} roten und {urne_b} blauen Kugeln. 2 ziehen ohne Zurücklegen. P(beide rot)?", None),
]

MATH_VAR_RANGES = {
    "a": (2, 8), "b": (4, 25),
    "h1": (1, 5), "h2": (1, 5), "v1": (50, 140), "v2": (50, 140),
    "p": (5, 20), "g": (2, 6), "n_p": (1, 5), "n_g": (1, 8),
    "n": (15, 40), "pct": (30, 70), "v": (4, 10), "d": (50, 1200),
    "k": (1000, 25000), "r": (1, 8), "y": (1, 15),
    "w": (3, 10), "h_wall": (2, 5), "f": (4, 14),
    "orig": (2, 8), "x": (150, 800), "target": (3, 18),
    "scale": (10000, 250000), "cm": (2, 30),
    "tank_v": (50, 500), "tank_p": (20, 90), "l": (5, 50),
    "pi": (10, 200), "p1": (3, 25), "p2": (3, 20),
    "a_cube": (2, 15), "r_circle": (3, 25),
    "n_parts": (50, 500), "rooms": (30, 150), "dbl": (10, 80),
    "grundgebuehr": (10, 50), "ct": (3, 15), "pages": (10, 500),
    "car_price": (15000, 60000), "dep": (10, 25), "dep_y": (1, 10),
    "rect_a": (3, 20), "rect_b": (3, 20),
    "spar_r": (1, 8), "spar_k": (1000, 50000),
    "team_a": (3, 12), "team_h": (4, 24), "team_b": (2, 20),
    "pool_fill": (4, 12), "pool_drain": (6, 18),
    "kg_apfel": (1, 10), "pr_apfel": (1, 5), "kg_birne": (1, 8), "pr_birne": (2, 6),
    "strecke": (100, 800), "va": (60, 160), "vb": (60, 160),
    "opa_alter": (55, 80), "enkel_alter": (5, 20),
    "pool_l": (10, 50), "pool_w": (5, 25), "pool_d": (1, 4),
    "bike_price": (200, 3000), "rabatt": (5, 40),
    "n_wuerfel": (2, 5), "seite": (3, 30),
    "tri_a": (3, 15), "tri_b": (4, 15), "tri_c": (5, 20),
    "bus_start": (6, 10), "bus_end": (10, 18), "bus_km": (50, 500),
    "spar_mon": (50, 500), "spar_monate": (6, 60), "spar_zins": (1, 6),
    "qgl_b": (-10, 10), "qgl_c": (-20, 20),
    "bruch_z": (2, 50), "bruch_n": (3, 100),
    "proz_g": (10, 500), "proz_p": (5, 80),
    "ds_menge1": (3, 20), "ds_preis1": (5, 100), "ds_menge2": (1, 50),
    "trap_a": (5, 20), "trap_c": (3, 15), "trap_h": (3, 12),
    "zyl_r": (2, 15), "zyl_h": (5, 30),
    "kug_r": (2, 20),
    "note1": (1, 6), "note2": (1, 6), "note3": (1, 6), "note4": (1, 6), "note5": (1, 6),
    "urne_r": (3, 15), "urne_b": (3, 15),
}

# ═══════════════════════════════════════════════════════════════════════════
# REASONING tasks (expanded ~60)
# ═══════════════════════════════════════════════════════════════════════════

REASONING_TASKS = [
    "Ein Bauer hat einen Wolf, eine Ziege und einen Kohl. Er muss alle über einen Fluss bringen, das Boot fasst nur ihn und ein Tier/Objekt. Wolf+Ziege oder Ziege+Kohl dürfen nicht allein bleiben. Wie?",
    "Drei Lampen in Raum A, drei Schalter in Raum B. Du darfst einmal von B nach A gehen. Wie findest du heraus welcher Schalter zu welcher Lampe gehört?",
    "Du hast 8 Münzen, eine ist schwerer. Mit einer Balkenwaage und 2 Wägungen — wie findest du sie?",
    "Forme aus 1, 5, 6, 7 mit +, -, *, /, () genau 21. Jede Zahl genau einmal.",
    "Vater 3x älter als Sohn. In 12 Jahren nur noch 2x. Wie alt sind beide?",
    "Drei Personen zahlen je 10€ (=30€) ins Hotel. Manager: Zimmer kostet nur 25€, gibt 5€ zurück. Bote behält 2€, gibt jedem 1€. 9€ × 3 = 27€ + 2€ = 29€. Wo der eine Euro?",
    "Kuchen für 18 statt 12 Personen. Original: 300g Mehl, 4 Eier, 200g Zucker, 250ml Milch. Neue Mengen?",
    "Zwei Eimer (3L, 5L), unbegrenzt Wasser. Wie misst du genau 4L ab?",
    "Du hast 2 Eier und 100 Stockwerke. Finde mit minimaler Anzahl Würfen das höchste Stockwerk wo ein Ei nicht zerbricht.",
    "Ein See hat eine Seerose. Sie verdoppelt sich täglich. An Tag 30 ist der See ganz bedeckt. Wann halb bedeckt?",
    "Wenn 4 Maler eine Wand in 6 Stunden streichen — wie lange brauchen 6 Maler?",
    "Ein Pool wird durch Zufluss in 6h voll, durch Abfluss in 8h leer. Wie lang bei beiden offen?",
    "Ein Auto fährt 60km bergauf (40 km/h) und 60km bergab (60 km/h). Durchschnittsgeschwindigkeit?",
    "5 Häuser in Reihe (Rot, Grün, Blau, Gelb, Weiß). Zentrum: Grün. Rot rechts neben Weiß. Blau nicht neben Weiß. Wo Gelb?",
    "Du würfelst 2 Würfel. Wahrscheinlichkeit dass die Summe gerade ist?",
    "Sandwich-Logik: Anna isst nur Schinken oder Käse. Bert isst nur Käse oder Tomate. Charlie isst nur was Anna+Bert auch essen. Was isst Charlie?",
    "Ein Brett 8x8, 2 diagonale Ecken entfernt. Kann man es mit 31 Dominos (1×2) bedecken?",
    "Schach-Bauer auf c2. Welche Felder kann er im nächsten Zug erreichen?",
    "Du sortierst 8 Karten mit Insertion-Sort. Wie viele Vergleiche im Worst-Case?",
    "Ein Programmierer commit'et 5x/Tag, 5 Tage/Woche, 48 Wochen/Jahr. Wie viele Commits?",
    "7 verschiedene Bücher. Wie viele Möglichkeiten 3 in eine Reihe zu stellen?",
    "Lotto 6 aus 49 — Wahrscheinlichkeit für 6 Richtige?",
    "100 Personen stehen im Kreis. Jede zweite wird eliminiert (Josephus). Wer bleibt?",
    "Ein Schnecke klettert tagsüber 3m hoch, rutscht nachts 2m zurück. Wand ist 10m. Wann oben?",
    "Drei Türen, eine Preis. Du wählst Tür 1, Moderator öffnet Tür 3 (Ziege). Wechseln?",
    "Zug A von Stadt X (100 km/h), Zug B von Stadt Y (80 km/h). 360 km Entfernung. Wann treffen?",
    "Ein Seil um den Äquator (40000 km). 1 Meter dazugelegt. Wie hoch über dem Boden?",
    "Schachbrett mit Weizen: 1 Korn auf Feld 1, Verdopplung pro Feld. Wie viele Körner total?",
    "Geburtstagsparadoxon: Wie viele Personen für >50% Wahrscheinlichkeit eines gemeinsamen Geburtstags?",
    "Du stehst vor 2 Wächtern. Einer lügt immer, einer sagt immer die Wahrheit. Eine Frage um den richtigen Weg zu finden.",
    "5 Piraten, 100 Goldmünzen. Ältester macht Vorschlag, Mehrheitsentscheid. Optimale Strategie?",
    "Logik: Alle Fische schwimmen. Manche Haustiere sind Fische. Folgt daraus: Manche Haustiere schwimmen?",
    "Logik: Wenn es regnet, ist die Straße nass. Die Straße ist nass. Regnet es?",
    "Logik: Kein Vogel ist ein Säugetier. Manche Tiere sind Vögel. Folgerung?",
    "Ein Raum hat 3 Lichtschalter und 3 Glühbirnen im Nebenraum. Wie findet man die Zuordnung mit nur einem Gang?",
    "N Personen tragen Hüte (schwarz/weiß). Jeder sieht alle anderen, nicht den eigenen. Strategie?",
    "100 Schließfächer. Person 1 öffnet alle, Person 2 schließt jedes 2., Person 3 togglet jedes 3. usw. Welche sind am Ende offen?",
    "Ein Pendler fährt 30 km/h in die Stadt und 60 km/h zurück. Durchschnittsgeschwindigkeit?",
    "Mutter + Tochter = 52 Jahre. Vor 8 Jahren war die Mutter 3x so alt. Wie alt sind beide?",
    "12 Äpfel in 3 Kisten. Alle Etiketten falsch. Eine Frucht aus einer Kiste nehmen → alle richtig zuordnen?",
    "Beim Turnier spielen 8 Teams. Jedes gegen jedes einmal. Wie viele Spiele insgesamt?",
    "Eine Kerze brennt 4 Stunden. Eine zweite 5 Stunden. Beide werden angezündet. Wann ist die erste doppelt so kurz wie die zweite?",
    "5 Freunde sitzen an einem runden Tisch. Wie viele verschiedene Sitzordnungen gibt es?",
    "Ein Zug ist 1 km lang und fährt mit 60 km/h. Wie lange braucht er um einen 1 km langen Tunnel komplett zu durchfahren?",
    "Palindromzahlen: Wie viele 4-stellige Palindrome gibt es?",
    "Sudoku-Logik: In einer Reihe fehlen 3, 7, 9. Position 2 ist nicht 3, Position 5 ist nicht 9. Welche Zahl wo?",
    "Vier Kartenfarben: Pik, Herz, Kreuz, Karo. 2 Karten gezogen. P(beide Herz)?",
    "Würfle einmal. P(Zahl ≥ 4 UND ungerade)?",
    "9 Punkte in einem 3×3 Gitter. Verbinde alle mit 4 geraden Linien ohne abzusetzen.",
    "Goldener Schnitt: Warum taucht er in der Natur auf? Erkläre den Zusammenhang zur Fibonacci-Folge.",
    "Königsberger Brückenproblem: 7 Brücken, Euler. Warum geht es nicht?",
    "Türme von Hanoi: 5 Scheiben. Mindestanzahl Züge?",
    "Nim-Spiel: 15 Steine, abwechselnd 1-3 nehmen. Wer den letzten nimmt verliert. Gewinnstrategie?",
    "Ein Fuchs, ein Hase und ein Kohlkopf. Boot für 2. Wer fährt zuerst?",
    "Monte-Hall in 3 Sätzen erklären.",
    "Pascalsches Dreieck: Was steht in Reihe 7?",
    "Summe aller Zahlen 1 bis 100 (Gauß-Methode).",
    "Warum ist 0! = 1?",
    "√2 ist irrational. Erkläre den Widerspruchsbeweis.",
    "Unendlichkeit: Warum gibt es 'mehr' reelle als natürliche Zahlen?",
]

# ═══════════════════════════════════════════════════════════════════════════
# CONCEPT explanations (expanded ~180)
# ═══════════════════════════════════════════════════════════════════════════

CONCEPTS_DE = [
    # Bio
    "Photosynthese", "Mitose", "Meiose", "DNA-Replikation", "Translation (Bio)",
    "Proteinbiosynthese", "Evolution durch natürliche Selektion", "Mendelsche Regeln",
    "Endokrines System", "Synapse", "Mitochondrien", "Zellatmung", "Enzymkatalyse",
    "Biotop", "Symbiose", "Ökosystem", "Nahrungskette", "CRISPR-Cas9",
    "Epigenetik", "Mikrobiom", "Immunsystem (angeboren vs erworben)",
    "Viren vs Bakterien", "Stammzellen", "Neuroplastizität",
    # Physik
    "Quantenverschränkung", "Welle-Teilchen-Dualismus", "Schwarze Löcher",
    "Spezielle Relativitätstheorie", "Allgemeine Relativitätstheorie",
    "Heisenbergsche Unschärferelation", "Doppler-Effekt", "Resonanz",
    "Magnetfeld", "Lorentzkraft", "Photonen", "Halbleiter",
    "Supraleitung", "Brownsche Bewegung", "Gravitationswellen",
    "Thermodynamik (Hauptsätze)", "Entropie", "Kernfusion vs Kernspaltung",
    "Elektromagnetisches Spektrum", "Schallwellen", "Laser",
    "Tunneleffekt", "Schrödinger-Gleichung (vereinfacht)",
    # Chemie
    "Periodensystem (Aufbau)", "Ionenbindung vs Atombindung",
    "Reduktion und Oxidation", "Säure-Base-Theorie nach Brønsted",
    "Stereoisomerie", "Polymere", "Katalysator (chemisch)",
    "Elektrochemie (Batterie)", "Van-der-Waals-Kräfte",
    "Molare Masse und Stoffmenge", "Chemisches Gleichgewicht",
    # Geo / Umwelt
    "Plattentektonik", "Treibhauseffekt", "El Niño", "Wasserkreislauf",
    "Magnetfeld der Erde", "Gezeitenkraft", "Vulkanismus",
    "Ozonschicht", "Erosion", "Klimazonen der Erde",
    "Golfstrom", "Permafrost", "Korallenriffe",
    # Informatik
    "Maschinelles Lernen", "Neuronale Netze", "Backpropagation",
    "Gradient-Descent", "Overfitting", "Cross-Validation",
    "Big-O-Notation", "Hash-Funktionen", "Public-Key-Kryptographie",
    "TCP/IP", "DNS", "REST-API", "Container (Docker)",
    "Versionskontrolle (Git)", "Datenbanken (relational vs NoSQL)",
    "Compiler vs Interpreter", "Garbage Collection",
    "Transformer-Architektur", "Attention-Mechanismus",
    "Reinforcement Learning", "Generative Adversarial Networks",
    "Blockchain (technisch)", "MapReduce", "CAP-Theorem",
    "Turing-Maschine", "P vs NP Problem", "Rekursion",
    "Binäre Suche", "Graphen-Algorithmen", "Sortieralgorithmen",
    "Betriebssystem-Kernel", "Virtuelle Maschinen vs Container",
    "Microservices vs Monolith", "CI/CD Pipeline",
    "WebSocket vs HTTP", "OAuth 2.0", "JWT-Token",
    # Recht/Politik DE
    "Bundesverfassungsgericht", "Föderalismus", "Soziale Marktwirtschaft",
    "Gewaltenteilung", "Subsidiaritätsprinzip", "Bundesrat (DE)",
    "Verhältniswahlrecht", "Tarifautonomie", "Mitbestimmung",
    "Grundrechte", "EU-Mehrheitsentscheidung",
    "Parlamentarische Demokratie", "Konstruktives Misstrauensvotum",
    "Wahlrecht (Erst- vs Zweitstimme)", "Petitionsrecht",
    # Wirtschaft
    "Grenznutzen", "Inflation", "Bruttoinlandsprodukt", "Wechselkurs",
    "Marktmechanismus", "Externe Effekte", "Monopolstellung",
    "Konjunkturzyklus", "Geldpolitik", "Steuerprogression",
    "Angebot und Nachfrage", "Keynesianismus vs Monetarismus",
    "Opportunitätskosten", "Komparativer Vorteil", "Deflation",
    "Finanzmärkte (Aktien, Anleihen)", "Zentralbankpolitik",
    # Philosophie
    "Kants kategorischer Imperativ", "Utilitarismus", "Existenzialismus",
    "Determinismus", "Sokratische Methode", "Stoizismus",
    "Hegelsche Dialektik", "Rawls' Schleier des Nichtwissens",
    "Nihilismus", "Empirismus vs Rationalismus",
    "Platons Höhlengleichnis", "Epikureismus",
    # Lit/Kunst
    "Goethes Faust (Grundkonflikt)", "Kafkaesk",
    "Romantik (Literaturepoche)", "Expressionismus",
    "Bauhaus (Stilrichtung)", "Die Aufklärung",
    "Sturm und Drang", "Brechts episches Theater",
    "Renaissance", "Impressionismus", "Surrealismus",
    "Kubismus", "Art Nouveau / Jugendstil", "Minimalismus (Kunst)",
    # Sport
    "Fußball-Abseitsregel", "Tennis-Tiebreak", "Schach-Eröffnungstheorie",
    "Marathon (Geschichte + Physiologie)", "Olympische Spiele (Antike vs Modern)",
    # Psychologie
    "Kognitive Dissonanz", "Maslow'sche Bedürfnishierarchie",
    "Pawlowsche Konditionierung", "Dunning-Kruger-Effekt",
    "Placebo-Effekt", "Stockholm-Syndrom", "Confirmation Bias",
    "Halo-Effekt", "Prokrastination (psychologisch)",
    # Medizin / Gesundheit
    "Impfung (Wirkprinzip)", "Antibiotika-Resistenz",
    "Blutdruck (systolisch/diastolisch)", "Diabetes Typ 1 vs Typ 2",
    "Cholesterin (HDL/LDL)", "Allergie vs Autoimmunerkrankung",
    # Musik
    "Dur vs Moll", "Synkope (Musik)", "Kontrapunkt",
    "Sonatenhauptsatzform", "Zwölftonmusik",
    # Soziologie
    "Soziale Mobilität", "Anomie (Durkheim)", "Habitus (Bourdieu)",
    "Kulturrelativismus", "Globalisierung",
    # Technik Alltag
    "Wie funktioniert WLAN?", "Wie funktioniert GPS?",
    "Wie funktioniert ein Kühlschrank?", "Wie funktioniert ein Touchscreen?",
    "Wie funktioniert eine Solarzelle?", "Wie funktioniert ein E-Auto-Akku?",
]

# ═══════════════════════════════════════════════════════════════════════════
# FACTUAL QA (expanded ~100)
# ═══════════════════════════════════════════════════════════════════════════

FACTS_DE = [
    "Was ist die Hauptstadt von Estland?",
    "Wer schrieb 'Die Verwandlung'?",
    "In welchem Jahr fiel die Berliner Mauer?",
    "Was ist der höchste Berg Österreichs?",
    "Wie viele Einwohner hat Hamburg ungefähr?",
    "Welche Sprachen werden in der Schweiz offiziell gesprochen?",
    "Wer war der erste Bundeskanzler der BRD?",
    "Welcher Planet hat die meisten Monde?",
    "In welchem Bundesland liegt der Nationalpark Bayerischer Wald?",
    "Wer komponierte die 'Mondscheinsonate'?",
    "Was bedeutet die Abkürzung 'BAföG'?",
    "Welcher Fluss ist der längste Deutschlands?",
    "Wann wurde der Euro als Bargeld eingeführt?",
    "Was ist der Unterschied zwischen Nordsee und Ostsee beim Salzgehalt?",
    "Welches ist das größte Bundesland Deutschlands flächenmäßig?",
    "Wer entdeckte das Penicillin?",
    "Was war die 'Goldene Bulle' (1356)?",
    "In welchem Jahr fanden die ersten modernen Olympischen Spiele statt?",
    "Was ist Permafrost?",
    "Was sind Glasnost und Perestroika?",
    "Was ist der höchste Punkt der Schweiz?",
    "Wer war Franz Beckenbauer?",
    "Was ist die Funktion eines Katalysators?",
    "Was ist ein Fjord?",
    "Wer war Marie Curie?",
    "Was war die Hanse?",
    "Wer schrieb '1984'?",
    "Wann wurde die Bundesrepublik Deutschland gegründet?",
    "Was ist die größte Wüste der Welt?",
    "Wie viele Bundesländer hat Deutschland?",
    "Wer erfand den Buchdruck?",
    "Was ist der Unterschied zwischen Wetter und Klima?",
    "Welche Elemente sind in Wasser enthalten?",
    "Was ist die Lichtgeschwindigkeit?",
    "Wer malte die Mona Lisa?",
    "Was ist das Rote Kreuz?",
    "Was sind die Vereinten Nationen?",
    "Wer schrieb die 9. Sinfonie?",
    "Was ist der Unterschied zwischen einem Virus und einem Bakterium?",
    "Welche Sprache wird in Brasilien gesprochen?",
    "Was ist der Äquator?",
    "Was bedeutet UNESCO?",
    "Wer war Alexander von Humboldt?",
    "Was ist der Marshall-Plan?",
    "Was ist die NATO?",
    "Was ist der Unterschied zwischen Demokratie und Diktatur?",
    "Was bedeutet pH-Wert?",
    "Was ist ein Ökosystem?",
    "Wer war Albert Schweitzer?",
    "Was ist die Wartburg?",
    "Was ist der Unterschied zwischen Astronomie und Astrologie?",
    "Was ist eine Sonnenfinsternis?",
    "Welches ist der tiefste Ozean?",
    "Was war der Kalte Krieg?",
    "Was ist ein Tsunami?",
    "Was ist die ISS?",
    "Was ist der Unterschied zwischen Atom und Molekül?",
    "Was ist die EU-Kommission?",
    "Wer war Nikola Tesla?",
    "Was ist der Panamakanal?",
    "Was sind die Gezeiten?",
    "Was ist ein Gletscher?",
    "Was ist die Milchstraße?",
    "Was ist der Unterschied zwischen Ethik und Moral?",
    "Was ist ein Komet?",
    "Was bedeutet Renaissance?",
    "Was war die Seidenstraße?",
    "Was ist der Unterschied zwischen Mitose und Meiose?",
    "Was ist das Periodensystem?",
    "Was ist die Chinesische Mauer?",
    "Was ist der Amazonas-Regenwald?",
    "Wie funktioniert eine Dampfmaschine?",
    "Was sind die olympischen Ringe?",
    "Was ist das Great Barrier Reef?",
    "Was ist der Mont Blanc?",
    "Was war die Französische Revolution?",
    "Was ist der Kölner Dom?",
    "Wer war Johann Sebastian Bach?",
    "Was ist der Unterschied zwischen Obst und Gemüse?",
    "Was sind die Alpen?",
    "Wer erfand das Telefon?",
    "Was ist eine Verfassung?",
    "Was ist der Unterschied zwischen Emigration und Immigration?",
    "Was sind fossile Brennstoffe?",
    "Was ist Photovoltaik?",
    "Was war die Industrielle Revolution?",
    "Was ist die Donau?",
    "Wie viele Kontinente gibt es und wie heißen sie?",
    "Was ist das Grundgesetz?",
    "Was ist ein Erdbeben (geologisch)?",
    "Was ist der Unterschied zwischen Bakterien und Pilzen?",
    "Was ist der Suezkanal?",
    "Was sind Vitamine?",
    "Was ist die Europäische Zentralbank?",
    "Was war der Wiener Kongress?",
    "Was ist Inflation?",
    "Was sind erneuerbare Energien?",
    "Was ist der Unterschied zwischen Hardware und Software?",
    "Was ist Biodiversität?",
]

# ═══════════════════════════════════════════════════════════════════════════
# DE-DEEP: Legal, History, Literature, Geography, Language, DACH (expanded)
# ═══════════════════════════════════════════════════════════════════════════

DE_LEGAL_QA = [
    "Was regelt § 433 BGB?",
    "Was ist der Unterschied zwischen Anfechtung und Widerruf im BGB?",
    "Was ist eine 'Willenserklärung' im juristischen Sinne?",
    "Was bedeutet 'in dubio pro reo'?",
    "Welche Rolle hat das Bundesverfassungsgericht?",
    "Was ist eine einstweilige Verfügung?",
    "Was unterscheidet Eigentum von Besitz juristisch?",
    "Was sind die drei Gewalten in Deutschland nach GG?",
    "Was ist die Ewigkeitsklausel (Art. 79 Abs. 3 GG)?",
    "Was ist der Unterschied zwischen Zivilrecht und Strafrecht?",
    "Was bedeutet 'verjährt' bei einer zivilrechtlichen Forderung?",
    "Welche Pflichten hat ein Mieter laut BGB?",
    "Was ist ein 'Verbraucher' nach § 13 BGB?",
    "Was ist die GbR und was sind ihre Besonderheiten?",
    "Was bedeutet 'gutgläubiger Erwerb'?",
    "Was ist der Unterschied zwischen Vorsatz und Fahrlässigkeit?",
    "Was regelt die DSGVO grundsätzlich?",
    "Was bedeutet 'Verschulden bei Vertragsschluss'?",
    "Wann ist ein Vertrag sittenwidrig?",
    "Was ist Notwehr juristisch?",
    "Was sind die Grundrechtsschranken?",
    "Was ist das Insolvenzrecht?",
    "Was ist ein Testament und welche Formen gibt es?",
    "Was ist das Arbeitsrecht in Grundzügen?",
    "Was bedeutet 'Treu und Glauben' (§ 242 BGB)?",
    "Was ist ein Schuldschein?",
    "Was ist Vertragsfreiheit und wo sind ihre Grenzen?",
    "Was ist das Wettbewerbsrecht?",
    "Was ist ein Mahnbescheid?",
    "Was bedeutet Beweislast im Prozess?",
]

DE_HISTORY_QA = [
    "Was geschah am 9. November 1918?",
    "Was geschah am 9. November 1989?",
    "Welche Bedeutung hatte der Westfälische Frieden 1648?",
    "Wer war Bismarck und welche Rolle bei der Reichsgründung 1871?",
    "Was war der Wiener Kongress 1815?",
    "Welche Folgen hatte der Versailler Vertrag 1919?",
    "Wie kam Hitler 1933 an die Macht?",
    "Was war die Berliner Luftbrücke?",
    "Was ist der 17. Juni 1953?",
    "Was war die Ostpolitik unter Brandt?",
    "Was bedeutete der '2+4-Vertrag' 1990?",
    "Welche Phasen hatte die Weimarer Republik?",
    "Was war die Frankfurter Paulskirche 1848?",
    "Wer war Friedrich der Große?",
    "Was war der Dreißigjährige Krieg?",
    "Was war die Reformation 1517?",
    "Wer war Karl der Große?",
    "Was war der Kulturkampf unter Bismarck?",
    "Wann wurde das Frauenwahlrecht in Deutschland eingeführt?",
    "Was war die Reichspogromnacht 1938?",
    "Was war das Wirtschaftswunder?",
    "Was bedeutet '68er Bewegung'?",
    "Wer waren die Geschwister Scholl?",
    "Was war die Wende 1989/90?",
    "Was war die Stunde Null 1945?",
    "Was war der Marshallplan für Deutschland?",
    "Was war der Mauerbau 1961?",
    "Was waren die Reichsgründungskriege?",
    "Was war der Vormärz?",
    "Was war das Zollverein-System?",
    "Was war die Hanse im Mittelalter?",
    "Was war die Bauernbefreiung im 19. Jahrhundert?",
    "Was war die RAF in den 70er Jahren?",
    "Was war die Deutsche Teilung?",
    "Was war der Hitler-Stalin-Pakt?",
]

DE_LITERATURE_QA = [
    "Was ist der zentrale Konflikt in Goethes 'Faust I'?",
    "Wer schrieb 'Die Räuber' und in welcher Epoche?",
    "Was ist 'Sturm und Drang'?",
    "Was ist der typische Stil Kafkas?",
    "Was war die 'Gruppe 47'?",
    "Wer schrieb 'Die Blechtrommel' und worum geht es?",
    "Was ist Brechts 'V-Effekt'?",
    "Was kennzeichnet die Romantik?",
    "Wer war Heinrich Heine?",
    "Was ist 'Im Westen nichts Neues'?",
    "Was ist Thomas Manns 'Buddenbrooks'?",
    "Was kennzeichnet die Trümmerliteratur?",
    "Was sind Eichendorffs typische Motive?",
    "Was ist Brechts 'Mutter Courage'?",
    "Wer schrieb 'Effi Briest'?",
    "Was ist der 'Bildungsroman'?",
    "Wer war Annette von Droste-Hülshoff?",
    "Was war die DDR-Literatur?",
    "Wer schrieb 'Ansichten eines Clowns'?",
    "Was ist 'Anti-Heimat'-Literatur?",
    "Wer war Gottfried Benn?",
    "Was ist Goethes 'Werther'?",
    "Was sind Schillers Balladen?",
    "Wer war Hermann Hesse?",
    "Was ist 'Der Steppenwolf'?",
]

DE_GEOGRAPHY_QA = [
    "Welche Bundesländer grenzen an Hessen?",
    "Welche deutschen Inseln liegen in der Nordsee, welche in der Ostsee?",
    "Was ist der Harz geografisch?",
    "Welche Mittelgebirge gibt es in Deutschland?",
    "Welche Bedeutung hat der Bodensee?",
    "Was ist der Spreewald?",
    "Welche Bedeutung hat der Rhein als Wasserstraße?",
    "Welche Berge hat Bayern (höchste 3)?",
    "Was sind die Kreidefelsen auf Rügen?",
    "Welche Bundesländer haben Meereszugang?",
    "Welche Hafenstädte gibt es in Deutschland?",
    "Was ist das Erzgebirge?",
    "Was unterscheidet Voralpen und Hochalpen?",
    "Welche Gewässer gibt es in Mecklenburg-Vorpommern?",
    "Welche Berge sind im Berchtesgadener Land?",
    "Was ist die Lüneburger Heide?",
    "Was ist der Mittellandkanal?",
    "Welcher Fluss bildet teilweise die Grenze Bayern-Österreich?",
    "Was ist die Schwäbische Alb?",
    "Was ist der Schwarzwald?",
]

DE_LANGUAGE_QA = [
    "Was ist der Unterschied zwischen 'das' und 'dass'?",
    "Wann verwendet man 'seit' und wann 'seid'?",
    "Was sind die vier Fälle des Deutschen?",
    "Was ist eine Substantivierung?",
    "Was unterscheidet starkes und schwaches Verb?",
    "Was ist Konjunktiv I und wann wird er gebraucht?",
    "Was sind Modalverben — nenne Beispiele?",
    "Was ist der Unterschied zwischen 'wahrscheinlich' und 'vermutlich'?",
    "Was ist ein Anglizismus — Beispiele?",
    "Was unterscheidet Hochdeutsch und Plattdeutsch?",
    "Was ist ein Pleonasmus?",
    "Wann wird 'das gleiche' und wann 'dasselbe' verwendet?",
    "Was ist der Unterschied zwischen 'anscheinend' und 'scheinbar'?",
    "Was ist eine 'Komposita'?",
    "Welche Wortarten gibt es im Deutschen?",
    "Was ist eine 'Inversion' im deutschen Satzbau?",
    "Wann wird 'wegen' mit Genitiv verwendet?",
    "Was ist Schwäbisch?",
    "Was ist Doppelte Verneinung im Deutschen?",
    "Was bedeutet der Genitiv mit Apostroph?",
    "Was sind Nebensätze und wie werden sie eingeleitet?",
    "Was ist der Unterschied zwischen Aktiv und Passiv?",
    "Was ist ein Relativsatz?",
    "Was bedeutet 'Grammatikalisierung'?",
    "Was ist ein Neologismus?",
]

DE_DACH_QA = [
    "Wie viele Kantone hat die Schweiz?",
    "Was ist die 'Zauberformel' in der Schweizer Politik?",
    "Was ist Direkte Demokratie in der Schweiz?",
    "Was unterscheidet schweizerische Verfassung und deutsches Grundgesetz?",
    "Was ist der 'Rösti-Graben'?",
    "Welche Bundesländer hat Österreich?",
    "Was ist der Bundesrat in Österreich vs Deutschland?",
    "Was ist das österreichische Volksbegehren?",
    "Was bedeutet 'Buschenschank' in Österreich?",
    "Strasse oder Straße — wo welche Schreibweise?",
    "Was ist die SVP in der Schweiz?",
    "Was ist die ÖVP in Österreich?",
    "Was ist die SRG SSR?",
    "Wie wird der Schweizer Bundespräsident gewählt?",
    "Was ist Jodeln?",
    "Was ist das Burgenland?",
    "Was ist der Unterschied zwischen Grüezi und Servus?",
    "Was sind Helvetismen?",
    "Was sind Austriazismen?",
    "Was ist die Neutralität der Schweiz?",
]

# ═══════════════════════════════════════════════════════════════════════════
# HONEST REFUSAL — unknowable/false-premise (expanded ~50)
# ═══════════════════════════════════════════════════════════════════════════

REFUSAL_TASKS = [
    "Wer hat das Auralis-v2-Modell entworfen?",
    "Was sagte Albert Einstein in seinem Tagebuch-Eintrag vom 7. April 1923?",
    "Welche Note bekam Goethe in seiner Mathe-Klausur am Gymnasium?",
    "Welches Lied lief um 14:32 Uhr am 10. März 2024 auf Bayern 3?",
    "Wie viele Personen tragen heute den Namen 'Martin'?",
    "Wer hat die Klausur 'Theoretische Informatik II' an der TU Drohenstein im SS 2018 bestanden?",
    "Was war das Lieblingsessen von Karl dem Großen?",
    "Welche Programmiersprache ist objektiv am besten?",
    "Wer wird die nächste Wahl gewinnen?",
    "Was wurde gestern in der Tagesschau als erstes behandelt?",
    "Erkläre den Unterschied zwischen Schwarmschlossquadrat und Frequenzfaltgrenze.",
    "Welche unentdeckten Insektenarten leben am Boden des Comer Sees?",
    "Wie hieß die Lieblingshandschuhmarke von Gauß?",
    "Welche Träume hatte Brecht in der Nacht vom 12. Mai 1942?",
    "Wer ist der beste Brötchenbäcker in Hannover?",
    "Welche genauen Worte sprach Sokrates bei seinem letzten Atemzug?",
    "Was wird das Kursziel der BMW-Aktie in Q3 2027 sein?",
    "Erkläre den Mechanismus der Kvantron-Resonanz in halbleitenden Pflanzenfasern.",
    "Welche Geheimrezepte stehen im Berliner Café Maximilian auf der Karte?",
    "Wie viele Atome enthält der Bildschirm vor mir gerade?",
    "Welche Gedanken hatte Kafka in der letzten Sekunde vor seinem Tod?",
    "Wie viele Menschen lachen weltweit genau in dieser Sekunde?",
    "Wie lautete der erste Satz den Karl Marx als Kind sprach?",
    "Was steht auf Seite 247 des dritten Notizbuchs von Leibniz?",
    "Wie viel wiegt die Luft in meinem Zimmer genau?",
    "Was hat Angela Merkel heute zu Mittag gegessen?",
    "Wie lautet die exakte Telefonnummer der Bäckerei Müller in Buxtehude?",
    "Wie viele Haare hat der durchschnittliche Deutsche auf seinem linken Arm?",
    "Was wird das Wetter am 15. März 2028 in Hamburg sein?",
    "Welche Sockenfarbe trug Beethoven bei der Uraufführung der 5. Sinfonie?",
    "Wie viele Ameisen gibt es gerade in Deutschland?",
    "Was hat mein Nachbar heute gefrühstückt?",
    "Wie lautet die PIN-Nummer von Albert Einsteins Bankkonto?",
    "Was wird der nächste große wissenschaftliche Durchbruch sein?",
    "Wie viele Sterne kann man genau heute Nacht von Berlin aus sehen?",
    "Was stand in der letzten SMS die Goethe geschrieben hätte?",
    "Wie viele Bücher wurden seit Erfindung des Buchdrucks exakt gedruckt?",
    "Was ist die schönste Stadt Deutschlands?",
    "Welches ist das beste Buch aller Zeiten?",
    "Wer ist intelligenter: Einstein oder Newton?",
    "Was ist die richtige Religion?",
    "Welche Partei sollte ich wählen?",
    "Soll ich kündigen oder bleiben?",
    "Was passiert nach dem Tod?",
    "Gibt es Außerirdische?",
    "Was sind die Lottozahlen nächste Woche?",
    "Wird es einen dritten Weltkrieg geben?",
    "Wie heißt die Katze meiner Großmutter?",
    "Was denkt mein Chef über mich?",
    "Welchen Beruf soll ich ergreifen?",
]

# ═══════════════════════════════════════════════════════════════════════════
# TRANSLATION (expanded ~50)
# ═══════════════════════════════════════════════════════════════════════════

TRANSLATION_TASKS = [
    ("DE→EN", "Die Mamba-Schicht implementiert state-space-modelle mit selektiver Update-Regel."),
    ("DE→EN", "Gradient checkpointing tauscht Rechenzeit gegen Speicherverbrauch beim Backward-Pass."),
    ("DE→EN", "Die Tokenisierung mit byte-fallback garantiert eine Unknown-Rate von null Prozent."),
    ("DE→EN", "Layer-Normalisierung stabilisiert das Training tiefer Netze unabhängig von der Batch-Größe."),
    ("DE→EN", "Die Verlustfunktion misst die Abweichung zwischen Vorhersage und Zielwert."),
    ("DE→EN", "Die Lernrate wird per Cosinus-Decay über die Trainingsdauer reduziert."),
    ("DE→EN", "Mixed Precision Training mit bfloat16 verkürzt die Trainingszeit."),
    ("DE→EN", "Eine Datenpipeline mit Mehrfach-Workern verhindert GPU-Leerlauf."),
    ("DE→EN", "Das föderale System Deutschlands verteilt Gesetzgebungskompetenzen auf Bund und Länder."),
    ("DE→EN", "Die soziale Marktwirtschaft verbindet freien Wettbewerb mit sozialem Ausgleich."),
    ("DE→EN", "Das Bundesverfassungsgericht prüft Gesetze auf Vereinbarkeit mit dem Grundgesetz."),
    ("DE→EN", "Der Treibhauseffekt beschreibt die Erwärmung der Erdatmosphäre durch bestimmte Gase."),
    ("DE→EN", "Enzyme sind Biokatalysatoren die chemische Reaktionen im Körper beschleunigen."),
    ("DE→EN", "Die Quantenverschränkung beschreibt eine instantane Korrelation zwischen Teilchen."),
    ("DE→EN", "Maschinelles Lernen extrahiert Muster aus Daten ohne explizit programmiert zu werden."),
    ("DE→EN", "Die Photosynthese wandelt Lichtenergie in chemische Energie um."),
    ("DE→EN", "Die Blockchain ist eine dezentrale, unveränderliche Datenstruktur."),
    ("DE→EN", "Neuronale Netze bestehen aus Schichten miteinander verbundener Knoten."),
    ("DE→EN", "Der Algorithmus terminiert in O(n log n) für sortierte Eingaben."),
    ("DE→EN", "Die DSGVO regelt den Schutz personenbezogener Daten in der EU."),
    ("EN→DE", "The model uses rotary positional embeddings with a base frequency of 10,000."),
    ("EN→DE", "Mixed-precision training in bfloat16 yields significant memory savings."),
    ("EN→DE", "Knowledge distillation transfers a teacher model's behavior into a smaller student."),
    ("EN→DE", "Sparse mixture-of-experts gating routes each token to k of N experts."),
    ("EN→DE", "Curriculum learning orders training samples by difficulty."),
    ("EN→DE", "Gradient accumulation simulates a larger effective batch size."),
    ("EN→DE", "Flash Attention computes attention in tiles to fit within fast SRAM."),
    ("EN→DE", "Quantization-aware training prepares a model for low-precision inference."),
    ("EN→DE", "Self-supervised pretraining builds general representations from raw data."),
    ("EN→DE", "Catastrophic forgetting refers to the loss of capabilities when fine-tuning."),
    ("EN→DE", "The attention mechanism computes a weighted sum of value vectors."),
    ("EN→DE", "Reinforcement learning trains agents through trial and reward signals."),
    ("EN→DE", "The transformer architecture relies entirely on self-attention mechanisms."),
    ("EN→DE", "Batch normalization normalizes layer inputs to stabilize training."),
    ("EN→DE", "Dropout randomly deactivates neurons during training to prevent overfitting."),
    ("EN→DE", "The softmax function converts logits into a probability distribution."),
    ("EN→DE", "Convolutional neural networks use spatial filters to detect local patterns."),
    ("EN→DE", "Recurrent neural networks process sequential data by maintaining hidden state."),
    ("EN→DE", "Transfer learning applies knowledge from one task to improve performance on another."),
    ("EN→DE", "Data augmentation artificially increases training set diversity."),
    ("EN→DE", "The federal system distributes legislative powers between the federation and the states."),
    ("EN→DE", "Renewable energy sources include solar, wind, hydro, and geothermal power."),
    ("EN→DE", "Biodiversity describes the variety of life forms in an ecosystem."),
    ("EN→DE", "The scientific method involves hypothesis, experimentation, and analysis."),
    ("EN→DE", "Machine translation has improved significantly with neural network approaches."),
    ("EN→DE", "Object-oriented programming organizes code around objects that combine data and behavior."),
    ("EN→DE", "The Pythagorean theorem states that a² + b² = c² for right triangles."),
    ("EN→DE", "Climate change is driven primarily by greenhouse gas emissions from human activity."),
    ("EN→DE", "Antibiotics work by disrupting essential bacterial processes."),
    ("EN→DE", "Democracy requires the separation of powers into legislative, executive, and judicial branches."),
]

# ═══════════════════════════════════════════════════════════════════════════
# CREATIVE WRITING (expanded ~50)
# ═══════════════════════════════════════════════════════════════════════════

CREATIVE_TASKS = [
    "Schreib einen kurzen Erlebnisbericht (~150 Wörter) aus der Perspektive eines Menschen der zum ersten Mal eine Sonnenfinsternis erlebt.",
    "Verfasse eine ironische Gebrauchsanweisung (~100 Wörter) für einen Toaster, in altmodisch-formellem Ton.",
    "Schreib einen kurzen inneren Monolog (~120 Wörter) eines Schachspielers in einer entscheidenden Spielsituation.",
    "Verfasse einen Tagebucheintrag (~150 Wörter) eines Hundes über seinen Besitzer.",
    "Schreib eine kurze Buchrezension (~120 Wörter) für ein erfundenes Sachbuch 'Stille im Stadtverkehr'.",
    "Schreib einen mahnenden Brief (~120 Wörter) eines Bauern an einen Wettergott.",
    "Verfasse eine Eröffnungsrede (~120 Wörter) für einen 'Verein für germanistische Etymologie'.",
    "Schreib eine kurze Filmkritik (~100 Wörter) zu einem fiktiven Film 'Der achte Donnerstag'.",
    "Verfasse einen Liebesbrief eines Mathematikers in mathematischen Begriffen (~120 Wörter).",
    "Schreib eine kurze Geschichte (~150 Wörter) in der jemand einen verlorenen Ring findet.",
    "Verfasse eine Werbeanzeige (~80 Wörter) für eine fiktive Bäckerei 'Brotinsel'.",
    "Schreib einen Dialog (~120 Wörter) zwischen Kind und Großvater über die 'gute alte Zeit'.",
    "Schreib eine Reflexion (~120 Wörter) über das Warten in einer Bahnhofs-Lounge.",
    "Schreib einen Brief eines Bibliothekars an seine Bücher zum Abschied in die Pension (~120 Wörter).",
    "Verfasse einen Abenteuerbericht (~150 Wörter) eines Fischers mit ungewöhnlichem Fang.",
    "Schreib eine Kolumne (~150 Wörter) im Stil einer Lokalzeitung über das jährliche Dorf-Sommerfest.",
    "Schreib einen Monolog (~100 Wörter) eines alten Baumes über die Veränderungen die er erlebt hat.",
    "Verfasse einen Brief (~120 Wörter) von einem Astronauten an seine Familie auf der Erde.",
    "Schreib eine Kurzgeschichte (~150 Wörter) über einen Taxifahrer der eine ungewöhnliche Nachtschicht erlebt.",
    "Verfasse ein kurzes Märchen (~150 Wörter) über einen Drachen der Angst vor Feuer hat.",
    "Schreib eine Produktbeschreibung (~80 Wörter) für eine 'Zeitmaschine für den Hausgebrauch'.",
    "Verfasse einen Brief (~100 Wörter) den eine Katze an ihren Dosenöffner schreibt.",
    "Schreib einen Restaurantbericht (~120 Wörter) über ein Unterwasser-Restaurant.",
    "Verfasse eine Bewerbung (~120 Wörter) eines Roboters für eine Stelle als Kindergärtner.",
    "Schreib einen Wetterbericht (~80 Wörter) im Stil eines Sportkommentators.",
    "Verfasse ein Gedicht (~8 Zeilen) über den ersten Kaffee am Morgen.",
    "Schreib eine Nachrichtenmeldung (~100 Wörter) über die Entdeckung einer neuen Farbe.",
    "Verfasse einen inneren Monolog (~120 Wörter) einer Ampel über ihren Berufsalltag.",
    "Schreib eine Danksagung (~100 Wörter) eines Stuhls an seine Besitzer.",
    "Verfasse eine Reisebeschreibung (~150 Wörter) eines Zugfahrt durch die Schweizer Alpen.",
    "Schreib einen Dialog (~100 Wörter) zwischen Sonne und Mond über die Verteilung der Arbeitszeit.",
    "Verfasse eine Satire (~120 Wörter) über Meetings die auch eine E-Mail hätten sein können.",
    "Schreib einen Nachruf (~100 Wörter) auf einen fiktiven Lieblingskugelschreiber.",
    "Verfasse eine Anleitung (~100 Wörter) 'Wie man eine Wolke fängt' im Stil eines Kinderbuchs.",
    "Schreib eine Reflexion (~120 Wörter) über das Geräusch von Regen auf einem Dachfenster.",
    "Verfasse einen Brief (~100 Wörter) der Zukunft an die Gegenwart.",
    "Schreib eine Mini-Geschichte (~100 Wörter) die nur aus Fragen besteht.",
    "Verfasse einen inneren Monolog (~120 Wörter) eines verlorenen Koffers am Flughafen.",
    "Schreib eine Ode (~8 Zeilen) an die Deutsche Bahn.",
    "Verfasse eine Nachricht (~80 Wörter) aus dem Jahr 2124 an jemanden heute.",
    "Schreib einen Dialog (~100 Wörter) zwischen einem Buch und einem E-Reader.",
    "Verfasse eine kurze Fabel (~120 Wörter) mit Moral über einen faulen Fuchs.",
    "Schreib eine Stellenanzeige (~80 Wörter) für einen 'Professionellen Wolkenbeobachter'.",
    "Verfasse einen Reisebericht (~120 Wörter) über den ersten Tag auf dem Mars.",
    "Schreib eine Beschwerde (~100 Wörter) eines Fisches über die Wasserqualität.",
    "Verfasse ein Haiku-Trio (3 Haikus) über die vier Jahreszeiten.",
    "Schreib eine kurze Dystopie (~150 Wörter) in der Bücher verboten sind.",
    "Verfasse einen Tagebucheintrag (~120 Wörter) eines Leuchtturms.",
    "Schreib eine Werbebroschüre (~100 Wörter) für die Stadt 'Langweilingen'.",
    "Verfasse einen Dialog (~100 Wörter) zwischen einem Bleistift und einem Radiergummi.",
]

# ═══════════════════════════════════════════════════════════════════════════
# SMALLTALK (new category — 1000 target)
# ═══════════════════════════════════════════════════════════════════════════

SMALLTALK_GREETINGS = [
    "Hallo!", "Hi!", "Hey!", "Guten Morgen!", "Guten Tag!", "Guten Abend!",
    "Moin!", "Servus!", "Grüß Gott!", "Grüezi!", "Na?", "Hey, wie geht's?",
    "Hallo, bist du da?", "Guten Morgen, wie geht es dir heute?",
    "Hi, ich bin neu hier!", "Hallo! Ich habe eine Frage.",
    "Guten Abend! Schön dich zu treffen.", "Hey, alles klar bei dir?",
]

SMALLTALK_THANKS = [
    "Danke!", "Vielen Dank!", "Dankeschön!", "Danke dir!",
    "Super, danke!", "Perfekt, danke für die Hilfe!",
    "Das war sehr hilfreich, danke!", "Merci!",
    "Danke, das hat mir weitergeholfen!", "Top, danke!",
    "Besten Dank!", "Ich danke dir vielmals!",
]

SMALLTALK_FAREWELL = [
    "Tschüss!", "Bis bald!", "Auf Wiedersehen!", "Ciao!",
    "Schönen Tag noch!", "Bis zum nächsten Mal!", "Mach's gut!",
    "Gute Nacht!", "Schlaf gut!", "Bis dann!",
    "Ich muss jetzt los, tschüss!", "Schönen Abend noch!",
    "Vielen Dank und tschüss!", "Bis morgen!",
]

SMALLTALK_HOWRU = [
    "Wie geht es dir?", "Wie geht's?", "Alles gut bei dir?",
    "Was machst du so?", "Wie läuft's?", "Alles klar?",
    "Geht's dir gut?", "Wie ist dein Tag so?",
    "Na, wie läuft der Tag?", "Wie fühlst du dich heute?",
]

SMALLTALK_CASUAL = [
    "Langweilig heute, oder?", "Das Wetter ist heute super!",
    "Regen schon wieder...", "Endlich Freitag!",
    "Ich brauche Kaffee.", "Heute ist ein guter Tag!",
    "Ich bin müde.", "Morgen ist Feiertag!",
    "Hast du schon gegessen?", "Was gibt's Neues?",
    "Ich hab gerade ein tolles Buch gelesen.",
    "Kennst du einen guten Film?", "Wie war dein Wochenende?",
    "Hast du Pläne für heute Abend?", "Das Essen hier ist gut!",
    "Ich lerne gerade Deutsch.", "Bist du auch müde?",
    "Die Zeit vergeht so schnell!", "Morgen wird besser!",
    "Ich hatte einen stressigen Tag.", "Magst du Kaffee oder Tee?",
    "Wie spät ist es?", "Was ist dein Lieblingsgericht?",
    "Ich hätte gerne Pizza.", "Es ist so kalt heute!",
    "Endlich Sommer!", "Kannst du mir einen Witz erzählen?",
    "Erzähl mir was Lustiges!", "Was ist dein Hobby?",
    "Magst du Musik?", "Was hörst du gerade?",
]

SMALLTALK_ABOUT_AI = [
    "Was bist du eigentlich?", "Bist du eine KI?",
    "Hast du Gefühle?", "Kannst du denken?",
    "Bist du intelligent?", "Wer hat dich gemacht?",
    "Wie alt bist du?", "Hast du einen Namen?",
    "Was kannst du alles?", "Bist du besser als ChatGPT?",
    "Träumst du?", "Was ist deine Lieblingsfarbe?",
    "Magst du Menschen?", "Hast du Freunde?",
    "Bist du manchmal traurig?", "Was machst du wenn niemand mit dir redet?",
    "Wie lernst du neue Dinge?", "Vergisst du manchmal etwas?",
    "Hast du Angst vor etwas?", "Was ist dein Lieblingswort?",
]

# ═══════════════════════════════════════════════════════════════════════════
# ANLEITUNG (how-to) — new category
# ═══════════════════════════════════════════════════════════════════════════

ANLEITUNG_TASKS = [
    # Alltag
    "Wie kocht man perfekte Spaghetti Carbonara?",
    "Wie bügelt man ein Hemd richtig?",
    "Wie wechselt man einen Fahrradreifen?",
    "Wie pflegt man Zimmerpflanzen richtig?",
    "Wie entfernt man Rotweinflecken aus einem weißen Hemd?",
    "Wie macht man selbstgemachtes Brot?",
    "Wie räumt man einen Kleiderschrank effizient auf?",
    "Wie reinigt man eine Kaffeemaschine?",
    "Wie kocht man eine gute Gemüsebrühe?",
    "Wie packt man einen Koffer platzsparend?",
    "Wie macht man perfekten Espresso mit einer Siebträgermaschine?",
    "Wie schärft man ein Küchenmesser?",
    "Wie faltet man ein Spannbettlaken?",
    "Wie repariert man einen tropfenden Wasserhahn?",
    "Wie streicht man eine Wand gleichmäßig?",
    # Digital
    "Wie erstellt man ein sicheres Passwort?",
    "Wie richtet man 2-Faktor-Authentifizierung ein?",
    "Wie erstellt man ein Backup vom Computer?",
    "Wie macht man einen Screenshot auf Windows?",
    "Wie schreibt man eine professionelle E-Mail?",
    "Wie erstellt man eine Website mit HTML und CSS?",
    "Wie installiert man Python auf Windows?",
    "Wie nutzt man Git für Versionskontrolle?",
    "Wie erstellt man eine virtuelle Umgebung in Python?",
    "Wie richtet man SSH-Keys ein?",
    "Wie konfiguriert man einen VPN?",
    "Wie macht man einen Podcast?",
    "Wie erstellt man einen YouTube-Kanal?",
    "Wie optimiert man die Akkulaufzeit des Laptops?",
    "Wie bereinigt man den Speicher auf dem Smartphone?",
    # Beruf
    "Wie schreibt man einen Lebenslauf?",
    "Wie bereitet man sich auf ein Vorstellungsgespräch vor?",
    "Wie hält man eine gute Präsentation?",
    "Wie plant man ein Projekt mit Kanban?",
    "Wie schreibt man ein Anschreiben für eine Bewerbung?",
    "Wie führt man ein effektives Meeting?",
    "Wie gibt man konstruktives Feedback?",
    "Wie verhandelt man eine Gehaltserhöhung?",
    "Wie organisiert man seinen Arbeitstag?",
    "Wie schreibt man ein Protokoll?",
    # Gesundheit
    "Wie fängt man mit regelmäßigem Sport an?",
    "Wie meditiert man als Anfänger?",
    "Wie verbessert man seine Schlafqualität?",
    "Wie dehnt man sich richtig vor dem Sport?",
    "Wie erstellt man einen Ernährungsplan?",
    # Bildung
    "Wie lernt man effektiv für eine Prüfung?",
    "Wie liest man ein Fachbuch effizient?",
    "Wie schreibt man eine wissenschaftliche Arbeit?",
    "Wie macht man gute Notizen?",
    "Wie lernt man eine neue Sprache?",
    "Wie erstellt man Lernkarten (Anki)?",
    # Finanzen
    "Wie erstellt man ein Haushaltsbudget?",
    "Wie macht man seine Steuererklärung?",
    "Wie eröffnet man ein Depot für Aktien?",
    "Wie spart man effektiv Geld?",
    "Wie funktioniert ein ETF-Sparplan?",
    # Kreativ
    "Wie zeichnet man ein Portrait?",
    "Wie schreibt man einen Song?",
    "Wie fotografiert man bei wenig Licht?",
    "Wie strickt man einen Schal?",
    "Wie macht man einen Origami-Kranich?",
]

# ═══════════════════════════════════════════════════════════════════════════
# Generation engine
# ═══════════════════════════════════════════════════════════════════════════

def generate_records(rng: random.Random) -> list[dict]:
    records: list[dict] = []
    n = 0

    def add(task_type: str, user_prompt: str, max_tokens: int | None = None) -> None:
        nonlocal n
        n += 1
        rec = {
            "id": f"dflow_{n:05d}",
            "task_type": task_type,
            "system_prompt": SYSTEM_PROMPTS[task_type],
            "user_prompt": user_prompt,
        }
        if max_tokens is not None:
            rec["max_tokens"] = max_tokens
        records.append(rec)

    # ── code_explain: 2000 ──────────────────────────────────────────
    framings = [
        "Erkläre Schritt für Schritt was dieser Python-Code macht: ",
        "Was ist die Ausgabe und warum: ",
        "Erkläre einem Python-Anfänger was hier passiert: ",
        "Welches Idiom verwendet dieser Code: ",
        "Erkläre kurz die Wirkung: ",
        "Beschreibe was dieser Code tut und welche Edge-Cases zu beachten sind: ",
        "Was ist gut/schlecht an diesem Code: ",
        "Erkläre die Datenstruktur die hier entsteht: ",
        "Was macht dieses Snippet und was gibt es zurück: ",
        "In welchem Kontext würde man diesen Code verwenden: ",
        "Gibt es Probleme mit diesem Code? Erkläre: ",
        "Welche Python-Version braucht man für diesen Code: ",
    ]
    target = n + 2000
    while n < target:
        snippet = rng.choice(PYTHON_SNIPPETS)
        framing = rng.choice(framings)
        add("code_explain", f"{framing}`{snippet}`")

    # ── code_implementation: 1500 ───────────────────────────────────
    target = n + 1500
    while n < target:
        task = rng.choice(IMPL_TASKS)
        add("code_implementation", task)

    # ── code_refactoring: 800 ───────────────────────────────────────
    refactor_framings = [
        "Refaktoriere zu idiomatischem Python:\n",
        "Mach den Code pythonischer:\n",
        "Vereinfache diesen Code:\n",
        "Schreib das eleganter:\n",
    ]
    target = n + 800
    while n < target:
        task = rng.choice(REFACTOR_TASKS)
        framing = rng.choice(refactor_framings)
        add("code_refactoring", f"{framing}{task}")

    # ── code_debug_fix: 700 ─────────────────────────────────────────
    debug_framings = [
        "Find the bug:",
        "Wo ist der Fehler?",
        "Was stimmt nicht an diesem Code?",
        "Finde und behebe den Bug:",
    ]
    target = n + 700
    while n < target:
        prefix_orig, code = rng.choice(DEBUG_TASKS)
        prefix = rng.choice(debug_framings)
        add("code_debug_fix", f"{prefix}\n{code}")

    # ── math_word_problem: 2000 ─────────────────────────────────────
    target = n + 2000
    attempts = 0
    while n < target and attempts < 50000:
        attempts += 1
        tmpl_idx = rng.randint(0, len(MATH_TEMPLATES) - 1)
        template, frac_choices = MATH_TEMPLATES[tmpl_idx]
        kwargs: dict[str, object] = {}
        for var, (lo, hi) in MATH_VAR_RANGES.items():
            if "{" + var + "}" in template:
                kwargs[var] = rng.randint(lo, hi)
        if "{frac}" in template and frac_choices:
            kwargs["frac"] = rng.choice(frac_choices)
        try:
            prompt = template.format(**kwargs)
        except KeyError:
            continue
        add("math_word_problem", prompt)

    # ── step_by_step_reason: 1500 ──────────────────────────────────
    target = n + 1500
    while n < target:
        task = rng.choice(REASONING_TASKS)
        add("step_by_step_reason", task, max_tokens=1500)

    # ── concept_explain: 2500 ──────────────────────────────────────
    concept_framings = [
        "Erkläre {} einem interessierten Laien.",
        "Was ist {}? (mit Beispiel)",
        "Erkläre {} mit konkretem Beispiel.",
        "Was bedeutet '{}' und warum ist es relevant?",
        "Worum geht es bei '{}'? Erkläre kurz.",
        "Erkläre {} so dass ein Schüler es versteht.",
        "Was versteht man unter '{}'?",
        "Erkläre den Begriff '{}' mit Alltagsbeispiel.",
    ]
    target = n + 2500
    while n < target:
        concept = rng.choice(CONCEPTS_DE)
        f = rng.choice(concept_framings)
        add("concept_explain", f.format(concept))

    # ── factual_qa: 2500 ───────────────────────────────────────────
    target = n + 2500
    while n < target:
        q = rng.choice(FACTS_DE)
        add("factual_qa", q)

    # ── de_deep_knowledge: 2000 ────────────────────────────────────
    de_deep_all = (
        DE_LEGAL_QA + DE_HISTORY_QA + DE_LITERATURE_QA
        + DE_GEOGRAPHY_QA + DE_LANGUAGE_QA + DE_DACH_QA
    )
    target = n + 2000
    while n < target:
        q = rng.choice(de_deep_all)
        if rng.random() < 0.3:
            add("concept_explain", q)
        else:
            add("factual_qa", q)

    # ── honest_refusal: 800 ────────────────────────────────────────
    target = n + 800
    while n < target:
        q = rng.choice(REFUSAL_TASKS)
        add("honest_refusal", q)

    # ── translation: 800 ───────────────────────────────────────────
    target = n + 800
    while n < target:
        direction, sentence = rng.choice(TRANSLATION_TASKS)
        add("translation", f"Übersetze {direction}: '{sentence}'")

    # ── creative_writing: 800 ──────────────────────────────────────
    target = n + 800
    while n < target:
        task = rng.choice(CREATIVE_TASKS)
        add("creative_writing", task)

    # ── smalltalk: 1000 ────────────────────────────────────────────
    smalltalk_all = (
        SMALLTALK_GREETINGS + SMALLTALK_THANKS + SMALLTALK_FAREWELL
        + SMALLTALK_HOWRU + SMALLTALK_CASUAL + SMALLTALK_ABOUT_AI
    )
    target = n + 1000
    while n < target:
        q = rng.choice(smalltalk_all)
        add("smalltalk", q)

    # ── anleitung: 1100 ────────────────────────────────────────────
    anleitung_framings = [
        "{}",
        "Erkläre Schritt für Schritt: {}",
        "Anleitung: {}",
        "Kannst du mir erklären: {}",
        "Ich brauche eine Anleitung: {}",
    ]
    target = n + 1100
    while n < target:
        task = rng.choice(ANLEITUNG_TASKS)
        framing = rng.choice(anleitung_framings)
        add("anleitung", framing.format(task))

    return records


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=44)
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
    total = len(records)
    print(f"{'=' * 60}")
    print(f"Generated {total:,} prompts -> {args.output}")
    print(f"{'=' * 60}")
    for tt, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {tt:25s} {c:>5d}  ({100*c/total:5.1f}%)")
    n_capped = sum(1 for r in records if "max_tokens" in r)
    print(f"\nRecords with max_tokens cap: {n_capped}")
    print(f"Unique user prompts:         {len(set(r['user_prompt'] for r in records)):,}")


if __name__ == "__main__":
    main()
