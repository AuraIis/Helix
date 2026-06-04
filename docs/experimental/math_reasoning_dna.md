# Math / Reasoning DNA Idea

Status: parked idea for a later experiment.

## Core idea

Auralis should not only memorize answers like `5 + 5 = 10`. For arithmetic,
logic, and small algorithmic tasks, it should learn a reusable internal working
method: represent the problem, manipulate the representation, verify the result,
then answer.

This is the technical version of a human counting with fingers, cookies, or a
mental workspace.

## Why the tokenizer is not enough

The tokenizer can make useful units visible, for example:

```text
5
+
5
=
```

or:

```text
5 Kekse + 5 Kekse
```

But the tokenizer does not reason. The reasoning has to be learned by the model
weights, or supported by an external verifier/checker. The tokenizer can only
make the learning problem easier by exposing stable symbols and formats.

## Training format candidates

### Concrete quantity representation

```text
<math_work>
Aufgabe: 5 Kekse + 5 Kekse

Menge A:
Keks Keks Keks Keks Keks

Menge B:
Keks Keks Keks Keks Keks

Zusammen:
Keks Keks Keks Keks Keks Keks Keks Keks Keks Keks

Zaehlen:
1 2 3 4 5 6 7 8 9 10
</math_work>
<answer>10</answer>
```

### Tally representation

```text
<count_work>
||||| + ||||| = ||||||||||
</count_work>
<answer>10</answer>
```

### Place-value representation

```text
<math_work>
Aufgabe: 17 + 25

Zerlege:
17 = 1 Zehner + 7 Einer
25 = 2 Zehner + 5 Einer

Einer:
7 + 5 = 12 -> 2 Einer, Uebertrag 1

Zehner:
1 + 2 + 1 = 4
</math_work>
<answer>42</answer>
```

### Error recognition / verifier examples

```text
<math_check>
Aufgabe: 17 + 25
Antwort: 43
Bewertung: falsch
Korrektur: 42
Grund: Die Einerstelle ist 2, nicht 3.
</math_check>
```

## Experiment sketch

1. Generate a small Math-DNA dataset with addition, subtraction, multiplication,
   comparisons, simple word problems, and explicit wrong-answer examples.
2. Mix the examples into a tiny/100M pretraining or continued-training run.
3. Compare against a baseline with the same token budget but normal answer-only
   math examples.
4. Evaluate exact arithmetic, robustness to changed wording, repetition rate,
   and ability to reject wrong answers.

## Hypothesis

The useful signal is not the final answer token. The useful signal is the stable
intermediate workspace:

```text
problem -> representation -> operation -> check -> answer
```

If this works, later Auralis can combine:

- generator model for the reasoning trace,
- verifier/checker for correctness,
- optional Python/tool execution for exact math/code,
- final answer distilled back into natural language.

## Go / No-Go

Use it in larger training only if it improves arithmetic and logical transfer
without making normal text overly tag-heavy or repetitive.
