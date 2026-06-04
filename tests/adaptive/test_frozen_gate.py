import unittest

from auralis.adaptive.frozen_gate import FrozenGateLiveEvaluator, summarize_frozen_results


class FrozenGateMetricsTests(unittest.TestCase):
    def test_summarizes_target_and_retention_separately(self):
        metrics = summarize_frozen_results(
            [
                {"split": "target", "semantic_score": 1.0},
                {"split": "target", "semantic_score": 0.0},
                {"split": "retention", "semantic_score": 1.0},
                {"split": "retention", "semantic_score": 1.0},
            ]
        )

        self.assertEqual(metrics["frozen_target_pass"], 0.5)
        self.assertEqual(metrics["frozen_target_failures"], 1.0)
        self.assertEqual(metrics["frozen_retention_pass"], 1.0)
        self.assertEqual(metrics["frozen_retention_failures"], 0.0)
        self.assertEqual(metrics["frozen_promotable"], 0.0)

    def test_promotable_only_when_both_splits_are_clean(self):
        metrics = summarize_frozen_results(
            [
                {"split": "target", "semantic_score": 1.0},
                {"split": "retention", "semantic_score": 1.0},
            ]
        )

        self.assertEqual(metrics["frozen_target_pass"], 1.0)
        self.assertEqual(metrics["frozen_retention_pass"], 1.0)
        self.assertEqual(metrics["frozen_promotable"], 1.0)

    def test_every_n_evals_reuses_previous_metrics(self):
        evaluator = object.__new__(FrozenGateLiveEvaluator)
        evaluator.every_n_evals = 3
        evaluator._calls = 0
        evaluator._last_metrics = None
        evaluator.probes = [object()]
        evaluator.tok = None
        evaluator.evaluate_answer = None
        evaluator._write_trace = lambda *args, **kwargs: None
        calls = {"generate": 0}

        def fake_generate(_prompt_ids):
            calls["generate"] += 1
            return "ok"

        evaluator._generate = fake_generate
        evaluator.probes = [type("Probe", (), {"prompt": "p"})()]
        evaluator.tok = type("Tok", (), {"encode": lambda self, x: [1], "chat_prompt": lambda self, x: x})()
        evaluator.evaluate_answer = lambda probe, answer: {"split": "target", "semantic_score": 1.0}

        first = evaluator(10)
        second = evaluator(20)
        third = evaluator(30)

        self.assertEqual(first["frozen_gate_ran"], 1.0)
        self.assertEqual(second["frozen_gate_ran"], 0.0)
        self.assertEqual(third["frozen_gate_ran"], 1.0)
        self.assertEqual(calls["generate"], 2)


if __name__ == "__main__":
    unittest.main()
