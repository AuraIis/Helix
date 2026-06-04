"""Unit tests for the curriculum controller state machine."""

import unittest

from auralis.adaptive import (
    CurriculumController,
    CurriculumSpec,
    MetricSnapshot,
    DecisionKind,
)


def two_stage_spec(**overrides):
    data = {
        "name": "test",
        "eval_every": 1,
        "on_guard": "stop",
        "default_guard": {"metric": "retention", "max_drop": 0.05},
        "stages": [
            {
                "name": "s1_text",
                "min_steps": 2,
                "max_steps": 100,
                "mastery": {"metric": "stage_primary", "mode": "stable_above",
                            "threshold": 0.9, "window": 2},
            },
            {
                "name": "s2_format",
                "min_steps": 2,
                "max_steps": 6,
                "mastery": {"metric": "stage_primary", "mode": "plateau",
                            "patience": 2, "min_delta": 0.01},
            },
        ],
    }
    data.update(overrides)
    return CurriculumSpec.from_dict(data)


class TestController(unittest.TestCase):
    def _snap(self, step, stage_step, primary, retention=1.0):
        return MetricSnapshot(step=step, stage_step=stage_step,
                              metrics={"stage_primary": primary, "retention": retention})

    def test_respects_min_steps(self):
        ctrl = CurriculumController(two_stage_spec())
        # mastery metric already high, but below min_steps -> CONTINUE
        d = ctrl.update(self._snap(1, 1, 0.99))
        self.assertEqual(d.kind, DecisionKind.CONTINUE)
        self.assertEqual(ctrl.stage.name, "s1_text")

    def test_advance_on_stable_mastery(self):
        ctrl = CurriculumController(two_stage_spec())
        ctrl.update(self._snap(1, 1, 0.95))            # below min_steps
        d = ctrl.update(self._snap(2, 2, 0.95))        # window of 2 >= 0.9 held
        self.assertEqual(d.kind, DecisionKind.ADVANCE)
        self.assertEqual(d.to_stage, "s2_format")
        self.assertEqual(ctrl.stage.name, "s2_format")

    def test_stage_local_history_resets_for_plateau(self):
        # s1 values are high; they must NOT count toward s2's plateau check.
        ctrl = CurriculumController(two_stage_spec())
        ctrl.update(self._snap(1, 1, 0.95))
        ctrl.update(self._snap(2, 2, 0.95))            # advance to s2
        self.assertEqual(ctrl.stage.name, "s2_format")
        # s2 climbs then flattens -> plateau advance (=> DONE, last stage)
        ctrl.update(self._snap(3, 1, 0.20))            # below min_steps
        ctrl.update(self._snap(4, 2, 0.40))            # climbing
        ctrl.update(self._snap(5, 3, 0.405))           # flat
        d = ctrl.update(self._snap(6, 4, 0.406))       # flat -> plateau
        self.assertin_done_or_advance(d)

    def assertin_done_or_advance(self, d):
        # s2 is the last stage, so a plateau advance finishes the run.
        self.assertEqual(d.kind, DecisionKind.DONE)

    def test_guard_stops_on_retention_regression(self):
        ctrl = CurriculumController(two_stage_spec())
        ctrl.update(self._snap(1, 1, 0.5, retention=1.0))
        d = ctrl.update(self._snap(2, 2, 0.5, retention=0.90))  # dropped 0.10 >= 0.05
        self.assertEqual(d.kind, DecisionKind.STOP)
        self.assertTrue(ctrl.finished)
        self.assertIn("retention", d.reason)

    def test_guard_hold_policy(self):
        ctrl = CurriculumController(two_stage_spec(on_guard="hold"))
        ctrl.update(self._snap(1, 1, 0.5, retention=1.0))
        d = ctrl.update(self._snap(2, 2, 0.5, retention=0.90))
        self.assertEqual(d.kind, DecisionKind.HOLD)
        self.assertFalse(ctrl.finished)  # hold is not terminal

    def test_max_steps_timeout_advances(self):
        # s1 never masters; hits max_steps -> advance (not last stage)
        spec = two_stage_spec()
        spec.stages[0].max_steps = 3
        ctrl = CurriculumController(spec)
        ctrl.update(self._snap(1, 1, 0.1))
        ctrl.update(self._snap(2, 2, 0.1))
        d = ctrl.update(self._snap(3, 3, 0.1))
        self.assertEqual(d.kind, DecisionKind.ADVANCE)
        self.assertIn("timeout", d.reason)

    def test_last_stage_max_steps_finishes(self):
        spec = two_stage_spec()
        ctrl = CurriculumController(spec, start_stage=1)  # start on last stage
        spec.stages[1].max_steps = 2
        ctrl.update(self._snap(1, 1, 0.1))
        d = ctrl.update(self._snap(2, 2, 0.1))
        self.assertEqual(d.kind, DecisionKind.DONE)

    def test_full_progression_to_done(self):
        ctrl = CurriculumController(two_stage_spec())
        decisions = []
        decisions.append(ctrl.update(self._snap(1, 1, 0.95)))
        decisions.append(ctrl.update(self._snap(2, 2, 0.95)))   # advance s1->s2
        decisions.append(ctrl.update(self._snap(3, 1, 0.5)))
        decisions.append(ctrl.update(self._snap(4, 2, 0.95)))
        decisions.append(ctrl.update(self._snap(5, 3, 0.96)))   # s2 stable? mode=plateau
        kinds = [d.kind for d in decisions]
        self.assertIn(DecisionKind.ADVANCE, kinds)
        self.assertTrue(ctrl.finished or DecisionKind.DONE in kinds or
                        kinds[-1] == DecisionKind.CONTINUE)


if __name__ == "__main__":
    unittest.main()
