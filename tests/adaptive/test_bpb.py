"""Unit tests for the torch-free parts of the bpb module."""

import unittest

from auralis.adaptive.bpb import bits_per_byte, bpb_gap, combine_extra_metrics


class TestBpb(unittest.TestCase):
    def test_bits_per_byte_matches_measured(self):
        # Values measured in-container on curated_40b.
        self.assertAlmostEqual(bits_per_byte(5.216, 0.1990), 1.498, places=2)
        self.assertAlmostEqual(bits_per_byte(8.283, 0.2338), 2.794, places=2)

    def test_bpb_gap(self):
        self.assertAlmostEqual(bpb_gap({"en": 1.498, "de": 2.794}), 2.794 / 1.498, places=4)
        self.assertEqual(bpb_gap({"en": 1.498}), 1.0)        # need >=2 langs
        self.assertEqual(bpb_gap({}), 1.0)

    def test_combine_none_and_single(self):
        self.assertIsNone(combine_extra_metrics(None, None))
        f = lambda s: {"a": 1.0}
        self.assertIs(combine_extra_metrics(f, None), f)     # single passes through

    def test_combine_merges(self):
        a = lambda s: {"a": float(s)}
        b = lambda s: {"b": 2.0}
        merged = combine_extra_metrics(a, b)
        self.assertEqual(merged(3), {"a": 3.0, "b": 2.0})

    def test_combine_isolates_failure(self):
        good = lambda s: {"ok": 1.0}
        def boom(s):
            raise RuntimeError("nope")
        merged = combine_extra_metrics(good, boom)
        out = merged(0)
        self.assertEqual(out["ok"], 1.0)                     # good one survived
        self.assertTrue(any(k.startswith("extra_metrics_error") for k in out))


if __name__ == "__main__":
    unittest.main()
