"""Unit tests for the torch-free signal helpers."""

import unittest

from auralis.adaptive.signals import (
    ema,
    trend_slope,
    relative_improvement,
    is_plateaued,
    is_stable_above,
    detect_regression,
)


class TestSignals(unittest.TestCase):
    def test_ema_tracks_and_smooths(self):
        out = ema([0.0, 1.0, 1.0, 1.0], alpha=0.5)
        self.assertEqual(out[0], 0.0)
        self.assertAlmostEqual(out[1], 0.5)
        self.assertAlmostEqual(out[2], 0.75)
        self.assertTrue(out[3] > out[2])  # converging upward

    def test_ema_rejects_bad_alpha(self):
        with self.assertRaises(ValueError):
            ema([1.0], alpha=0.0)

    def test_trend_slope_sign(self):
        self.assertGreater(trend_slope([0.1, 0.2, 0.3, 0.4]), 0)
        self.assertLess(trend_slope([0.4, 0.3, 0.2, 0.1]), 0)
        self.assertAlmostEqual(trend_slope([0.5, 0.5, 0.5]), 0.0)
        self.assertEqual(trend_slope([0.5]), 0.0)  # too few points

    def test_relative_improvement(self):
        # latest beats the prior window max -> positive
        self.assertAlmostEqual(relative_improvement([0.1, 0.2, 0.3], window=2), 0.1)
        # latest below prior max -> negative
        self.assertAlmostEqual(relative_improvement([0.1, 0.4, 0.3], window=2), -0.1)

    def test_is_plateaued(self):
        # climbing -> not plateaued
        self.assertFalse(is_plateaued([0.1, 0.3, 0.5, 0.7], patience=2, min_delta=0.01))
        # flat after a rise -> plateaued
        self.assertTrue(is_plateaued([0.1, 0.5, 0.5, 0.5], patience=2, min_delta=0.01))
        # not enough points -> not plateaued
        self.assertFalse(is_plateaued([0.5], patience=2, min_delta=0.01))

    def test_is_stable_above(self):
        self.assertTrue(is_stable_above([0.5, 0.95, 0.96], threshold=0.9, window=2))
        # one dip inside the window -> not stable
        self.assertFalse(is_stable_above([0.95, 0.5, 0.96], threshold=0.9, window=3))
        # too few points -> not stable
        self.assertFalse(is_stable_above([0.95], threshold=0.9, window=2))

    def test_detect_regression(self):
        # dropped 0.1 below peak -> regression at max_drop 0.05
        self.assertTrue(detect_regression([0.9, 0.95, 0.85], max_drop=0.05))
        # tiny dip below threshold -> no regression
        self.assertFalse(detect_regression([0.9, 0.95, 0.93], max_drop=0.05))
        # single point -> no regression
        self.assertFalse(detect_regression([0.9], max_drop=0.05))


if __name__ == "__main__":
    unittest.main()
