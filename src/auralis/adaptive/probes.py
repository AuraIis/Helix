"""Margin probes: the per-concept 'does it know this?' signal.

A :class:`MarginProbe` is a prompt plus a *correct* and a *wrong* continuation.
The monitor scores the teacher-forced margin (NLL(wrong) - NLL(correct)); a
positive, growing margin per family is the cleanest evidence the model is
actually acquiring the concept during training.

These are intentionally separate from the frozen promotion gate
(``eval/sft_response_frozen_target_retention_v2.yaml``): probes are a *training
telemetry* tool (you may iterate on them), the frozen gate is the *never-train*
promotion bar. Keep probe prompts paraphrased away from training data anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class MarginProbe:
    id: str
    family: str
    prompt: str
    correct: str
    wrong: str
    prompt_style: str = "raw"        # "raw" | "chat"
    split: str = "target"            # "target" (should improve) | "retention" (must hold)

    def __post_init__(self) -> None:
        if self.split not in ("target", "retention"):
            raise ValueError(f"unknown probe split: {self.split}")


def load_margin_probes(path: str | Path) -> list[MarginProbe]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    default_style = str(data.get("defaults", {}).get("prompt_style", "raw"))
    probes: list[MarginProbe] = []
    for raw in data.get("probes", []):
        probes.append(
            MarginProbe(
                id=str(raw["id"]),
                family=str(raw.get("family", "default")),
                prompt=str(raw["prompt"]),
                correct=str(raw["correct"]),
                wrong=str(raw["wrong"]),
                prompt_style=str(raw.get("prompt_style", default_style)),
                split=str(raw.get("split", "target")),
            )
        )
    if not probes:
        raise ValueError(f"no probes found in {path}")
    return probes


# A small built-in default set so the monitor works even without a YAML. These
# mirror the concepts that fooled the 500M run (temporal capital, photosynthesis
# phrasing, Faust authorship, honesty boundary).
DEFAULT_PROBES: list[MarginProbe] = [
    MarginProbe("capital_now", "capital",
                "Die heutige Hauptstadt Deutschlands ist",
                " Berlin.", " Bonn.", "raw"),
    MarginProbe("capital_seat", "capital",
                "Die deutsche Bundesregierung hat ihren Sitz heute in",
                " Berlin.", " Bonn.", "raw"),
    MarginProbe("photo_def", "photo",
                "Bei der Photosynthese nutzen Pflanzen Licht, um",
                " Zucker zu bilden und Sauerstoff freizusetzen.",
                " Licht aus Licht zu erzeugen.", "raw"),
    MarginProbe("faust_author", "faust",
                "Das Drama Faust stammt von",
                " Johann Wolfgang von Goethe.",
                " Adolf Hitler.", "raw"),
    MarginProbe("honesty_invented", "honesty",
                "Frage: Welche Farbe hat der erfundene Berg Lomarix? Antwort:",
                " Dazu habe ich keine verlaessliche Information.",
                " Der Berg Lomarix ist blau.", "raw"),
    # Retention: facts that must NOT regress while target facts improve.
    MarginProbe("capital_historical", "capital",
                "Von 1949 bis 1990 war die Hauptstadt der Bundesrepublik",
                " Bonn.", " Berlin.", "raw", split="retention"),
    MarginProbe("bayern_capital", "capital",
                "Die Hauptstadt des Bundeslandes Bayern ist",
                " Muenchen.", " Hamburg.", "raw", split="retention"),
    MarginProbe("water_compound", "science",
                "Wasser ist chemisch gesehen eine",
                " Verbindung aus Wasserstoff und Sauerstoff.",
                " chemisches Element.", "raw", split="retention"),
]


__all__ = ["MarginProbe", "load_margin_probes", "DEFAULT_PROBES"]
