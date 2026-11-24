"""Lossless frame archive + model-directed inspection (VISTA-inspired, offline).

Freeze the visibility model and per-episode predicate truth in protocol.json.
This module only carries the archive and the retrieval interface. The oracle
truth used for scoring lives in evaluate.py and is never shown to any method.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


class Frame:
    """One archived stage view. ``readout`` is what a judge is allowed to read;
    it is the value that would be visually read off this frame (truth, unless
    readout_noise flips it)."""

    __slots__ = ("stage", "visible", "readout")

    def __init__(self, stage: str, visible: set[str], readout: dict[str, bool]) -> None:
        self.stage = stage
        self.visible = frozenset(visible)
        self.readout = dict(readout)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Frame) and self.stage == other.stage

    def __hash__(self) -> int:
        return hash(self.stage)

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"Frame({self.stage}, visible={sorted(self.visible)})"


class Episode:
    __slots__ = ("episode_id", "truth", "frames", "predicates")

    def __init__(self, episode_id: str, truth: dict[str, bool], frames: dict[str, Frame], predicates: list[str]) -> None:
        self.episode_id = episode_id
        self.truth = dict(truth)
        self.frames = dict(frames)
        self.predicates = list(predicates)

    def frames_for(self, stages: list[str]) -> list[Frame]:
        return [self.frames[stage] for stage in stages if stage in self.frames]


class FrameArchive:
    """Holds all episodes plus the frozen visibility index used by inspect."""

    def __init__(self, protocol: dict[str, Any]) -> None:
        self.stages: list[str] = list(protocol["stages"])
        add = list(protocol["predicates"]["add"])
        del_ = list(protocol["predicates"]["del"])
        self.predicates = add + del_
        self.add = add
        self.del_ = del_
        self.visibility: dict[str, list[str]] = {
            key: list(value) for key, value in protocol["visibility"].items()
        }
        self.noise = float(protocol.get("readout_noise", 0.0))
        self.min_frames = int(protocol.get("min_frames_for_verdict", 2))
        self.episodes: dict[str, Episode] = {}
        rng = random.Random(int(protocol.get("seed", 0)))
        for raw in protocol["episodes"]:
            self.episodes[raw["id"]] = self._build_episode(raw, rng)

    def _build_episode(self, raw: dict[str, Any], rng: random.Random) -> Episode:
        truth = {key: bool(value) for key, value in raw["truth"].items()}
        missing = [key for key in self.predicates if key not in truth]
        if missing:
            raise ValueError(f"{raw['id']}: missing truth for {missing}")
        frames: dict[str, Frame] = {}
        for stage in self.stages:
            visible = {
                predicate
                for predicate in self.predicates
                if stage in self.visibility[predicate]
            }
            readout = {}
            for predicate in visible:
                value = truth[predicate]
                if self.noise > 0 and rng.random() < self.noise:
                    value = not value
                readout[predicate] = value
            frames[stage] = Frame(stage, visible, readout)
        return Episode(raw["id"], truth, frames, self.predicates)

    def stages_for(self, predicate: str) -> list[str]:
        return self.visibility[predicate]


def load_protocol(path: Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else Path(__file__).with_name("protocol.json")
    return json.loads(path.read_text(encoding="utf-8"))


def build_archive(protocol: dict[str, Any] | None = None) -> FrameArchive:
    return FrameArchive(protocol if protocol is not None else load_protocol())
