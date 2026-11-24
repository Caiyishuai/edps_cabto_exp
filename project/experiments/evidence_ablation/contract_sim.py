"""Map a judge verdict to a contract-update outcome, then score a method.

Outcome classes (relative to the predicate's ground truth):
  truth=True  -> supported = correct_accept, violated = wrong_tighten (skill
                 wrongly excluded), insufficient = abstain
  truth=False -> violated = correct_tighten, supported = wrong_accept (contract
                 leak), insufficient = abstain

Task coverage counts only add predicates: an episode (task) is executable iff
all its add predicates are actually True, and is lost if any add predicate is
wrongly tightened.
"""

from __future__ import annotations

from evidence import Episode, FrameArchive
from judge import (
    VERDICT_INSUFFICIENT,
    VERDICT_SUPPORTED,
    VERDICT_VIOLATED,
    EvidenceSession,
    judge,
)

OUTCOMES = ("correct", "wrong_tighten", "wrong_accept", "abstain")


def classify(verdict: str, truth: bool) -> str:
    """Outcome is relative to the predicate's ground truth only:
    correct (verdict matches truth), wrong_tighten (truth=True judged violated:
    a working skill is wrongly excluded), wrong_accept (truth=False judged
    supported: a broken effect leaks through), abstain (insufficient)."""
    if verdict == VERDICT_INSUFFICIENT:
        return "abstain"
    if truth:
        return "correct" if verdict == VERDICT_SUPPORTED else "wrong_tighten"
    return "correct" if verdict == VERDICT_VIOLATED else "wrong_accept"


def visible_reads(predicate: str, frames) -> list[bool]:
    return [frame.readout[predicate] for frame in frames if predicate in frame.visible]


def evaluate_episode(
    archive: FrameArchive,
    episode: Episode,
    method: dict,
) -> tuple[list[dict], int, int, int]:
    session = EvidenceSession(archive, episode, method)
    records: list[dict] = []
    for predicate in episode.predicates:
        verdict = judge(predicate, session.frames)
        low_confidence = len(visible_reads(predicate, session.frames)) < archive.min_frames
        if session.can_inspect() and (verdict == VERDICT_INSUFFICIENT or low_confidence):
            if session.inspect(predicate):
                verdict = judge(predicate, session.frames)
        records.append(
            {
                "predicate": predicate,
                "verdict": verdict,
                "truth": episode.truth[predicate],
                "outcome": classify(verdict, episode.truth[predicate]),
            }
        )
    evidence_frames = len(session.frames)
    inspect_frames = session.inspect_frames_added
    inspect_calls = sum(session.inspect_counts.values())
    return records, evidence_frames, inspect_frames, inspect_calls


def summarize(
    archive: FrameArchive,
    method: dict,
    *,
    task_coverage_denominator: int | None = None,
) -> dict:
    counts = {key: 0 for key in OUTCOMES}
    evidence_frames_total = 0
    inspect_frames_total = 0
    inspect_calls_total = 0
    per_episode = []
    executable = 0
    retained = 0
    for episode in archive.episodes.values():
        records, evidence_frames, inspect_frames, inspect_calls = evaluate_episode(
            archive, episode, method
        )
        for record in records:
            counts[record["outcome"]] += 1
        evidence_frames_total += evidence_frames
        inspect_frames_total += inspect_frames
        inspect_calls_total += inspect_calls
        per_episode.append(
            {
                "episode": episode.episode_id,
                "records": records,
                "evidence_frames": evidence_frames,
                "inspect_frames": inspect_frames,
                "inspect_calls": inspect_calls,
            }
        )
        # Task coverage: an episode is executable iff all add predicates are True.
        add_true = all(episode.truth[key] for key in archive.add)
        if add_true:
            executable += 1
            add_outcomes = {
                record["outcome"]
                for record in records
                if record["predicate"] in archive.add
            }
            if "wrong_tighten" not in add_outcomes:
                retained += 1
    coverage = None
    if executable:
        coverage = retained / executable
    elif task_coverage_denominator is not None:
        coverage = None
    return {
        "outcomes": counts,
        "evidence_frames_total": evidence_frames_total,
        "inspect_frames_total": inspect_frames_total,
        "inspect_calls_total": inspect_calls_total,
        "task_coverage": coverage,
        "executable_tasks": executable,
        "retained_tasks": retained,
        "per_episode": per_episode,
    }
