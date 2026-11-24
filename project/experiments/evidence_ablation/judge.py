"""Three evidence-supply methods and the read-only predicate judge.

A judge is only allowed to read the ``readout`` of the frames it has been
given. It must not consult episode truth. The only difference between methods
is WHICH frames they are given; the readout model is identical.
"""

from __future__ import annotations

from evidence import Episode, FrameArchive

VERDICT_SUPPORTED = "supported"
VERDICT_VIOLATED = "violated"
VERDICT_INSUFFICIENT = "insufficient"


def judge(predicate: str, frames) -> str:
    """Return supported / violated / insufficient from supplied frames only."""
    reads = [frame.readout[predicate] for frame in frames if predicate in frame.visible]
    if not reads:
        return VERDICT_INSUFFICIENT
    trues = sum(1 for value in reads if value)
    falses = len(reads) - trues
    if trues > falses:
        return VERDICT_SUPPORTED
    if falses > trues:
        return VERDICT_VIOLATED
    return VERDICT_INSUFFICIENT


class EvidenceSession:
    """A method's working set of frames, plus its optional inspect capability."""

    def __init__(self, archive: FrameArchive, episode: Episode, method: dict) -> None:
        self.archive = archive
        self.episode = episode
        self.method = method
        self.frames = list(episode.frames_for(method["stages"]))
        self.inspect_counts: dict[str, int] = {}
        self.inspect_frames_added = 0

    def can_inspect(self) -> bool:
        return bool(self.method.get("inspect"))

    def inspect(self, predicate: str) -> int:
        """Retrieve the frames where this predicate is visible. Returns number
        of frames added (0 if not allowed, already inspected, or over budget)."""
        if not self.can_inspect():
            return 0
        limit = self.method.get("max_inspect_per_predicate", 1)
        if self.inspect_counts.get(predicate, 0) >= limit:
            return 0
        stages = self.archive.stages_for(predicate)
        added = []
        for frame in self.episode.frames_for(stages):
            if frame not in self.frames:
                added.append(frame)
        self.frames.extend(added)
        self.inspect_counts[predicate] = self.inspect_counts.get(predicate, 0) + 1
        self.inspect_frames_added += len(added)
        return len(added)


def make_session(method_key: str, method: dict, archive: FrameArchive, episode: Episode) -> EvidenceSession:
    return EvidenceSession(archive, episode, method)
