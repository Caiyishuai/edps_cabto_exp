"""Pure-logic regression tests for the evidence-ablation experiment.

No Mujoco, no VLM, no network. Verifies the frozen protocol, the readout
model, the three-state judge, the contract-outcome mapping, and the structural
properties that make methods A/B/C differ.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract_sim import classify, summarize  # noqa: E402
from evidence import Frame, build_archive, load_protocol  # noqa: E402
from judge import (  # noqa: E402
    VERDICT_INSUFFICIENT,
    VERDICT_SUPPORTED,
    VERDICT_VIOLATED,
    EvidenceSession,
    judge,
)


class EvidenceAblationTests(unittest.TestCase):
    def protocol(self, **overrides):
        protocol = load_protocol()
        protocol.update(overrides)
        return protocol

    def archive(self, noise=0.0):
        return build_archive(self.protocol(readout_noise=noise))

    def test_protocol_structure(self):
        archive = self.archive()
        self.assertEqual(archive.add, ["holding(A)", "clear(S)"])
        self.assertEqual(archive.del_, ["empty()", "ontable(A)"])
        # The crux of method A's disadvantage: holding is invisible before/after.
        self.assertNotIn("before", archive.stages_for("holding(A)"))
        self.assertNotIn("after", archive.stages_for("holding(A)"))
        self.assertIn("close_gripper", archive.stages_for("holding(A)"))
        self.assertIn("lift", archive.stages_for("holding(A)"))

    def test_readout_matches_truth_at_zero_noise(self):
        archive = self.archive(noise=0.0)
        for episode in archive.episodes.values():
            for frame in episode.frames.values():
                for predicate in frame.visible:
                    self.assertEqual(frame.readout[predicate], episode.truth[predicate])

    def test_judge_three_states(self):
        ep = self.archive(noise=0.0).episodes["e2_fail_holding"]
        close, lift = ep.frames["close_gripper"], ep.frames["lift"]
        before, after = ep.frames["before"], ep.frames["after"]
        self.assertEqual(judge("holding(A)", [close, lift]), VERDICT_VIOLATED)
        self.assertEqual(judge("holding(A)", []), VERDICT_INSUFFICIENT)
        self.assertEqual(judge("clear(S)", [before, after]), VERDICT_SUPPORTED)

    def test_judge_majority_and_tie(self):
        yes = Frame("s1", {"p"}, {"p": True})
        no = Frame("s2", {"p"}, {"p": False})
        yes2 = Frame("s3", {"p"}, {"p": True})
        self.assertEqual(judge("p", [yes, no, yes2]), VERDICT_SUPPORTED)
        self.assertEqual(judge("p", [yes, no]), VERDICT_INSUFFICIENT)
        self.assertEqual(judge("q", [yes]), VERDICT_INSUFFICIENT)

    def test_classify(self):
        self.assertEqual(classify(VERDICT_SUPPORTED, True), "correct")
        self.assertEqual(classify(VERDICT_VIOLATED, True), "wrong_tighten")
        self.assertEqual(classify(VERDICT_VIOLATED, False), "correct")
        self.assertEqual(classify(VERDICT_SUPPORTED, False), "wrong_accept")
        self.assertEqual(classify(VERDICT_INSUFFICIENT, True), "abstain")

    def test_method_A_holding_always_insufficient(self):
        protocol = self.protocol(readout_noise=0.3)
        archive = build_archive(protocol)
        for episode in archive.episodes.values():
            session = EvidenceSession(archive, episode, protocol["methods"]["A"])
            self.assertEqual(judge("holding(A)", session.frames), VERDICT_INSUFFICIENT)

    def test_method_B_sees_every_predicate(self):
        protocol = self.protocol(readout_noise=0.3)
        archive = build_archive(protocol)
        episode = archive.episodes["e1_success"]
        session = EvidenceSession(archive, episode, protocol["methods"]["B"])
        self.assertEqual(len(session.frames), 6)
        for predicate in archive.predicates:
            visible = [f for f in session.frames if predicate in f.visible]
            self.assertGreaterEqual(len(visible), 3, predicate)

    def test_method_C_inspects_to_recover_holding(self):
        protocol = self.protocol(readout_noise=0.0)
        archive = build_archive(protocol)
        episode = archive.episodes["e1_success"]
        session = EvidenceSession(archive, episode, protocol["methods"]["C"])
        self.assertEqual(judge("holding(A)", session.frames), VERDICT_INSUFFICIENT)
        self.assertEqual(session.inspect("holding(A)"), 3)
        self.assertEqual(judge("holding(A)", session.frames), VERDICT_SUPPORTED)

    def test_method_C_inspect_is_bounded(self):
        protocol = self.protocol(readout_noise=0.0)
        archive = build_archive(protocol)
        episode = archive.episodes["e1_success"]
        session = EvidenceSession(archive, episode, protocol["methods"]["C"])
        self.assertEqual(session.inspect("holding(A)"), 3)
        self.assertEqual(session.inspect("holding(A)"), 0)  # already inspected
        self.assertEqual(session.inspect_counts["holding(A)"], 1)

    def test_task_coverage_at_zero_noise(self):
        protocol = self.protocol(readout_noise=0.0)
        archive = build_archive(protocol)
        for key in ("A", "B", "C"):
            summary = summarize(archive, protocol["methods"][key])
            self.assertEqual(summary["executable_tasks"], 3)
            self.assertEqual(summary["task_coverage"], 1.0)

    def test_summarize_is_reproducible(self):
        protocol = self.protocol(readout_noise=0.3)
        first = summarize(build_archive(protocol), protocol["methods"]["C"])
        second = summarize(build_archive(protocol), protocol["methods"]["C"])
        self.assertEqual(first["outcomes"], second["outcomes"])
        self.assertEqual(first["task_coverage"], second["task_coverage"])
        self.assertEqual(first["evidence_frames_total"], second["evidence_frames_total"])

    def test_A_abstains_more_than_B_at_zero_noise(self):
        protocol = self.protocol(readout_noise=0.0)
        archive = build_archive(protocol)
        a = summarize(archive, protocol["methods"]["A"])
        b = summarize(archive, protocol["methods"]["B"])
        # A cannot ever see holding -> 6 abstains; B sees it -> 0 abstains.
        self.assertEqual(a["outcomes"]["abstain"], 6)
        self.assertEqual(b["outcomes"]["abstain"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
