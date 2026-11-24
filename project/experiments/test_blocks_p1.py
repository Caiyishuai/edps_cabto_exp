"""P1 grammar, blind input, genuine candidate execution and sequential-budget tests."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocks_p0 import ROOT, action_models, digest, policy_source, save
from blocks_p1_candidates import (CandidateRejected, compile_candidate, parse_proposal,
                                 proposal_messages, public_evidence, recipe, strict_json, validate_candidate)
from run_blocks_p1 import (BASELINE, candidate_order, protocol_at, search, trial,
                           checked_path, file_hash, load_search_gate, sum_cost)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.protocol = protocol_at(Path(__file__).with_name("blocks_p1_protocol.json"))
        self.layout = self.protocol["layouts"][0]
        self.candidate = {"id": "test", "pre_add": [{"action": "stack", "predicate": "clear", "args": ["support"]}],
                          "offset_z_m": .003, "reason": "test fixture, not model evidence"}

    def test_new_heldout_pairings_and_geometry(self):
        self.assertEqual(len([r for r in self.protocol["layouts"] if r["split"] == "heldout"]), 3)
        old = json.loads(Path(__file__).with_name("blocks_p0_protocol.json").read_text())
        self.assertFalse({(r["payload"], r["support"]) for r in old["layouts"]} &
                         {(r["payload"], r["support"]) for r in self.protocol["layouts"] if r["split"] == "heldout"})

    def test_parser_rejects_duplicates_nonfinite_trailing_and_thinking(self):
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{} trailing', '<think>{}</think>', '```json\n{}'):
            with self.subTest(raw=raw), self.assertRaises(CandidateRejected):
                strict_json(raw)

    def test_strict_fenced_json_allowed(self):
        self.assertEqual(strict_json('```json\n{"x":2}\n```'), {"x": 2})

    def test_huge_integer_and_deep_json_are_rejections(self):
        with self.assertRaises(CandidateRejected):
            validate_candidate({**self.candidate, "offset_z_m": 10 ** 1000}, self.protocol)
        for depth in (20, 2000):
            with self.subTest(depth=depth), self.assertRaises(CandidateRejected):
                strict_json("[" * depth + "0" + "]" * depth)

    def test_wrong_fields_predicates_arguments_and_numeric_types(self):
        examples = [{**self.candidate, "effects": []}, {**self.candidate, "offset_z_m": True},
                    {**self.candidate, "offset_z_m": float("inf")}, {**self.candidate, "offset_z_m": .2},
                    {**self.candidate, "offset_z_m": "0"},
                    {**self.candidate, "pre_add": [{"action": "stack", "predicate": "exec", "args": []}]},
                    {**self.candidate, "pre_add": [{"action": "stack", "predicate": "clear", "args": ["yellow"]}]}]
        for item in examples:
            with self.subTest(item=item), self.assertRaises(CandidateRejected):
                validate_candidate(item, self.protocol)

    def test_arbitrary_legal_precondition_not_just_clear(self):
        candidate = {**self.candidate, "pre_add": [{"action": "stack", "predicate": "ontable", "args": ["support"]}]}
        models, value = compile_candidate(candidate, self.protocol, self.layout, .045)
        stack = next(m for m in models if m["name"] == "stack")
        self.assertIn(f"ontable({self.layout['support']})", stack["pre"])
        self.assertNotIn(f"clear({self.layout['support']})", stack["pre"])
        self.assertEqual(value, .003)

    def test_parameter_is_replace_not_delta(self):
        _, value = compile_candidate(self.candidate, self.protocol, self.layout, .045)
        self.assertEqual(value, .003)
        _, value = compile_candidate({**self.candidate, "offset_z_m": None}, self.protocol, self.layout, .045)
        self.assertEqual(value, .045)

    def test_non_target_layers_immutable(self):
        objects = [self.layout[k] for k in ("payload", "support", "blocker", "distractor")]
        before = action_models(objects, self.layout["support"], self.layout["payload"], False)
        fixed, _ = compile_candidate(self.candidate, self.protocol, self.layout, 0.)
        for old, new in zip(before, fixed):
            self.assertEqual({k: v for k, v in old.items() if k != "pre"}, {k: v for k, v in new.items() if k != "pre"})

    def test_bad_candidates_retained_and_abstention_not_repaired(self):
        bad = {**self.candidate, "id": "bad", "offset_z_m": True}
        raw = {"decision": "propose", "reason": "r", "candidates": [bad, self.candidate, self.candidate]}
        parsed = parse_proposal({"raw": json.dumps(raw), "generation_tokens": 10}, self.protocol)
        self.assertEqual(len(parsed["accepted"]), 1)
        self.assertEqual(len(parsed["rejected"]), 2)
        abstain = parse_proposal({"raw": '{"decision":"abstain","reason":"unknown","candidates":[]}'}, self.protocol)
        self.assertEqual(abstain["accepted"], [])
        self.assertEqual(abstain["decision"], "abstain")

    def test_truncation_rejected_even_if_json_complete(self):
        raw = json.dumps({"decision": "propose", "reason": "r", "candidates": [self.candidate]})
        with self.assertRaises(CandidateRejected):
            parse_proposal({"raw": raw, "finish_reason": "length"}, self.protocol)

    def test_private_tags_do_not_reach_model(self):
        result = {"success": False, "steps": [], "initial": {"state": ["empty()"]}, "exception": "failure"}
        clean = public_evidence(result, "opaque", .045)
        polluted = {**result, "truth": "model", "initial_on": [["red", "blue"]], "candidate_precondition": "SECRET", "split": "heldout"}
        self.assertEqual(clean, public_evidence(polluted, "opaque", .045))
        messages = json.dumps(proposal_messages([clean], self.protocol))
        for key in ("target_occupied", "blocker", "candidate_precondition", "p1-h1", "generation_seeds"):
            self.assertNotIn(key, messages)


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.protocol = protocol_at(Path(__file__).with_name("blocks_p1_protocol.json"))
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "outputs", prefix="p1-test-")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.good = {"id": "good", "pre_add": [], "offset_z_m": .001, "reason": "mock"}
        self.bad = {"id": "bad", "pre_add": [], "offset_z_m": .05, "reason": "mock"}
        self.pool = [{"candidate": c, "origin": "test_not_real_LM", "rank": i} for i, c in enumerate([self.good, self.bad])]

    def mock_trial(self, protocol, candidate, layout, condition, out, render=False):
        return {"success": candidate["id"] == "good", "trial_valid": True,
                "steps": [], "simulation_seconds": 1., "wall_seconds": .1, "exception": None, "physical_rollouts": 1}

    def test_ranked_stops_after_independent_confirmation(self):
        with patch("run_blocks_p1.trial", side_effect=self.mock_trial) as mocked:
            result = search(self.protocol, self.pool, "ranked", self.path / "rank", {})
        self.assertEqual(mocked.call_count, 6)
        self.assertEqual(result["status"], "confirmed")
        self.assertEqual(result["cost"]["rollouts"], 6)
        self.assertEqual(len(result["attempts"]), 1)

    def test_enumeration_actually_evaluates_later_bad_candidate(self):
        with patch("run_blocks_p1.trial", side_effect=self.mock_trial) as mocked:
            result = search(self.protocol, self.pool, "enumerate", self.path / "all", {})
        self.assertEqual(mocked.call_count, 7)
        self.assertEqual(len(result["attempts"]), 2)
        self.assertIsNotNone(result["selected"])

    def test_enumeration_later_infrastructure_failure_keeps_confirmed_candidate(self):
        def simulate(protocol, candidate, layout, condition, out, render=False):
            result = self.mock_trial(protocol, candidate, layout, condition, out, render)
            if candidate["id"] == "bad":
                result["trial_valid"] = False
            return result
        with patch("run_blocks_p1.trial", side_effect=simulate):
            result = search(self.protocol, self.pool, "enumerate", self.path / "later-invalid", {})
        self.assertEqual(result["status"], "confirmed")
        self.assertEqual(result["search_completion_status"], "infrastructure_invalid")
        self.assertIsNotNone(result["selected"])

    def test_paths_cannot_escape_result_directory(self):
        with self.assertRaises(ValueError):
            checked_path(self.path, "../outside")

    def test_gate_duplicate_result_cannot_replace_checked_artifact(self):
        frozen = {"protocol_sha256": digest(self.protocol), "source_hashes": {}, "pools": [], "baseline_cost": {}}
        save(self.path / "pools_frozen.json", frozen)
        record = {"selected": None, "attempts": [], "status": "pool_exhausted_abstain"}
        path = self.path / "search.json"
        save(path, record)
        gate = {"pools_freeze_sha256": file_hash(self.path / "pools_frozen.json"),
                "protocol_sha256": digest(self.protocol), "source_hashes": {},
                "searches": [{"path": "search.json", "sha256": file_hash(path),
                              "result": {**record, "selected": self.pool[0], "generation_seed": 1701, "shared_baseline_cost": {}}}]}
        save(self.path / "searches_frozen.json", gate)
        with self.assertRaisesRegex(ValueError, "Duplicated gate"):
            load_search_gate(self.path, self.protocol, frozen)

    def test_all_bad_pool_abstains(self):
        with patch("run_blocks_p1.trial", side_effect=self.mock_trial):
            result = search(self.protocol, [self.pool[1]], "ranked", self.path / "bad", {})
        self.assertEqual(result["status"], "pool_exhausted_abstain")
        self.assertIsNone(result["selected"])

    def test_budget_cannot_accept_unconfirmed_candidate(self):
        protocol = {**self.protocol, "max_search_rollouts": 4}
        with patch("run_blocks_p1.trial", side_effect=self.mock_trial):
            result = search(protocol, self.pool, "ranked", self.path / "budget", {})
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertIsNone(result["selected"])
        self.assertEqual(result["cost"]["rollouts"], 4)

    def test_no_candidates_no_rollout(self):
        with patch("run_blocks_p1.trial") as mocked:
            result = search(self.protocol, [], "ranked", self.path / "empty", {})
        mocked.assert_not_called()
        self.assertEqual(result["status"], "no_candidates")

    def test_invalid_trial_abstains(self):
        with patch("run_blocks_p1.trial", return_value={**self.mock_trial(None, self.good, None, None, None), "trial_valid": False}):
            result = search(self.protocol, self.pool, "ranked", self.path / "invalid", {})
        self.assertEqual(result["status"], "infrastructure_invalid")
        self.assertIsNone(result["selected"])

    def test_real_bad_offset_fails_without_model_fallback(self):
        layout = self.protocol["layouts"][0]
        result = trial(self.protocol, self.bad, layout, [False, 0.], self.path / "actual")
        self.assertTrue(result["trial_valid"])
        self.assertFalse(result["success"])
        self.assertIn("two-finger", result["exception"])


if __name__ == "__main__":
    unittest.main()
