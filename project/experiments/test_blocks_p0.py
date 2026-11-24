"""Protocol, isolation, symbolic composition and physical-observer regression tests."""
from copy import deepcopy
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocks_p0 import (BlocksExperiment, ModelLibrary, action_models, build_scene, diagnose,
                       digest, load_protocol, make_config, policy_source)
from formal_bt import symbolic_dry_run
from policy_codegen import PolicyProgram
from core.cover_cabto_api import API_DOCS
from blocks_p0 import ROOT, execute_trial, save
from run_blocks_p0 import file_hash, freeze_candidates, load_frozen, summarize


class BlocksP0Tests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol()
        self.layout = self.protocol["layouts"][0]

    def runtime(self, occupied=False, layout=None):
        layout = layout or self.layout
        runtime = BlocksExperiment(make_config(self.protocol, layout, occupied), layout, self.protocol)
        self.addCleanup(runtime.close)
        return runtime

    def test_layout_splits_and_new_pairings(self):
        layouts = self.protocol["layouts"]
        self.assertEqual(len(layouts), len({row["id"] for row in layouts}))
        train_pairs = {(r["payload"], r["support"]) for r in layouts if r["split"] != "heldout"}
        test_pairs = {(r["payload"], r["support"]) for r in layouts if r["split"] == "heldout"}
        self.assertFalse(train_pairs & test_pairs)
        self.assertEqual({"development", "validation", "heldout"}, {r["split"] for r in layouts})

    def test_clear_is_observed_not_an_injection_flag(self):
        runtime = self.runtime(True)
        self.assertIn("on(red,green)", runtime.state())
        self.assertNotIn("clear(green)", runtime.state())
        runtime.r.config["initial_on"] = []
        self.assertNotIn("clear(green)", runtime.state())
        self.assertTrue(runtime.r.supporting_contact("red", "green"))

    def test_unoccupied_initial_scene_is_clear(self):
        runtime = self.runtime()
        self.assertIn("clear(green)", runtime.state())
        self.assertIn("ontable(red)", runtime.state())
        self.assertFalse(runtime.final_check()["success"])
        self.assertEqual(runtime.r.active_welds(), [])
        self.assertFalse(runtime.r.arm._grasp_enabled)

    def test_oracle_reads_do_not_change_execution_state(self):
        runtime = self.runtime()
        runtime.bind({"name": "pick_table", "args": {"object": "yellow"}})
        before = (runtime.env.d.time, runtime.env.d.qpos.copy(), runtime.env.d.qvel.copy(), runtime.env.d.ctrl.copy())
        runtime.grasp_point()
        runtime.snapshot()
        self.assertEqual(before[0], runtime.env.d.time)
        for old, new in zip(before[1:], (runtime.env.d.qpos, runtime.env.d.qvel, runtime.env.d.ctrl)):
            np.testing.assert_array_equal(old, new)

    def test_scene_builder_does_not_mutate_configuration(self):
        config = make_config(self.protocol, self.layout, True)
        original = deepcopy(config)
        xml, names = build_scene(config)
        self.assertEqual(config, original)
        self.assertEqual(set(names), set(config["blocks"]))
        self.assertIn("red_free", xml)

    def test_model_patch_changes_only_stack_precondition(self):
        runtime = self.runtime()
        base = action_models(runtime.env.obj_names, "green", "yellow", False)
        fixed = action_models(runtime.env.obj_names, "green", "yellow", True)
        for old, new in zip(base, fixed):
            if old["name"] == "stack":
                self.assertEqual(set(new["pre"]) - set(old["pre"]), {"clear(green)"})
                new = {**new, "pre": old["pre"]}
            self.assertEqual(old, new)

    def test_repaired_bt_clears_target_before_payload_pick(self):
        runtime = self.runtime(True)
        models = action_models(runtime.env.obj_names, "green", "yellow", True)
        tree = ModelLibrary(models).build(runtime.state(), runtime.goal)
        outcome = symbolic_dry_run(tree, runtime.state(), runtime.goal, 16)
        self.assertTrue(outcome["reached_goal"], outcome)
        self.assertTrue(outcome["action_ids"][0].startswith("unstack(object=red"), outcome)
        self.assertEqual(outcome["steps"], 5)

    def test_missing_clear_allows_unsafe_symbolic_shortcut(self):
        runtime = self.runtime(True)
        models = action_models(runtime.env.obj_names, "green", "yellow", False)
        outcome = symbolic_dry_run(ModelLibrary(models).build(runtime.state(), runtime.goal), runtime.state(), runtime.goal, 16)
        self.assertTrue(outcome["reached_goal"])
        self.assertEqual(outcome["steps"], 3)
        self.assertTrue(outcome["action_ids"][0].startswith("pick_table(object=yellow"))

    def test_policy_candidates_are_finite_programs(self):
        for name in ("pick_table", "unstack", "stack", "put_table", "home"):
            api = "pick" if name in ("pick_table", "unstack") else "home" if name == "home" else "place"
            for offset in (0., .045):
                PolicyProgram(policy_source(name, offset), API_DOCS[api])
        self.assertNotEqual(digest(policy_source("pick_table", 0.)), digest(policy_source("pick_table", .045)))
        self.assertEqual(policy_source("stack", 0.), policy_source("stack", .045))

    def test_diagnosis_uses_only_intervention_outcomes(self):
        keys = ("none", "model_only", "policy_only", "cross_level")
        for bits, label in [((1, 1, 1, 1), "ok"), ((0, 1, 0, 1), "model"),
                            ((0, 0, 1, 1), "policy"), ((0, 0, 0, 1), "mixed"), ((0, 0, 0, 0), "abstain")]:
            self.assertEqual(diagnose(dict(zip(keys, map(bool, bits)))), label)
        with self.assertRaises(ValueError):
            diagnose({**dict.fromkeys(keys, True), "fault_label": "model"})
        with self.assertRaises(ValueError):
            diagnose(dict.fromkeys(keys, 1))

    def test_no_direct_execution_state_writes_in_experiment(self):
        tree = ast.parse(Path(__file__).with_name("blocks_p0.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    attrs = {child.attr for child in ast.walk(target) if isinstance(child, ast.Attribute)}
                    self.assertFalse(attrs & {"qpos", "qvel", "eq_active", "body_mass"})


class ExperimentIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "outputs", prefix="p0-test-")
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.protocol = load_protocol()
        self.layout = self.protocol["layouts"][0]

    def artifact(self):
        keys = ("none", "model_only", "policy_only", "cross_level")
        cases = []
        for index, bits in enumerate(((True, True, True, True), (False, True, False, True), (False, False, True, True))):
            outcomes = dict(zip(keys, bits))
            label = diagnose(outcomes)
            cases.append({"id": str(index), "outcomes": outcomes, "diagnosis": label, "evaluation_only_truth": label})
        hashes = {"source.py": "test-hash"}
        summary = summarize(cases)
        evidence = {"phase": "development", "protocol_sha256": digest(self.protocol),
                    "source_unchanged": True, "cases": cases}
        save(self.directory / "results.json", evidence)
        save(self.directory / "manifest.json", {"source_hashes": hashes, "protocol_sha256": digest(self.protocol)})
        frozen = freeze_candidates(self.protocol, hashes, cases, summary)
        frozen.update(phase="development", evidence_results_sha256=file_hash(self.directory / "results.json"))
        path = self.directory / "frozen.json"
        save(path, frozen)
        return path, frozen

    def test_empty_and_unapproved_freezes_rejected(self):
        for value in ({}, None, [], {"approved": False}, {"approved": 1}):
            path = self.directory / "empty.json"
            save(path, value)
            with self.assertRaises(ValueError):
                load_frozen(path, "validation", self.protocol)

    def test_valid_freeze_matches_evidence(self):
        path, frozen = self.artifact()
        self.assertEqual(load_frozen(path, "validation", self.protocol), frozen)

    def test_tampered_candidate_rejected(self):
        path, frozen = self.artifact()
        for field, value in (("schema_patch", "clear(other)"), ("policy_offset_z_m", .025)):
            save(path, {**frozen, field: value})
            with self.assertRaises(ValueError):
                load_frozen(path, "validation", self.protocol)

    def test_tampered_evidence_rejected(self):
        path, _ = self.artifact()
        save(self.directory / "results.json", {"cases": []})
        with self.assertRaises(ValueError):
            load_frozen(path, "validation", self.protocol)

    def test_development_freeze_cannot_skip_validation(self):
        path, _ = self.artifact()
        with self.assertRaises(ValueError):
            load_frozen(path, "heldout", self.protocol)

    def run_trial(self, name):
        return execute_trial(self.protocol, make_config(self.protocol, self.layout, False), self.layout,
                             False, 0., self.directory / name)

    def test_render_failure_does_not_become_physical_failure(self):
        with patch.object(BlocksExperiment, "save_image", side_effect=RuntimeError("renderer unavailable")):
            result = self.run_trial("render")
        self.assertTrue(result["success"])
        self.assertTrue(result["trial_valid"])
        self.assertFalse(result["artifact_complete"])
        self.assertEqual(len(result["infrastructure_errors"]), 2)

    def test_close_failure_preserves_result(self):
        with patch.object(BlocksExperiment, "close", side_effect=RuntimeError("close failed")):
            result = self.run_trial("close")
        self.assertTrue(result["trial_valid"])
        self.assertTrue(result["success"])
        self.assertEqual(json.loads((self.directory / "close/result.json").read_text())["success"], True)
        self.assertEqual(result["infrastructure_errors"][-1]["stage"], "close")

    def test_observer_failure_is_invalid_and_persisted(self):
        with patch.object(BlocksExperiment, "snapshot", side_effect=RuntimeError("observer failed")):
            result = self.run_trial("observer")
        self.assertFalse(result["trial_valid"])
        self.assertFalse(result["success"])
        self.assertEqual(result["exception"], "RuntimeError: observer failed")
        self.assertTrue((self.directory / "observer/result.json").is_file())
        self.assertTrue(any(e["stage"] == "final_observation" for e in result["infrastructure_errors"]))

    def test_existing_trial_directory_is_never_overwritten(self):
        directory = self.directory / "old"
        directory.mkdir()
        sentinel = directory / "result.json"
        sentinel.write_text("original")
        with self.assertRaises(FileExistsError):
            self.run_trial("old")
        self.assertEqual(sentinel.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
