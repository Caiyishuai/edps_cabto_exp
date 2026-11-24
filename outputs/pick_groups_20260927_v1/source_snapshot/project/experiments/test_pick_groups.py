"""Small data-pipeline regression tests; no VLM training."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocks_p0 import ROOT
from collect_pick_groups import derive_source, model_for, load_reference, snapshot, PhaseRecorder, collect_trial
from export_pick_groups import public_trial, diagnose_target, intervention_descriptor, update_target, checked
from core.cover_cabto_api import FineCover, transition_check
from policy_codegen import PolicyProgram


class PickGroupTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(Path(__file__).with_name("pick_groups_protocol.json").read_text())
        self.reference = load_reference(self.protocol)

    def test_reference_hash_and_unchanged_source(self):
        self.assertEqual(derive_source(self.reference["source"], 0.), self.reference["source"])

    def test_parameter_mutation_preserves_call_order(self):
        original = ast.parse(self.reference["source"])
        modified = ast.parse(derive_source(self.reference["source"], .045))
        calls = lambda t: [n.func.attr for n in ast.walk(t) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertEqual(calls(original), calls(modified))
        self.assertEqual(len(modified.body[0].body), len(original.body[0].body) + 1)

    def test_only_delete_empty_changes(self):
        before = model_for("apple", "board", False)
        after = model_for("apple", "board", True)
        self.assertEqual(set(before["del"]) - set(after["del"]), {"empty()"})
        after["del"] = before["del"]
        self.assertEqual(before, after)

    def test_omission_can_fail_contract_despite_physical_grasp(self):
        before = {"empty()", "at_source(apple)"}
        after = {"holding(apple)"}
        self.assertTrue(transition_check(model_for("apple", "board", False), before, after)["effect_ok"])
        result = transition_check(model_for("apple", "board", True), before, after)
        self.assertFalse(result["effect_ok"])
        self.assertEqual(result["violations"]["unexpected_removed"], ["empty()"])

    def test_split_is_by_whole_scene_seed_not_example(self):
        by_seed = {}
        for context in self.protocol["contexts"]:
            by_seed.setdefault(context["seed"], set()).add(context["split"])
        self.assertTrue(all(len(x) == 1 for x in by_seed.values()))
        self.assertEqual(len(self.protocol["scenes"]) * 2 * len(self.protocol["contexts"]) * len(self.protocol["conditions"]), 32)

    def observation(self, physical=False, contract=False):
        return {"model": model_for("apple", "board", True), "source": self.reference["source"],
                "before": {"state": ["empty()"], "active_welds": []}, "after": {"state": [], "active_welds": []},
                "api_trace": [{"method": "close_gripper", "status": "failed"}] if not physical else [],
                "exception": None, "check": {"violations": {}}, "physical_pick_success": physical,
                "contract_consistent": contract, "physical_metrics": {}}

    def test_private_labels_cannot_influence_input_or_diagnosis(self):
        original = self.observation()
        public = public_trial(original)
        modified = {**original, "condition": "combined", "evaluation_only_truth": "model", "future": {"success": True}}
        self.assertEqual(public, public_trial(modified))
        self.assertEqual(diagnose_target(public), diagnose_target(public_trial(modified)))
        self.assertEqual(diagnose_target(public)["decision"], "request_discriminating_trial")
        self.assertFalse(diagnose_target(public)["deployable_repair"])

    def test_model_only_update_does_not_claim_physical_gain(self):
        original = public_trial(self.observation(True, False))
        candidate = public_trial(self.observation(True, True))
        target = update_target(original, candidate)
        self.assertFalse(target["physical_improved"])
        self.assertTrue(target["contract_improved"])
        self.assertFalse(target["deployable_repair"])

    def test_forecast_descriptor_contains_no_future_outcome(self):
        original = self.observation(False, False)
        candidate = self.observation(True, True)
        descriptor = intervention_descriptor(original, candidate)
        candidate["physical_pick_success"] = False
        candidate["after"]["state"] = ["SECRET"]
        self.assertEqual(descriptor, intervention_descriptor(original, candidate))
        self.assertNotIn("SECRET", json.dumps(descriptor))

    def test_output_paths_are_bounded(self):
        with self.assertRaises(ValueError):
            checked(ROOT, "../outside")

    def test_stage_rendering_does_not_step_or_control_physics(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs", prefix="pick-test-") as out:
            config = json.loads((ROOT / self.protocol["scenes"][0]["config"]).read_text())
            fine = FineCover(config, render=False, perception="oracle", out=out)
            recorder = PhaseRecorder(fine, out, {**self.protocol, "image_size": 128})
            try:
                fine.bind(model_for("apple", "board", False))
                before = fine.r.env.d.qpos.copy()
                t = fine.r.env.d.time
                frame = recorder.capture("initial", "observed")
                np.testing.assert_array_equal(before, fine.r.env.d.qpos)
                self.assertEqual(t, fine.r.env.d.time)
                self.assertEqual(frame["images"][0]["sim_time"], frame["images"][1]["sim_time"])
                self.assertNotIn("home_completed()", snapshot(fine)["state"])
            finally:
                recorder.close()
                fine.close()

    def test_no_actual_state_writes_in_new_collector(self):
        tree = ast.parse(Path(__file__).with_name("collect_pick_groups.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    attrs = {x.attr for x in ast.walk(target) if isinstance(x, ast.Attribute)}
                    self.assertFalse(attrs & {"qpos", "qvel", "eq_active", "body_mass"})


if __name__ == "__main__":
    unittest.main()
