"""Pure logic tests: no simulator construction, rendering or model inference."""
from copy import deepcopy
import inspect
import json
import unittest

from blocks_p0 import BlocksExperiment, ModelLibrary, PlanningError, action_models
from contract_tradeoff import (DEFAULT_PROTOCOL, RecordedExperiment, candidate_models,
                              choose_candidate, compact_result, inventory, scenario_config,
                              summarize, validate_protocol, verify_pairing)
from formal_bt import symbolic_dry_run


class ContractTradeoffTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(DEFAULT_PROTOCOL.read_text())
        self.candidates = {row["id"]: row for row in self.protocol["candidates"]}

    def rows(self):
        rows = []
        for scenario in ("H", "D"):
            for cid in self.candidates:
                rows.append({"scenario": scenario, "candidate": cid, "trial_valid": True,
                             "strict_success": cid in (("P", "J", "C") if scenario == "H" else ("J", "C"))})
        return rows

    def initial_state(self, scenario):
        config = scenario_config(self.protocol, scenario)
        state = {"empty()"}
        on = config["initial_on"]
        for obj in config["blocks"]:
            if not any(upper == obj for upper, _ in on):
                state.add(f"ontable({obj})")
            if not any(lower == obj for _, lower in on):
                state.add(f"clear({obj})")
        state.update(f"on({upper},{lower})" for upper, lower in on)
        return config, state

    def test_registered_budget_and_order(self):
        validate_protocol(self.protocol)
        self.protocol["max_trials"] = 16
        with self.assertRaises(ValueError):
            validate_protocol(self.protocol)

    def test_candidate_inventory_effects_and_unrelated_pre_unchanged(self):
        config = scenario_config(self.protocol, "H")
        base = action_models(config["blocks"], "green", "yellow", False)
        self.assertEqual(len(base), 11)
        for candidate in self.candidates.values():
            models = candidate_models(self.protocol, candidate, config["blocks"])
            self.assertEqual(inventory(base), inventory(models))
            for old, new in zip(base, models):
                if old["name"] != "stack":
                    self.assertEqual(old, new)

    def test_conservative_changes_only_one_pre_relative_to_joint(self):
        config = scenario_config(self.protocol, "H")
        joint = candidate_models(self.protocol, self.candidates["J"], config["blocks"])
        conservative = candidate_models(self.protocol, self.candidates["C"], config["blocks"])
        for j, c in zip(joint, conservative):
            if j["name"] == "stack":
                self.assertEqual(set(c["pre"]) - set(j["pre"]), {"clear(blue)"})
                c["pre"] = j["pre"]
            self.assertEqual(j, c)

    def test_extension_retains_target_on_table_and_protocol_unmutated(self):
        before = deepcopy(self.protocol)
        config = scenario_config(self.protocol, "E")
        self.assertEqual(config["initial_on"], [["red", "blue"]])
        self.assertNotIn("green", [upper for upper, _ in config["initial_on"]])
        self.assertEqual(before, self.protocol)

    def test_symbolic_clear_guard_composes_existing_skills(self):
        config, state = self.initial_state("D")
        models = candidate_models(self.protocol, self.candidates["J"], config["blocks"])
        goal = {"on(yellow,green)", "empty()", "home_completed()"}
        trace = symbolic_dry_run(ModelLibrary(models).build(state, goal), state, goal, 16)
        self.assertTrue(trace["reached_goal"])
        self.assertTrue(trace["action_ids"][0].startswith("unstack(object=red"))
        self.assertEqual(trace["steps"], 5)

    def test_extension_conservative_symbolic_rejection_is_not_physical_proof(self):
        config, state = self.initial_state("E")
        goal = {"on(yellow,green)", "empty()", "home_completed()"}
        joint = candidate_models(self.protocol, self.candidates["J"], config["blocks"])
        ModelLibrary(joint).build(state, goal)
        conservative = candidate_models(self.protocol, self.candidates["C"], config["blocks"])
        with self.assertRaises(PlanningError):
            ModelLibrary(conservative).build(state, goal)

    def test_selector_depends_on_demand_not_extension(self):
        rows = self.rows()
        self.assertEqual(choose_candidate(self.protocol, rows, ["H"])["candidate"], "P")
        self.assertEqual(choose_candidate(self.protocol, rows, ["H", "D"])["candidate"], "J")
        rows.append({"scenario": "E", "candidate": "B", "strict_success": True, "trial_valid": False})
        self.assertEqual(choose_candidate(self.protocol, rows, ["H", "D"])["candidate"], "J")
        with self.assertRaises(ValueError):
            choose_candidate(self.protocol, rows, ["E"])

    def test_selector_abstains_on_incomplete_invalid_or_zero_success(self):
        rows = self.rows()
        self.assertIsNone(choose_candidate(self.protocol, rows[:-1], ["H", "D"])["candidate"])
        rows[0]["trial_valid"] = False
        self.assertIsNone(choose_candidate(self.protocol, rows, ["H"])["candidate"])
        for row in rows:
            row.update(trial_valid=True, strict_success=False)
        self.assertIsNone(choose_candidate(self.protocol, rows, ["H", "D"])["candidate"])

    def test_rejection_has_no_vacuous_execution_success(self):
        result = {"trial_valid": True, "candidate_rejected": True, "steps": [], "success": False,
                  "exception": "PlanningError", "wall_seconds": 0.1}
        row = compact_result("E", self.candidates["C"], result, DEFAULT_PROTOCOL.parent)
        self.assertIsNone(row["all_executed_contracts_match"])
        self.assertFalse(row["action_execution_started"])
        report = summarize(self.protocol, [row])
        self.assertIsNone(report["aggregates"]["extended"]["C"]["coverage"])
        self.assertIsNone(report["aggregates"]["extended"]["C"]["conditional_execution_success"])
        self.assertFalse(report["mechanism_checks"]["overrestriction_excludes_witnessed_capability"])

    def test_pairing_rejects_control_state_mismatch(self):
        rows = [{"scenario": "H", "candidate": "P", "scene_sha256": "scene", "initial_state_sha256": "s1", "policies_sha256": "p"},
                {"scenario": "H", "candidate": "J", "scene_sha256": "scene", "initial_state_sha256": "s2", "policies_sha256": "p"}]
        with self.assertRaises(RuntimeError):
            verify_pairing(rows, self.protocol)

    def test_recorder_does_not_override_control_or_evaluator(self):
        for method in ("state", "final_check", "stable_final", "grasp_point", "destination_point", "contact_audit"):
            self.assertIs(getattr(RecordedExperiment, method), getattr(BlocksExperiment, method))
        self.assertNotIn("__init__", RecordedExperiment.__dict__)
        self.assertIn("snapshot", RecordedExperiment.__dict__)
        self.assertNotIn("render", inspect.getsource(RecordedExperiment.snapshot))


if __name__ == "__main__":
    unittest.main()
