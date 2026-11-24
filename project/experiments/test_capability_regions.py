"""Logic-only tests for parameter-region comparisons; no physics/model calls."""
from copy import deepcopy
import unittest

from capability_regions import (DEFAULT_PROTOCOL, assert_pairing, case_id, complete_valid,
                                expected_keys, freeze_selection, load, summarize, validate_protocol)
from blocks_p0 import action_models, digest, make_config, policy_source


class CapabilityRegionsTests(unittest.TestCase):
    def setUp(self):
        self.p = load(DEFAULT_PROTOCOL)

    def rows(self, outcomes=None):
        rows = []
        for layout in self.p["layouts"]:
            for occupied in self.p["occupied_conditions"]:
                scenario = case_id(layout, occupied)
                for c in self.p["candidates"]:
                    success = True if outcomes is None else outcomes(scenario, occupied, c["id"])
                    rows.append({"scenario": scenario, "candidate": c["id"], "layout_id": layout["id"],
                                 "split": layout["split"], "observed_clear_support": not occupied,
                                 "strict_success": success, "trial_valid": True, "action_execution_started": True,
                                 "wall_seconds": 1., "simulation_seconds": 2., "models_sha256": "shared",
                                 "policies_sha256": c["id"], "scene_sha256": scenario,
                                 "initial_state_sha256": scenario})
        return rows

    def test_fixed_budget(self):
        validate_protocol(self.p)
        self.assertEqual(len(expected_keys(self.p)), 24)
        self.assertEqual(len(expected_keys(self.p, "development")), 12)
        self.p["max_trials"] = 30
        with self.assertRaises(ValueError):
            validate_protocol(self.p)

    def test_extension_cannot_precede_development(self):
        self.p["layouts"].reverse()
        with self.assertRaises(ValueError):
            validate_protocol(self.p)

    def test_same_contract_and_geometry_across_candidate_parameters(self):
        hashes = set()
        for layout in self.p["layouts"]:
            for occupied in (False, True):
                config = make_config(self.p, layout, occupied)
                self.assertEqual(config["control"]["grasp_mode"], "strict")
                self.assertEqual(config["cube_half_size"], .022)
                self.assertFalse(any(a == layout["support"] for a, _ in config["initial_on"]))
                models = action_models(config["blocks"], layout["support"], layout["payload"], True)
                self.assertEqual(len(models), 11)
                hashes.add(digest(models))
        self.assertEqual(len(hashes), 1)
        self.assertEqual(policy_source("stack", -.004), policy_source("stack", .004))
        self.assertNotEqual(policy_source("pick_table", -.004), policy_source("pick_table", .004))

    def test_frozen_global_tie_prefers_zero(self):
        selected = freeze_selection(self.p, self.rows())
        self.assertEqual(selected["global"]["candidate"], "zero")
        self.assertEqual(selected["evidence_trial_count"], 12)
        self.assertFalse(selected["extension_observed"])
        self.assertTrue(all(s["candidate"] == "zero" for s in selected["by_observed_clear"].values()))

    def test_no_extension_labels_in_selection(self):
        rows = self.rows()
        before = freeze_selection(self.p, rows)
        for r in rows:
            if r["split"] == "extension":
                r.update(strict_success=False, trial_valid=False, observed_clear_support=not r["observed_clear_support"])
        self.assertEqual(before, freeze_selection(self.p, rows))

    def test_router_uses_observed_condition(self):
        rows = self.rows(lambda _, occupied, cid: cid == ("lower4mm" if occupied else "higher4mm"))
        for row in rows:
            row["injection_label"] = "wrong_label_should_not_be_read"
        selected = freeze_selection(self.p, rows)
        self.assertEqual(selected["by_observed_clear"]["false"]["candidate"], "lower4mm")
        self.assertEqual(selected["by_observed_clear"]["true"]["candidate"], "higher4mm")
        summary = summarize(self.p, rows, selected)
        self.assertEqual(summary["investment_decision"], "finite_candidate_complementarity_only")
        self.assertEqual(summary["hindsight_diagnostic"]["upper_bound_gain_over_best_fixed"], 4)
        routed = [r for r in summary["frozen_extension_evaluation"] if r["method"] == "observable_clear_router"]
        self.assertEqual(len(routed), 4)
        self.assertTrue(all(r["strict_success"] for r in routed))

    def test_crossing_two_bad_candidates_does_not_beat_universal_third(self):
        rows = self.rows(lambda _, occupied, cid: cid == "zero" or cid == ("lower4mm" if occupied else "higher4mm"))
        summary = summarize(self.p, rows, freeze_selection(self.p, rows))
        self.assertTrue(summary["hindsight_diagnostic"]["crossing_pairs"])
        self.assertEqual(summary["hindsight_diagnostic"]["upper_bound_gain_over_best_fixed"], 0)
        self.assertEqual(summary["investment_decision"], "no_need_for_context_selection_in_this_matrix")

    def test_no_success_is_not_impossibility(self):
        rows = self.rows(lambda *args: False)
        selected = freeze_selection(self.p, rows)
        self.assertIsNone(selected["global"]["candidate"])
        summary = summarize(self.p, rows, selected)
        self.assertEqual(len(summary["hindsight_diagnostic"]["no_success_witness_cases"]), 8)
        self.assertEqual(summary["investment_decision"], "no_context_selection_gain_observed")
        self.assertTrue(all(r["abstained"] and r["strict_success"] is None for r in summary["frozen_extension_evaluation"]))

    def test_incomplete_or_duplicate_data_not_full_matrix(self):
        rows = self.rows()
        self.assertTrue(complete_valid(self.p, rows))
        self.assertFalse(complete_valid(self.p, rows[:-1]))
        self.assertFalse(complete_valid(self.p, rows + [rows[0]]))
        self.assertEqual(freeze_selection(self.p, rows[:11])["status"], "abstain")
        self.assertEqual(summarize(self.p, rows[:-1], {})["investment_decision"], "unresolved_incomplete_or_invalid")

    def test_pairing_uses_each_layout_and_condition(self):
        rows = self.rows()
        assert_pairing(self.p, rows)
        for key in ("initial_state_sha256", "scene_sha256", "models_sha256", "policies_sha256", "observed_clear_support"):
            bad = deepcopy(rows)
            bad[-1][key] = "different"
            with self.assertRaises(ValueError):
                assert_pairing(self.p, bad)


if __name__ == "__main__":
    unittest.main()
