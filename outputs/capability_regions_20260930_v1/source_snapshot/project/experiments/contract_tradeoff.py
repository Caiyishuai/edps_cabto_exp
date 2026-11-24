"""Bounded, no-model contract/capability pilot with frozen candidate selection."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time

from blocks_p0 import (ROOT, BlocksExperiment, action_models, digest, execute_trial,
                       make_config, save, validate_model)
from run_blocks_p0 import assert_sources, snapshot_sources, source_files

DEFAULT_PROTOCOL = Path(__file__).with_name("contract_tradeoff_protocol.json")


def plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


class RecordedExperiment(BlocksExperiment):
    """Add observation-only state records; inherit all control and evaluation."""

    def snapshot(self):
        row = super().snapshot()
        r, d = self.r, self.env.d
        physics = {key: plain(getattr(d, key)) for key in
                   ("qpos", "qvel", "act", "ctrl", "eq_active", "qacc_warmstart",
                    "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat", "userdata")}
        physics["time"] = float(d.time)
        controller = {key: plain(getattr(r, key)) for key in
                      ("phase", "target", "current_support", "completed", "home_event",
                       "qprev", "transport_monitor", "grasp_relative", "max_grasp_slip",
                       "current_contact_gap", "max_contact_gap", "force_now")}
        controller.update(arm_target=plain(r.arm.arm_target), grip_cmd=float(r.arm._grip_cmd),
                          source_supports={key: sorted(value) for key, value in self.source_supports.items()})
        state = {"physics": physics, "controller": controller}
        row["execution_state"] = state
        row["execution_state_sha256"] = digest(state)
        return row


def scenario_config(protocol, scenario):
    if scenario not in ("H", "D", "E"):
        raise ValueError("Unknown scenario")
    layout = protocol["layout"]
    config = make_config(protocol, layout, occupied=scenario == "D")
    if scenario == "E":
        config["blocks"][layout["blocker"]] = list(layout["distractor_xy"])
        config["initial_on"] = [[layout["blocker"], layout["distractor"]]]
    return config


def candidate_models(protocol, candidate, objects):
    layout = protocol["layout"]
    models = action_models(objects, layout["support"], layout["payload"], candidate["require_clear"])
    if candidate["require_distractor_clear"]:
        for model in models:
            if model["name"] == "stack":
                model["pre"].append(f"clear({layout['distractor']})")
    for model in models:
        validate_model(model)
    return models


def inventory(models):
    return [{key: deepcopy(model[key]) for key in ("name", "args", "add", "del")} for model in models]


def edit_count(protocol, candidate):
    return (int(candidate["require_clear"]) + int(candidate["require_distractor_clear"])
            + int(candidate["offset_z_m"] != protocol["initial_offset_z_m"]))


def choose_candidate(protocol, rows, scenarios):
    if not scenarios or not set(scenarios) <= {"H", "D"}:
        raise ValueError("Selection may only use registered development scenarios H/D")
    selected = [row for row in rows if row["scenario"] in scenarios]
    expected = {(scenario, candidate["id"]) for scenario in scenarios for candidate in protocol["candidates"]}
    keys = [(row["scenario"], row["candidate"]) for row in selected]
    if set(keys) != expected or len(keys) != len(expected) or not all(row["trial_valid"] for row in selected):
        return {"candidate": None, "reason": "incomplete_or_invalid_development_evidence"}
    scores = []
    for index, candidate in enumerate(protocol["candidates"]):
        count = sum(row["strict_success"] for row in selected if row["candidate"] == candidate["id"])
        scores.append({"candidate": candidate["id"], "success_count": count,
                       "task_count": len(scenarios), "edit_count": edit_count(protocol, candidate), "order": index})
    scores.sort(key=lambda row: (-row["success_count"], row["edit_count"], row["order"]))
    return {"candidate": scores[0]["candidate"] if scores[0]["success_count"] else None,
            "reason": "finite_success_then_edit_cost" if scores[0]["success_count"] else "no_success_witness",
            "scores": scores, "scenarios": scenarios,
            "evidence_sha256": digest(selected), "additional_physical_rollouts": 0}


def compact_result(scenario, candidate, result, path):
    steps = result["steps"]
    valid = bool(result["trial_valid"])
    rejected = bool(result.get("candidate_rejected", False))
    check = result.get("final_check") or {}
    failures = [step for step in steps if not step.get("check", {}).get("effect_ok", False)]
    return {"scenario": scenario, "candidate": candidate["id"], "trial_valid": valid,
            "planning_accepted": bool(valid and not rejected), "planning_rejected": rejected,
            "action_execution_started": bool(steps),
            "strict_success": bool(valid and result["success"] and check.get("stable_window_pass")),
            "goal_reached": bool(check.get("goal_reached", False)),
            "mechanical_ok": bool(check.get("mechanical_ok", False)) if steps else None,
            "all_executed_contracts_match": not failures if steps and valid else None,
            "action_count": len(steps), "actions": [step["model"]["name"] for step in steps],
            "contract_failure_count": len(failures), "exception": result["exception"],
            "forbidden_contacts": check.get("contact_audit", {}).get("forbidden_contacts", []),
            "scene_sha256": result.get("scene_sha256"),
            "initial_state_sha256": result.get("initial", {}).get("execution_state_sha256"),
            "initial_symbolic_state": result.get("initial", {}).get("state"),
            "models_sha256": result.get("models_sha256"), "policies_sha256": result.get("policies_sha256"),
            "wall_seconds": result["wall_seconds"], "simulation_seconds": result.get("simulation_seconds", 0),
            "result_path": str(path / "result.json")}


def verify_pairing(rows, protocol):
    checks = []
    for scenario in ("H", "D", "E"):
        group = [row for row in rows if row["scenario"] == scenario]
        if not group:
            continue
        for key in ("scene_sha256", "initial_state_sha256"):
            values = {row[key] for row in group}
            if None in values or len(values) != 1:
                raise RuntimeError(f"{scenario}: unmatched {key}")
        checks.append({"scenario": scenario, "paired_trials": len(group), "initial_state_match": True})
    for flag, hash_key in (("offset_z_m", "policies_sha256"),):
        for value in {candidate[flag] for candidate in protocol["candidates"]}:
            ids = {candidate["id"] for candidate in protocol["candidates"] if candidate[flag] == value}
            hashes = {row[hash_key] for row in rows if row["candidate"] in ids}
            if hashes and (None in hashes or len(hashes) != 1):
                raise RuntimeError("Same policy offset must imply identical policy source")
    return checks


def summarize(protocol, rows):
    aggregates = {}
    for name, scenarios in (("narrow", ["H"]), ("broader", ["H", "D"]), ("extended", ["H", "D", "E"])):
        aggregates[name] = {}
        for candidate in protocol["candidates"]:
            group = [row for row in rows if row["scenario"] in scenarios and row["candidate"] == candidate["id"]]
            n, expected = len(group), len(scenarios)
            complete = n == expected and all(row["trial_valid"] for row in group)
            executed = [row for row in group if row["action_execution_started"] and row["trial_valid"]]
            succeeded = sum(row["strict_success"] for row in group)
            aggregates[name][candidate["id"]] = {
                "recorded": n, "expected": expected, "complete_valid_matrix": complete,
                "strict_successes": succeeded, "coverage": succeeded / expected if complete else None,
                "planning_accepts": sum(row["planning_accepted"] for row in group),
                "planning_rejects": sum(row["planning_rejected"] for row in group),
                "invalid_trials": sum(not row["trial_valid"] for row in group),
                "executed_episodes": len(executed),
                "conditional_execution_success": sum(row["strict_success"] for row in executed) / len(executed) if executed else None,
                "contract_passing_actions": sum(row["action_count"] - row["contract_failure_count"] for row in group),
                "executed_actions": sum(row["action_count"] for row in group)}
    lookup = {(row["scenario"], row["candidate"]): row for row in rows}

    def success(scenario, candidate):
        return lookup.get((scenario, candidate), {}).get("strict_success", False)

    ej, ec = lookup.get(("E", "J"), {}), lookup.get(("E", "C"), {})
    return {"aggregates": aggregates,
            "mechanism_checks": {
                "clear_guard_enables_preparation": success("D", "J") and not success("D", "P")
                    and {"unstack", "put_table"} <= set(lookup[("D", "J")]["actions"]),
                "policy_repair_matters_under_same_guard": success("H", "J") and not success("H", "T"),
                "overrestriction_excludes_witnessed_capability": bool(ej.get("strict_success") and ec.get("trial_valid")
                    and ec.get("planning_rejected") and ej.get("policies_sha256") == ec.get("policies_sha256"))},
            "interpretation": "Finite designed interventions and selector baseline only; no learned predicates, new skill discovery or statistical reliability guarantee.",
            "cost": {"environment_trials": len(rows), "action_execution_episodes": sum(row["action_execution_started"] for row in rows),
                     "trial_wall_seconds": sum(row["wall_seconds"] for row in rows),
                     "simulation_seconds": sum(row["simulation_seconds"] or 0 for row in rows), "model_calls": 0}}


def validate_protocol(protocol):
    if protocol["max_trials"] != 15 or protocol["render"] or protocol["model_calls"] != 0:
        raise ValueError("This pilot has a fixed 15-trial, no-render, no-model budget")
    if [candidate["id"] for candidate in protocol["candidates"]] != ["B", "T", "P", "J", "C"]:
        raise ValueError("Unexpected candidate inventory")
    if protocol["extension_order"] != ["J", "B", "T", "P", "C"]:
        raise ValueError("Extension must start with the registered J witness")
    if protocol["scenario_order"] != ["H", "D", "E"]:
        raise ValueError("Development must precede extension")
    for candidate in protocol["candidates"]:
        if candidate["offset_z_m"] not in (0.0, 0.045):
            raise ValueError("Unregistered policy parameter")
    layout = protocol["layout"]
    if len({layout[key] for key in ("payload", "support", "blocker", "distractor")}) != 4:
        raise ValueError("Object roles must be distinct")


def run(output, protocol_path):
    protocol = json.loads(protocol_path.read_text())
    validate_protocol(protocol)
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Output must stay in the workspace")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "protocol.json", protocol)
    started = time.monotonic()
    hashes = snapshot_sources(source_files(protocol_path, protocol), output)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "protocol_sha256": digest(protocol),
                "source_hashes": hashes, "python": sys.version, "python_executable": sys.executable,
                "packages": {package: importlib.metadata.version(package) for package in ("mujoco", "numpy", "scipy")},
                "scope": protocol["scope"], "model_calls": 0, "render": False}
    save(output / "manifest.json", manifest)
    rows, definitions, selections = [], {}, {}
    status, error = "complete", None
    task = {key: protocol["layout"][key] for key in ("payload", "support", "distractor", "buffer_xy")}
    candidates = {candidate["id"]: candidate for candidate in protocol["candidates"]}
    try:
        for scenario in protocol["scenario_order"]:
            config = scenario_config(protocol, scenario)
            if config["control"]["grasp_mode"] != "strict":
                raise ValueError("Strict no-weld control required")
            order = protocol["extension_order"] if scenario == "E" else list(candidates)
            for cid in order:
                if len(rows) >= protocol["max_trials"]:
                    raise RuntimeError("Hard trial budget exhausted")
                assert_sources(hashes)
                candidate = candidates[cid]
                models = candidate_models(protocol, candidate, config["blocks"])
                base = action_models(config["blocks"], task["support"], task["payload"], False)
                if inventory(models) != inventory(base):
                    raise RuntimeError("Action inventory or effects changed")
                for before, after in zip(base, models):
                    if before["name"] != "stack" and before != after:
                        raise RuntimeError("Only stack preconditions may change")
                definitions[cid] = {"candidate": candidate, "models": models, "inventory_sha256": digest(inventory(models))}
                save(output / "candidate_definitions.json", definitions)
                path = output / "trials" / scenario / cid
                result = execute_trial(protocol, config, task, candidate["require_clear"], candidate["offset_z_m"],
                                       path, render=False, models_override=models, experiment_type=RecordedExperiment)
                row = compact_result(scenario, candidate, result, path)
                rows.append(row)
                save(output / "trials.json", rows)
                if not row["trial_valid"]:
                    raise RuntimeError(f"Invalid trial {scenario}/{cid}: {row['exception']}")
                verify_pairing(rows, protocol)
                print(f"{scenario}/{cid}: success={row['strict_success']} rejected={row['planning_rejected']} actions={row['actions']}", flush=True)
                if scenario == "E" and cid == "J" and not row["strict_success"]:
                    status = "extension_witness_failed"
                    break
            if scenario == "D":
                selections = {name: choose_candidate(protocol, rows, scenarios)
                              for name, scenarios in protocol["selection_demands"].items()}
                save(output / "selection_frozen.json", {"protocol_sha256": digest(protocol), "selections": selections,
                                                       "evidence_trial_count": len(rows), "extension_observed": False})
    except Exception as exc:
        status, error = "invalid_experiment", f"{type(exc).__name__}: {exc}"
    try:
        assert_sources(hashes)
        pairing = verify_pairing(rows, protocol)
    except Exception as exc:
        status, error, pairing = "invalid_experiment", f"{type(exc).__name__}: {exc}", []
    report = {"status": status, "error": error, "protocol_sha256": digest(protocol), "rows": rows,
              "pairing": pairing, "selections": selections, **summarize(protocol, rows),
              "total_wall_seconds": time.monotonic() - started,
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)}
    if status == "invalid_experiment":
        report["mechanism_checks"] = {key: None for key in report["mechanism_checks"]}
    save(output / "results.json", report)
    print(json.dumps({key: report[key] for key in ("status", "error", "mechanism_checks", "cost", "peak_rss_mib")}, ensure_ascii=False, indent=2), flush=True)
    return 1 if status == "invalid_experiment" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(args.output, args.protocol.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
