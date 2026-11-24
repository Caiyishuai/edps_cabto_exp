"""Finite grasp-parameter region probe; no learned models or renderer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import resource
import sys
import time

from blocks_p0 import ROOT, digest, execute_trial, make_config, save
from contract_tradeoff import RecordedExperiment, compact_result
from run_blocks_p0 import assert_sources, snapshot_sources, source_files

DEFAULT_PROTOCOL = Path(__file__).with_name("capability_regions_protocol.json")


def load(path):
    return json.loads(Path(path).read_text())


def validate_protocol(p):
    if p["max_trials"] != 24 or p["render"] or p["model_calls"] != 0 or p["require_clear"] is not True:
        raise ValueError("Expected fixed 24-trial no-model complete-contract pilot")
    if p["occupied_conditions"] != [False, True]:
        raise ValueError("Unexpected conditions")
    expected = [("zero", 0.0), ("lower4mm", -0.004), ("higher4mm", 0.004)]
    if [(c["id"], c["offset_z_m"]) for c in p["candidates"]] != expected:
        raise ValueError("Unexpected parameter candidates")
    layouts = p["layouts"]
    if len(layouts) != 4 or len({l["id"] for l in layouts}) != 4:
        raise ValueError("Four distinct layouts required")
    if [l["split"] for l in layouts] != ["development", "development", "extension", "extension"]:
        raise ValueError("Selection must precede extension")
    for layout in layouts:
        if [layout[k] for k in ("payload", "support", "blocker", "distractor")] != ["yellow", "green", "red", "blue"]:
            raise ValueError("Object roles must remain fixed")
    if p["conditional_feature"] != "observed initial clear(support); scene id and injected condition are not selector inputs":
        raise ValueError("Unsupported conditional selector")


def case_id(layout, occupied):
    return f"{layout['id']}_{'occupied' if occupied else 'free'}"


def expected_keys(p, split=None):
    return {(case_id(l, o), c["id"]) for l in p["layouts"] if split is None or l["split"] == split
            for o in p["occupied_conditions"] for c in p["candidates"]}


def complete_valid(p, rows, split=None):
    selected = [r for r in rows if split is None or r["split"] == split]
    keys = [(r["scenario"], r["candidate"]) for r in selected]
    return len(keys) == len(set(keys)) and set(keys) == expected_keys(p, split) and all(r["trial_valid"] for r in selected)


def rank_candidate(p, rows):
    scores = [{"candidate": c["id"], "successes": sum(r["strict_success"] for r in rows if r["candidate"] == c["id"]),
               "observed_cases": sum(r["candidate"] == c["id"] for r in rows),
               "abs_offset_m": abs(c["offset_z_m"]), "order": i} for i, c in enumerate(p["candidates"])]
    scores.sort(key=lambda s: (-s["successes"], s["abs_offset_m"], s["order"]))
    return {"candidate": scores[0]["candidate"] if scores[0]["successes"] else None, "scores": scores}


def freeze_selection(p, rows):
    dev = [r for r in rows if r["split"] == "development"]
    if not complete_valid(p, dev, "development"):
        return {"status": "abstain", "reason": "incomplete_or_invalid_development_matrix", "global": None, "by_observed_clear": {}}
    return {"status": "frozen", "global": rank_candidate(p, dev),
            "by_observed_clear": {str(value).lower(): rank_candidate(p, [r for r in dev if r["observed_clear_support"] is value])
                                  for value in (False, True)},
            "evidence_sha256": digest(dev), "evidence_trial_count": len(dev), "extension_observed": False,
            "selection_scope": "finite_grid_baseline_not_learned_router"}


def assert_pairing(p, rows):
    keys = [(r["scenario"], r["candidate"]) for r in rows]
    if len(keys) != len(set(keys)) or not set(keys) <= expected_keys(p):
        raise ValueError("Duplicate or unregistered trial")
    for scenario in {r["scenario"] for r in rows}:
        group = [r for r in rows if r["scenario"] == scenario]
        for key in ("scene_sha256", "initial_state_sha256", "observed_clear_support"):
            values = {r[key] for r in group}
            if len(values) != 1 or None in values:
                raise ValueError(f"Unmatched {key}: {scenario}")
    models = {r["models_sha256"] for r in rows}
    if rows and (len(models) != 1 or None in models):
        raise ValueError("Contracts must remain identical across all candidates and layouts")
    for c in p["candidates"]:
        policies = {r["policies_sha256"] for r in rows if r["candidate"] == c["id"]}
        if policies and (len(policies) != 1 or None in policies):
            raise ValueError("Same parameter has different policy source")


def summarize(p, rows, selection):
    fixed = {}
    for split in ("development", "extension", "all"):
        group = [r for r in rows if split == "all" or r["split"] == split]
        valid = complete_valid(p, group, None if split == "all" else split)
        fixed[split] = {c["id"]: {"successes": sum(r["strict_success"] for r in group if r["candidate"] == c["id"]),
                                   "recorded": sum(r["candidate"] == c["id"] for r in group),
                                   "complete_valid": valid} for c in p["candidates"]}
    regions = {}
    for scenario in sorted({r["scenario"] for r in rows}):
        regions[scenario] = [r["candidate"] for r in rows if r["scenario"] == scenario and r["strict_success"]]
    full = complete_valid(p, rows)
    per_candidate = {c["id"]: {r["scenario"] for r in rows if r["candidate"] == c["id"] and r["strict_success"]}
                     for c in p["candidates"]}
    cases = len({key[0] for key in expected_keys(p)})
    best = max((len(s) for s in per_candidate.values()), default=0)
    upper = sum(bool(s) for s in regions.values())
    crossing = []
    for a, b in itertools.combinations(per_candidate, 2):
        only_a, only_b = per_candidate[a] - per_candidate[b], per_candidate[b] - per_candidate[a]
        if only_a and only_b:
            crossing.append({"a": a, "b": b, "a_only_cases": sorted(only_a), "b_only_cases": sorted(only_b)})
    deployed = []
    if selection.get("status") == "frozen":
        for scenario in sorted({r["scenario"] for r in rows if r["split"] == "extension"}):
            group = [r for r in rows if r["scenario"] == scenario]
            feature = str(group[0]["observed_clear_support"]).lower()
            for method, selected in (("global", selection["global"]), ("observable_clear_router", selection["by_observed_clear"][feature])):
                cid = selected["candidate"]
                trial = next((r for r in group if r["candidate"] == cid), None)
                deployed.append({"scenario": scenario, "method": method, "candidate": cid,
                                 "abstained": cid is None, "record_available": trial is not None,
                                 "strict_success": trial["strict_success"] if trial else None})
    decision = ("unresolved_incomplete_or_invalid" if not full else
                "no_need_for_context_selection_in_this_matrix" if best == cases else
                "finite_candidate_complementarity_only" if upper > best else
                "no_context_selection_gain_observed")
    return {"fixed_candidate_results": fixed, "success_regions": regions,
            "hindsight_diagnostic": {"complete_valid": full, "cases": cases,
                                     "best_fixed_successes": best if full else None,
                                     "per_case_oracle_upper_bound": upper if full else None,
                                     "upper_bound_gain_over_best_fixed": upper - best if full else None,
                                     "crossing_pairs": crossing if full else [],
                                     "no_success_witness_cases": [s for s, c in regions.items() if not c] if full else [],
                                     "not_deployed_performance": True},
            "frozen_extension_evaluation": deployed, "investment_decision": decision,
            "cost": {"environment_trials": len(rows), "action_execution_episodes": sum(r["action_execution_started"] for r in rows),
                     "trial_wall_seconds": sum(r["wall_seconds"] for r in rows),
                     "simulation_seconds": sum(r["simulation_seconds"] or 0 for r in rows),
                     "model_calls": 0, "additional_selector_rollouts": 0}}


def grasp_evidence(result):
    entries = []
    for step in result["steps"]:
        if step["model"]["name"] not in ("pick_table", "unstack"):
            continue
        calls = step["api_trace"]
        point = next((c.get("result") for c in calls if c["method"] == "grasp_point"), None)
        descent = next((c.get("args", [None])[0] for c in calls if c["method"] == "descend"), None)
        after = step.get("after", {})
        entries.append({"action": step["model"], "read_grasp_xyz": point, "commanded_descent_xyz": descent,
                        "measured_command_offset_z": descent[2] - point[2] if descent and point else None,
                        "first_api_exception": next((c for c in calls if c.get("status") != "returned"), None),
                        "after_tcp": after.get("tcp"), "after_finger_forces_N": after.get("finger_forces_N"),
                        "after_objects": after.get("objects"), "contract_check": step.get("check"),
                        "note": "after_* is action-end evidence, not a closure-time force measurement"})
    return entries


def run(protocol_path, output):
    p = load(protocol_path)
    validate_protocol(p)
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Output must remain in workspace")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "protocol.json", p)
    started = time.monotonic()
    hashes = snapshot_sources(source_files(protocol_path, p), output)
    save(output / "manifest.json", {"created_at": datetime.now(timezone.utc).isoformat(), "protocol_sha256": digest(p),
                                   "source_hashes": hashes, "python": sys.version, "executable": sys.executable,
                                   "packages": {n: importlib.metadata.version(n) for n in ("mujoco", "numpy", "scipy")}})
    rows, selection = [], {}
    status, error = "complete", None
    try:
        for layout in p["layouts"]:
            if layout["split"] == "extension" and not selection:
                selection = freeze_selection(p, rows)
                if selection["status"] != "frozen":
                    raise RuntimeError("Cannot freeze selection")
                save(output / "selection_frozen.json", {"protocol_sha256": digest(p), **selection})
            for occupied in p["occupied_conditions"]:
                config = make_config(p, layout, occupied)
                if config["control"]["grasp_mode"] != "strict":
                    raise ValueError("Strict no-weld control required")
                task = {k: layout[k] for k in ("payload", "support", "distractor", "buffer_xy")}
                scenario = case_id(layout, occupied)
                for c in p["candidates"]:
                    if len(rows) >= p["max_trials"]:
                        raise RuntimeError("Hard trial budget exhausted")
                    assert_sources(hashes)
                    path = output / "trials" / scenario / c["id"]
                    result = execute_trial(p, config, task, True, c["offset_z_m"], path, render=False,
                                           experiment_type=RecordedExperiment)
                    row = compact_result(scenario, c, result, path)
                    initial = result.get("initial", {}).get("state", [])
                    row.update(layout_id=layout["id"], split=layout["split"],
                               observed_clear_support=f"clear({layout['support']})" in initial,
                               grasp_evidence=grasp_evidence(result))
                    rows.append(row)
                    save(output / "trials.json", rows)
                    if not row["trial_valid"]:
                        raise RuntimeError(f"Invalid trial {scenario}/{c['id']}: {row['exception']}")
                    if row["observed_clear_support"] != (not occupied):
                        raise RuntimeError("Observed initial condition does not match scene construction")
                    assert_pairing(p, rows)
                    print(f"{scenario}/{c['id']}: success={row['strict_success']} error={row['exception']}", flush=True)
    except Exception as exc:
        status, error = "invalid_experiment", f"{type(exc).__name__}: {exc}"
    try:
        assert_sources(hashes)
        assert_pairing(p, rows)
    except Exception as exc:
        status, error = "invalid_experiment", f"{type(exc).__name__}: {exc}"
    report = {"status": status, "error": error, "protocol_sha256": digest(p), "rows": rows, "selection": selection,
              **summarize(p, rows, selection), "total_wall_seconds": time.monotonic() - started,
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)}
    if status != "complete":
        report["investment_decision"] = "unresolved_invalid_experiment"
    save(output / "results.json", report)
    print(json.dumps({k: report[k] for k in ("status", "error", "hindsight_diagnostic", "investment_decision", "cost", "peak_rss_mib")}, indent=2), flush=True)
    return 0 if status == "complete" else 1


def audit(output):
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Audit output must remain in workspace")
    p, report, manifest = (load(output / n) for n in ("protocol.json", "results.json", "manifest.json"))
    validate_protocol(p)
    if report["status"] != "complete" or not complete_valid(p, report["rows"]):
        raise ValueError("Completed valid matrix required for audit")
    if digest(p) != report["protocol_sha256"] or digest(p) != manifest["protocol_sha256"]:
        raise ValueError("Protocol digest mismatch")
    for name, expected in manifest["source_hashes"].items():
        if hashlib.sha256((output / "source_snapshot" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Snapshot changed: {name}")
    rows = report["rows"]
    if rows != load(output / "trials.json"):
        raise ValueError("Trial summary mismatch")
    assert_pairing(p, rows)
    for row in rows:
        path = output / "trials" / row["scenario"] / row["candidate"]
        result, models, policies = (load(path / n) for n in ("result.json", "models.json", "policies.json"))
        if digest(models) != row["models_sha256"] or digest(policies) != row["policies_sha256"]:
            raise ValueError("Raw model/policy mismatch")
        if hashlib.sha256((path / "scene.xml").read_bytes()).hexdigest() != row["scene_sha256"]:
            raise ValueError("Raw scene mismatch")
        if digest(result["initial"]["execution_state"]) != row["initial_state_sha256"]:
            raise ValueError("Initial state mismatch")
        if row["strict_success"] != result["success"] or not result["trial_valid"]:
            raise ValueError("Success/trial validity mismatch")
        expected_row = compact_result(row["scenario"], {"id": row["candidate"]}, result, path)
        if any(row[k] != value for k, value in expected_row.items()):
            raise ValueError("Compact evidence differs from raw trial")
        if row["grasp_evidence"] != grasp_evidence(result):
            raise ValueError("Grasp evidence mismatch")
        layout = next(l for l in p["layouts"] if l["id"] == row["layout_id"])
        observed = f"clear({layout['support']})" in result["initial"]["state"]
        if row["split"] != layout["split"] or row["observed_clear_support"] != observed:
            raise ValueError("Split or observable-condition mismatch")
        candidate = next(c for c in p["candidates"] if c["id"] == row["candidate"])
        for entry in row["grasp_evidence"]:
            offset = entry["measured_command_offset_z"]
            if offset is not None and abs(offset - candidate["offset_z_m"]) > 1e-10:
                raise ValueError("Commanded offset differs from registered candidate")
        if result["initial"]["active_welds"] or result["final"]["active_welds"]:
            raise ValueError("Unexpected active weld")
        if result["success"]:
            if not result["final_check"].get("stable_window_pass") or not result["final_check"]["mechanical_ok"]:
                raise ValueError("Missing strict physical witness")
            if not all(s["check"]["effect_ok"] and not s["contact_audit"]["forbidden_contacts"] for s in result["steps"]):
                raise ValueError("Successful trial has contract/contact failure")
    selection = freeze_selection(p, rows)
    frozen = load(output / "selection_frozen.json")
    if frozen != {"protocol_sha256": digest(p), **selection} or report["selection"] != selection:
        raise ValueError("Frozen selection differs from development-only result")
    metrics = summarize(p, rows, selection)
    if any(report[k] != v for k, v in metrics.items()):
        raise ValueError("Recomputed metrics disagree")
    result = {"status": "passed", "trials": len(rows), "paired_cases": len({r['scenario'] for r in rows}),
              "source_snapshot_files": len(manifest["source_hashes"]), "model_calls": 0,
              "checked": ["raw traces", "source snapshots", "initial pairing", "shared contracts", "frozen development selection", "metrics"],
              "artifact_hashes": {n: hashlib.sha256((output / n).read_bytes()).hexdigest() for n in
                                  ("protocol.json", "manifest.json", "results.json", "selection_frozen.json")},
              "limits": "Finite deterministic development probe; no statistical generalization or learned skill claim."}
    save(output / "audit.json", result)
    print(json.dumps(result, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "audit"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return audit(args.output) if args.command == "audit" else run(args.protocol.resolve(), args.output)


if __name__ == "__main__":
    raise SystemExit(main())
