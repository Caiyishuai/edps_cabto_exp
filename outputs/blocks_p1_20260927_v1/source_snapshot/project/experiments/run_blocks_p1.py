"""Real-model proposal, budgeted sequential search and sealed transfer evaluation."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

from blocks_p0 import ROOT, digest, execute_trial, make_config, save
from blocks_p1_candidates import (CandidateRejected, OfflineProposer, compile_candidate,
                                 parse_proposal, proposal_messages, public_evidence, recipe)
from run_blocks_p0 import assert_sources, file_hash, snapshot_sources, source_files

BASELINE = {"id": "unchanged", "pre_add": [], "offset_z_m": None, "reason": "Unmodified experimental baseline"}


def protocol_at(path):
    protocol = json.loads(Path(path).read_text())
    old = json.loads(Path(__file__).with_name("blocks_p0_protocol.json").read_text())
    old_pairs = {(r["payload"], r["support"]) for r in old["layouts"]}
    layouts = protocol["layouts"]
    if len({r["id"] for r in layouts}) != len(layouts):
        raise ValueError("Duplicate layout ID")
    non_test_pairs = {(r["payload"], r["support"]) for r in layouts if r["split"] != "heldout"}
    for row in layouts:
        if len({row[key] for key in ("payload", "support", "blocker", "distractor")}) != 4:
            raise ValueError("Objects must have distinct roles")
        if row["split"] == "heldout" and (row["payload"], row["support"]) in old_pairs | non_test_pairs:
            raise ValueError("Heldout directed pairing overlaps prior experiments")
        for previous in old["layouts"] + [r for r in layouts if r["id"] != row["id"]]:
            fields = ("source_xy", "target_xy", "buffer_xy", "distractor_xy")
            if all(row[k] == previous[k] for k in fields):
                raise ValueError("Physical layout overlaps another split")
    return protocol


def evaluation_task(layout):
    return {key: deepcopy(layout[key]) for key in ("payload", "support", "distractor", "buffer_xy")}


def trial(protocol, candidate, layout, condition, out, render=False):
    occupied, offset = condition
    try:
        models, value = compile_candidate(candidate, protocol, layout, offset)
    except (ValueError, TypeError, KeyError) as exc:
        Path(out).mkdir(parents=True, exist_ok=False)
        result = {"success": False, "trial_valid": True, "candidate_rejected": True,
                  "physical_rollouts": 0, "steps": [], "simulation_seconds": 0., "wall_seconds": 0.,
                  "exception": f"{type(exc).__name__}: {exc}"}
        save(Path(out) / "result.json", result)
        return result
    result = execute_trial(protocol, make_config(protocol, layout, occupied), evaluation_task(layout),
                           False, value, out, render, models_override=models)
    result["physical_rollouts"] = 1
    result["candidate_recipe_sha256"] = digest(recipe(candidate))
    save(Path(out) / "result.json", result)
    return result


def cost(result):
    return {"rollouts": result.get("physical_rollouts", 1), "actions": len(result.get("steps", [])),
            "api_calls": sum(len(row.get("api_trace", [])) for row in result.get("steps", [])),
            "simulation_seconds": result.get("simulation_seconds") or 0.,
            "wall_seconds": result.get("wall_seconds") or 0.}


def sum_cost(results):
    keys = ("rollouts", "actions", "api_calls", "simulation_seconds", "wall_seconds")
    return {key: sum(cost(r)[key] for r in results) for key in keys}


def candidate_order(pool, method, seed=None):
    indices = list(range(len(pool)))
    if method == "random":
        random.Random(seed).shuffle(indices)
    elif method == "enumerate":
        indices.sort(key=lambda i: json.dumps(recipe(pool[i]["candidate"]), sort_keys=True))
    elif method != "ranked":
        raise ValueError("Unsupported search order")
    return indices


def search(protocol, pool, method, output, hashes, seed=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    development = next(r for r in protocol["layouts"] if r["split"] == "development")
    confirmation = next(r for r in protocol["layouts"] if r["split"] == "confirmation")
    order = candidate_order(pool, method, seed)
    record = {"method": method, "seed": seed, "order": order, "attempts": [], "selected": None,
              "status": "no_candidates" if not pool else "running", "max_rollouts": protocol["max_search_rollouts"]}
    all_results = []
    save(output / "search.json", record)
    for index in order:
        candidate = pool[index]["candidate"]
        attempt = {"pool_index": index, "candidate": candidate, "trials": [], "confirmed": False}
        record["attempts"].append(attempt)
        valid_candidate = True
        for phase, layout, conditions in (("probe", development, protocol["probe_order"]),
                                           ("confirmation", confirmation, protocol["confirmation_order"])):
            for number, condition in enumerate(conditions):
                if sum_cost(all_results)["rollouts"] >= protocol["max_search_rollouts"]:
                    record["status"] = "budget_exhausted"
                    valid_candidate = False
                    break
                assert_sources(hashes)
                path = output / f"candidate{index:02d}" / f"{phase}{number:02d}"
                result = trial(protocol, candidate, layout, condition, path)
                all_results.append(result)
                attempt["trials"].append({"phase": phase, "success": result["success"],
                                           "trial_valid": result["trial_valid"], "cost": cost(result),
                                           "exception": result["exception"], "result_path": str(path / "result.json")})
                record["cost"] = sum_cost(all_results)
                save(output / "search.json", record)
                if not result["trial_valid"]:
                    record["status"] = "infrastructure_invalid"
                    valid_candidate = False
                    break
                if not result["success"]:
                    valid_candidate = False
                    break
            if not valid_candidate:
                break
        if valid_candidate:
            attempt["confirmed"] = True
            if record["selected"] is None:
                record["selected"] = deepcopy(pool[index])
                record["first_success_cost"] = sum_cost(all_results)
                record["first_success_attempt"] = len(record["attempts"])
            if method != "enumerate":
                record["status"] = "confirmed"
                break
        if record["status"] in ("budget_exhausted", "infrastructure_invalid"):
            break
    if record["status"] == "running":
        record["status"] = "confirmed" if record["selected"] else "pool_exhausted_abstain"
    record["cost"] = sum_cost(all_results)
    record["phase_costs"] = {phase: {key: sum(row["cost"][key] for attempt in record["attempts"]
                                                           for row in attempt["trials"] if row["phase"] == phase)
                                    for key in record["cost"]}
                             for phase in ("probe", "confirmation")}
    record["deployment_rule"] = "only selected independently confirmed candidate; otherwise abstain"
    save(output / "search.json", record)
    return record


def freeze_pool(call, protocol, output):
    try:
        if call.get("status") != "complete":
            raise CandidateRejected("Model call failed")
        parsed = parse_proposal(call, protocol)
        parsed["status"] = "parsed"
    except CandidateRejected as exc:
        parsed = {"status": "rejected_response", "reason": str(exc), "accepted": [], "rejected": []}
    save(output, parsed)
    return parsed


def prepare(args, protocol):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    paths = source_files(args.protocol, protocol)
    paths.append(Path(__file__).with_name("blocks_p1_candidates.py"))
    hashes = snapshot_sources(sorted(set(paths)), output)
    manifest = {"version": protocol["version"], "created_at": datetime.now(timezone.utc).isoformat(),
                "protocol": protocol, "protocol_sha256": digest(protocol), "source_hashes": hashes,
                "argv": sys.argv, "python": sys.version,
                "versions": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-vlm", "mujoco", "numpy", "transformers")}}
    save(output / "manifest.json", manifest)
    layout = next(r for r in protocol["layouts"] if r["split"] == "development")
    evidence, baselines = [], []
    for index, condition in enumerate(protocol["probe_order"]):
        assert_sources(hashes)
        result = trial(protocol, BASELINE, layout, condition, output / "baseline" / f"trial{index:02d}", True)
        if not result["trial_valid"]:
            raise RuntimeError("Invalid evidence trial, not a physical failure")
        evidence.append(public_evidence(result, hashlib.sha256(f"p1-observation-{index}".encode()).hexdigest()[:12], condition[1]))
        baselines.append(result)
    save(output / "public_evidence.json", evidence)
    save(output / "baseline_cost.json", sum_cost(baselines))
    messages = proposal_messages(evidence, protocol)
    save(output / "proposal_messages.json", messages)
    proposer = OfflineProposer(protocol)
    save(output / "model_identity.json", proposer.identity)
    pools = []
    for seed in protocol["generation_seeds"]:
        assert_sources(hashes)
        call_path = output / "proposals" / f"seed{seed}" / "call.json"
        call = proposer.call(messages, seed, call_path)
        parsed = freeze_pool(call, protocol, call_path.with_name("parsed.json"))
        frozen = {"seed": seed, "parsed": parsed, "call_path": str(call_path.relative_to(output)),
                  "call_sha256": file_hash(call_path), "protocol_sha256": digest(protocol), "source_hashes": hashes}
        save(call_path.with_name("pool.json"), frozen)
        pools.append({"path": str(call_path.with_name("pool.json").relative_to(output)), "sha256": file_hash(call_path.with_name("pool.json"))})
        print("PROPOSAL", seed, parsed["status"], len(parsed["accepted"]), len(parsed["rejected"]), flush=True)
    call = proposer.call(proposal_messages([], protocol), 1901, output / "insufficient_evidence" / "call.json")
    freeze_pool(call, protocol, output / "insufficient_evidence" / "parsed.json")
    assert_sources(hashes)
    save(output / "pools_frozen.json", {"protocol_sha256": digest(protocol), "source_hashes": hashes,
                                       "pools": pools, "public_evidence_sha256": file_hash(output / "public_evidence.json"),
                                       "model_identity_sha256": file_hash(output / "model_identity.json"),
                                       "baseline_cost": sum_cost(baselines)})


def load_run(output, protocol):
    frozen = json.loads((output / "pools_frozen.json").read_text())
    if frozen["protocol_sha256"] != digest(protocol):
        raise ValueError("Protocol changed after proposal freeze")
    assert_sources(frozen["source_hashes"])
    for entry in frozen["pools"]:
        if file_hash(output / entry["path"]) != entry["sha256"]:
            raise ValueError("Pool was modified")
        pool = json.loads((output / entry["path"]).read_text())
        if file_hash(output / pool["call_path"]) != pool["call_sha256"]:
            raise ValueError("Raw model output was modified")
    if file_hash(output / "public_evidence.json") != frozen["public_evidence_sha256"]:
        raise ValueError("Public evidence changed")
    return frozen


def run_searches(args, protocol):
    frozen = load_run(args.output, protocol)
    output = args.output / "searches"
    output.mkdir(exist_ok=False)
    methods = [("ranked", None)] + [("random", seed) for seed in protocol["random_seeds"]] + [("enumerate", None)]
    searches = []
    for entry in frozen["pools"]:
        pool = json.loads((args.output / entry["path"]).read_text())
        for method, seed in methods:
            name = method if seed is None else f"random{seed}"
            path = output / f"generation{pool['seed']}" / name
            record = search(protocol, pool["parsed"]["accepted"], method, path, frozen["source_hashes"], seed)
            record["generation_seed"] = pool["seed"]
            record["shared_baseline_cost"] = frozen["baseline_cost"]
            searches.append({"path": str((path / "search.json").relative_to(args.output)),
                             "sha256": file_hash(path / "search.json"), "result": record})
            print("SEARCH", pool["seed"], name, record["status"], record["cost"]["rollouts"], flush=True)
    assert_sources(frozen["source_hashes"])
    save(args.output / "searches_frozen.json", {"pools_freeze_sha256": file_hash(args.output / "pools_frozen.json"),
                                               "protocol_sha256": digest(protocol), "source_hashes": frozen["source_hashes"],
                                               "searches": searches})


def run_heldout(args, protocol):
    frozen = load_run(args.output, protocol)
    gate = json.loads((args.output / "searches_frozen.json").read_text())
    if gate["pools_freeze_sha256"] != file_hash(args.output / "pools_frozen.json"):
        raise ValueError("Search/proposal freeze mismatch")
    for entry in gate["searches"]:
        if file_hash(args.output / entry["path"]) != entry["sha256"]:
            raise ValueError("Search results changed after freeze")
    output = args.output / "heldout"
    output.mkdir(exist_ok=False)
    results = []
    for layout in protocol["layouts"]:
        if layout["split"] != "heldout":
            continue
        for index, condition in enumerate(protocol["probe_order"]):
            baseline = trial(protocol, BASELINE, layout, condition, output / "baseline" / layout["id"] / f"condition{index}", True)
            results.append({"method": "baseline", "generation_seed": None, "layout": layout["id"],
                            "condition": index, "success": baseline["success"], "trial_valid": baseline["trial_valid"], "cost": cost(baseline)})
            for entry in gate["searches"]:
                record = entry["result"]
                selected = record["selected"]
                label = record["method"] if record["seed"] is None else f"random{record['seed']}"
                row = {"method": label, "generation_seed": record["generation_seed"], "layout": layout["id"],
                       "condition": index, "abstained": selected is None}
                if selected is None or record["status"] == "infrastructure_invalid":
                    row.update(success=False, trial_valid=None, abstained=True, cost=sum_cost([]))
                else:
                    assert_sources(frozen["source_hashes"])
                    result = trial(protocol, selected["candidate"], layout, condition,
                                   output / f"generation{record['generation_seed']}" / label / layout["id"] / f"condition{index}",
                                   args.render)
                    row.update(success=result["success"], trial_valid=result["trial_valid"], cost=cost(result))
                results.append(row)
            save(output / "progress.json", results)
            print("HELDOUT", layout["id"], index, flush=True)
    assert_sources(frozen["source_hashes"])
    save(output / "results.json", {"results": results, "search_freeze_sha256": file_hash(args.output / "searches_frozen.json"),
                                   "scope": "frozen transfer only; no model calls or candidate changes on heldout"})


def negative_controls(args, protocol):
    frozen = load_run(args.output, protocol)
    output = args.output / "negative_controls"
    output.mkdir(exist_ok=False)
    layout = next(r for r in protocol["layouts"] if r["split"] == "development")
    candidates = [
        {"id": "wrong_numeric", "pre_add": [], "offset_z_m": .05, "reason": "engineered numerical negative control"},
        {"id": "blocking_pre", "pre_add": [{"action": "stack", "predicate": "empty", "args": []}], "offset_z_m": None, "reason": "engineered incompatible precondition"},
        {"id": "unknown_pred", "pre_add": [{"action": "stack", "predicate": "magic", "args": []}], "offset_z_m": None, "reason": "engineered grammar rejection"}
    ]
    records = []
    for index, candidate in enumerate(candidates):
        assert_sources(frozen["source_hashes"])
        result = trial(protocol, candidate, layout, [True, .045], output / str(index))
        records.append({"candidate": candidate, "success": result["success"], "valid": result["trial_valid"],
                        "exception": result["exception"], "cost": cost(result)})
    synthetic_pool = [{"rank": 0, "candidate": candidates[0], "origin": "engineered_negative_not_LM"}]
    refusal = search(protocol, synthetic_pool, "ranked", output / "all_bad_pool", frozen["source_hashes"])
    save(output / "results.json", {"controls": records, "all_bad_search_status": refusal["status"],
                                   "all_bad_search_selected": refusal["selected"], "all_bad_search_cost": refusal["cost"],
                                   "scope": "engineered negatives excluded from primary LM pool and search advantage"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("propose", "search", "heldout", "controls"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("blocks_p1_protocol.json"))
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    protocol = protocol_at(args.protocol)
    {"propose": prepare, "search": run_searches, "heldout": run_heldout, "controls": negative_controls}[args.phase](args, protocol)


if __name__ == "__main__":
    main()
