"""P2: staged-evidence incremental refinement vs one-shot vs rule vs fixed, same budget."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

from blocks_p0 import ROOT, digest, save
from blocks_p1_candidates import (CandidateRejected, OfflineProposer, compile_candidate,
                                  parse_proposal, recipe)
from blocks_p2_candidates import (accumulate, accumulated_candidate, describe_failure,
                                  proposal_messages_staged, staged_evidence)
from run_blocks_p0 import assert_sources, file_hash, snapshot_sources, source_files
from run_blocks_p1 import BASELINE, cost, trial


def protocol_at(path):
    protocol = json.loads(Path(path).read_text())
    p1 = json.loads(Path(__file__).with_name("blocks_p1_protocol.json").read_text())
    p0 = json.loads(Path(__file__).with_name("blocks_p0_protocol.json").read_text())
    prior_pairs = {(r["payload"], r["support"]) for r in p0["layouts"] + p1["layouts"]}
    for row in protocol["layouts"]:
        if row["split"] == "heldout" and (row["payload"], row["support"]) in prior_pairs:
            raise ValueError("Heldout pairing overlaps a prior experiment")
    return protocol


def evaluate_candidate(protocol, candidate, hashes, out_root, tag, *, render=False):
    """Run probe (dev) then confirmation (confirm); stop at first failure."""
    dev = next(r for r in protocol["layouts"] if r["split"] == "development")
    confirm = next(r for r in protocol["layouts"] if r["split"] == "confirmation")
    results, rollouts = [], 0
    for phase, layout, conditions in (("probe", dev, protocol["probe_order"]),
                                      ("confirmation", confirm, protocol["confirmation_order"])):
        for number, condition in enumerate(conditions):
            assert_sources(hashes)
            path = out_root / tag / f"{phase}{number:02d}"
            result = trial(protocol, candidate, layout, condition, path, render)
            results.append(result)
            rollouts += 1
            if not result.get("trial_valid", True):
                return {"confirmed": False, "rollouts": rollouts, "first_failure": describe_failure(result),
                        "infrastructure_invalid": True, "results": results}
            if not result.get("success"):
                return {"confirmed": False, "rollouts": rollouts, "first_failure": describe_failure(result),
                        "infrastructure_invalid": False, "results": results}
    return {"confirmed": True, "rollouts": rollouts, "first_failure": None,
            "infrastructure_invalid": False, "results": results}


def baseline_evidence(protocol, hashes, out_root):
    dev = next(r for r in protocol["layouts"] if r["split"] == "development")
    evidence, rollouts = [], 0
    for index, condition in enumerate(protocol["probe_order"]):
        assert_sources(hashes)
        result = trial(protocol, BASELINE, dev, condition, out_root / "baseline" / f"trial{index:02d}", False)
        if not result.get("trial_valid", True):
            raise RuntimeError("Invalid evidence trial")
        evidence.append(staged_evidence(result, hashlib.sha256(f"p2-observation-{index}".encode()).hexdigest()[:12], condition[1]))
        rollouts += 1
    return evidence, rollouts


def rule_diagnose(evidence):
    """Diagnose across all observed conditions: any grasp failure or placement failure."""
    grasp_failed = placement_blocked = False
    for observation in evidence:
        for _, steps in observation.get("actions", {}).items():
            for step in steps:
                if step["role"] == "grasp" and not step["success"]:
                    grasp_failed = True
                if step["role"] == "placement" and not step["success"]:
                    placement_blocked = True
    return grasp_failed, placement_blocked


def rule_candidate(grasp_failed, placement_blocked):
    pre_add = ([{"action": "stack", "predicate": "clear", "args": ["support"]}] if placement_blocked else [])
    offset = 0.0 if grasp_failed else None
    return {"id": "rule_joint", "pre_add": pre_add, "offset_z_m": offset, "reason": "deterministic staged diagnosis"}


def method_rule(protocol, hashes, output, evidence):
    grasp_failed, placement_blocked = rule_diagnose(evidence)
    candidate = rule_candidate(grasp_failed, placement_blocked)
    result = evaluate_candidate(protocol, candidate, hashes, output, "rule")
    return {"candidate": candidate, "grasp_failed": grasp_failed, "placement_blocked": placement_blocked, **result}


def method_fixed(protocol, hashes, output, budget):
    total = {"confirmed": False, "rollouts": 0, "first_failure": None, "infrastructure_invalid": False, "results": []}
    for index, candidate in enumerate(protocol["fixed_pool"]):
        if total["rollouts"] >= budget:
            total["budget_exhausted"] = True
            break
        result = evaluate_candidate(protocol, candidate, hashes, output, f"fixed{index:02d}")
        total["rollouts"] += result["rollouts"]
        if result["confirmed"]:
            total.update(confirmed=True, candidate=candidate, first_failure=None)
            break
        total["first_failure"] = result["first_failure"]
        if result.get("infrastructure_invalid"):
            total["infrastructure_invalid"] = True
            break
    return total


def method_onetime(protocol, hashes, output, evidence, proposer, seed, budget):
    messages = proposal_messages_staged(evidence, protocol, max_candidates=protocol["max_candidates"])
    call_path = output / "onetime" / f"seed{seed}" / "call.json"
    call = proposer.call(messages, seed, call_path)
    try:
        parsed = parse_proposal(call, protocol)
        parsed["status"] = "parsed"
    except CandidateRejected as exc:
        parsed = {"status": "rejected_response", "reason": str(exc), "accepted": [], "rejected": []}
    save(call_path.with_name("parsed.json"), parsed)
    record = {"seed": seed, "parsed_status": parsed["status"],
              "candidates": [p["candidate"] for p in parsed.get("accepted", [])],
              "confirmed": False, "rollouts": 0, "first_failure": None, "infrastructure_invalid": False}
    for rank, item in enumerate(parsed.get("accepted", [])):
        if record["rollouts"] >= budget:
            record["budget_exhausted"] = True
            break
        result = evaluate_candidate(protocol, item["candidate"], hashes, output, f"onetime/seed{seed}/rank{rank}")
        record["rollouts"] += result["rollouts"]
        if result["confirmed"]:
            record.update(confirmed=True, candidate=item["candidate"], first_failure=None)
            break
        record["first_failure"] = result["first_failure"]
        if result.get("infrastructure_invalid"):
            record["infrastructure_invalid"] = True
            break
    save(output / "onetime" / f"seed{seed}" / "record.json", record)
    return record


def method_incremental(protocol, hashes, output, evidence, proposer, seed, budget):
    accumulated = {"pre_add": [], "offset_z_m": None}
    last_failure = None
    record = {"seed": seed, "rounds": [], "confirmed": False, "rollouts": 0,
              "first_failure": None, "infrastructure_invalid": False}
    for round_index in range(protocol["max_rounds"]):
        if record["rollouts"] >= budget:
            record["budget_exhausted"] = True
            break
        messages = proposal_messages_staged(evidence, protocol, accumulated=accumulated, last_failure=last_failure)
        call_path = output / "incremental" / f"seed{seed}" / f"round{round_index}" / "call.json"
        call = proposer.call(messages, seed + round_index, call_path)
        try:
            parsed = parse_proposal(call, protocol)
            parsed["status"] = "parsed"
        except CandidateRejected as exc:
            parsed = {"status": "rejected_response", "reason": str(exc), "accepted": [], "rejected": []}
        save(call_path.with_name("parsed.json"), parsed)
        if parsed["status"] != "parsed" or not parsed.get("accepted"):
            record["rounds"].append({"round": round_index, "status": "no_candidate_abstain", "parsed_status": parsed["status"]})
            break
        proposed = parsed["accepted"][0]["candidate"]
        accumulated = accumulate(proposed, accumulated)
        full = accumulated_candidate(accumulated)
        result = evaluate_candidate(protocol, full, hashes, output, f"incremental/seed{seed}/round{round_index}")
        record["rollouts"] += result["rollouts"]
        record["rounds"].append({"round": round_index, "proposed": proposed, "accumulated": accumulated,
                                 "confirmed": result["confirmed"], "rollouts": result["rollouts"],
                                 "first_failure": result["first_failure"]})
        if result["confirmed"]:
            record.update(confirmed=True, candidate=full, first_failure=None)
            break
        record["first_failure"] = result["first_failure"]
        last_failure = result["first_failure"]
        if result.get("infrastructure_invalid"):
            record["infrastructure_invalid"] = True
            break
    save(output / "incremental" / f"seed{seed}" / "record.json", record)
    return record


def run_heldout(protocol, hashes, output, winners):
    rows = []
    for method, candidate in winners.items():
        if candidate is None:
            continue
        for layout in protocol["layouts"]:
            if layout["split"] != "heldout":
                continue
            for index, condition in enumerate(protocol["probe_order"]):
                assert_sources(hashes)
                result = trial(protocol, candidate, layout, condition,
                               output / "heldout" / method / layout["id"] / f"condition{index}", False)
                rows.append({"method": method, "layout": layout["id"], "condition": index,
                             "success": result["success"], "trial_valid": result["trial_valid"],
                             "cost": cost(result)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("blocks_p2_protocol.json"))
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    protocol = protocol_at(args.protocol)
    if args.output.exists():
        parser.error("Use a new output directory")
    args.output.mkdir(parents=True)

    paths = source_files(args.protocol, protocol) + [Path(__file__).with_name("blocks_p2_candidates.py"),
                                                     Path(__file__).with_name("blocks_p1_candidates.py")]
    hashes = snapshot_sources(sorted(set(paths)), args.output)
    save(args.output / "manifest.json", {"protocol": protocol, "protocol_sha256": digest(protocol),
                                          "source_hashes": hashes, "argv": sys.argv})

    evidence, baseline_rollouts = baseline_evidence(protocol, hashes, args.output)
    save(args.output / "baseline_evidence.json", {"evidence": evidence, "rollouts": baseline_rollouts})
    budget = protocol["max_rollouts_per_method"]

    results = {"shared_baseline_rollouts": baseline_rollouts, "budget_per_method": budget}

    results["rule"] = method_rule(protocol, hashes, args.output, evidence)

    results["fixed"] = method_fixed(protocol, hashes, args.output, budget)

    proposer = OfflineProposer(protocol)
    save(args.output / "model_identity.json", proposer.identity)
    results["onetime"] = []
    results["incremental"] = []
    for seed in protocol["generation_seeds"]:
        results["onetime"].append(method_onetime(protocol, hashes, args.output, evidence, proposer, seed, budget))
        results["incremental"].append(method_incremental(protocol, hashes, args.output, evidence, proposer, seed, budget))

    winners = {"rule": results["rule"].get("candidate") if results["rule"].get("confirmed") else None,
               "fixed": results["fixed"].get("candidate") if results["fixed"].get("confirmed") else None,
               "onetime": next((r.get("candidate") for r in results["onetime"] if r.get("confirmed")), None),
               "incremental": next((r.get("candidate") for r in results["incremental"] if r.get("confirmed")), None)}
    results["winners"] = winners
    results["heldout"] = run_heldout(protocol, hashes, args.output, winners)

    assert_sources(hashes)
    save(args.output / "results.json", results)
    for name in ("rule", "fixed", "onetime", "incremental"):
        value = results[name]
        if isinstance(value, list):
            print(name, [{"confirmed": r["confirmed"], "rollouts": r["rollouts"]} for r in value], flush=True)
        else:
            print(name, {"confirmed": value.get("confirmed"), "rollouts": value.get("rollouts")}, flush=True)


if __name__ == "__main__":
    main()
