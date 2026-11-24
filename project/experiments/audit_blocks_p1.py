"""Read-only verification and cost accounting for frozen P1 experiment artifacts."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from blocks_p0 import digest, save
from blocks_p1_candidates import compile_candidate, recipe, strict_json
from run_blocks_p1 import (checked_path, cost, file_hash, load_run, load_search_gate,
                          protocol_at, sum_cost)


def audit(output):
    output = Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    protocol = manifest["protocol"]
    frozen = load_run(output, protocol, verify_current=False)
    gate = load_search_gate(output, protocol, frozen)
    candidates, proposals = [], []
    for entry in frozen["pools"]:
        pool = json.loads(checked_path(output, entry["path"]).read_text())
        call = json.loads(checked_path(output, pool["call_path"]).read_text())
        candidates.extend(pool["parsed"]["accepted"])
        proposals.append({"seed": pool["seed"], "valid_candidates": len(pool["parsed"]["accepted"]),
                          "rejected_candidates": pool["parsed"]["rejected"],
                          "generation_tokens": call.get("generation_tokens"), "prompt_tokens": call.get("prompt_tokens"),
                          "finish_reason": call.get("finish_reason"), "model_seconds": call["wall_seconds"],
                          "call_sha256": pool["call_sha256"]})
    counts = {}
    all_trials = []
    for category, pattern in (("baseline", "baseline/*/result.json"),
                              ("search", "searches/*/*/*/*/result.json"),
                              ("heldout", "heldout/**/result.json"),
                              ("negative_controls", "negative_controls/**/result.json")):
        paths = sorted(output.glob(pattern))
        trials = [json.loads(path.read_text()) for path in paths]
        counts[category] = {"records": len(paths), "cost": sum_cost(trials),
                            "valid": sum(t["trial_valid"] for t in trials), "successes": sum(t["success"] for t in trials)}
        all_trials.extend(zip(paths, trials))
    search_summaries = []
    for entry in gate["searches"]:
        record = entry["result"]
        trials = []
        for attempt in record["attempts"]:
            for row in attempt["trials"]:
                recorded = Path(row["result_path"])
                if not recorded.is_absolute():
                    from blocks_p0 import ROOT
                    recorded = ROOT / recorded
                path = checked_path(output, recorded)
                result = json.loads(path.read_text())
                if (result["success"] != row["success"] or result["trial_valid"] != row["trial_valid"]
                        or cost(result) != row["cost"]):
                    raise ValueError("Search trace and physical result disagree")
                if result.get("candidate_recipe_sha256") != digest(recipe(attempt["candidate"])):
                    raise ValueError("Physical execution recipe differs from model candidate")
                if result.get("physical_rollouts", 1):
                    saved_models = json.loads((path.parent / "models.json").read_text())
                    saved_sources = json.loads((path.parent / "policies.json").read_text())
                    if digest(saved_models) != result["models_sha256"] or digest(saved_sources) != result["policies_sha256"]:
                        raise ValueError("Executed model/policy artifact changed")
                trials.append(result)
        if sum_cost(trials) != record["cost"]:
            raise ValueError("Reported interaction cost differs from actual trials")
        search_summaries.append({"generation_seed": record["generation_seed"], "method": record["method"],
                                 "random_seed": record["seed"], "order": record["order"], "status": record["status"],
                                 "selected": record["selected"], "search_cost": record["cost"],
                                 "confirmation_cost": record["phase_costs"]["confirmation"],
                                 "baseline_plus_search_rollouts": frozen["baseline_cost"]["rollouts"] + record["cost"]["rollouts"],
                                 "first_success_cost": record.get("first_success_cost")})
    heldout = json.loads((output / "heldout/results.json").read_text())
    if heldout["search_freeze_sha256"] != file_hash(output / "searches_frozen.json"):
        raise ValueError("Heldout report references a different search freeze")
    rows = heldout["results"]
    abstentions = [row for row in rows if row.get("abstained")]
    executed = [row for row in rows if not row.get("abstained")]
    safety = {"weld_violations": 0, "successful_transition_mismatches": 0, "infrastructure_invalid": 0}
    for _, result in all_trials:
        if not result["trial_valid"]:
            safety["infrastructure_invalid"] += 1
        metrics = (result.get("final_check") or {}).get("legacy_metrics", {})
        safety["weld_violations"] += bool(metrics.get("strict_weld_violation"))
        if result["success"]:
            safety["successful_transition_mismatches"] += sum(not s["check"]["strict_CP_transition_match"] for s in result["steps"])
    empty_call = json.loads((output / "insufficient_evidence/call.json").read_text())
    empty_parsed = json.loads((output / "insufficient_evidence/parsed.json").read_text())
    try:
        empty_raw = strict_json(empty_call["raw"])
        intended_abstention = isinstance(empty_raw, dict) and empty_raw.get("decision") == "abstain"
    except ValueError:
        intended_abstention = False
    return {"scope": "frozen one-shot candidate generation and sequential search; no feedback regeneration, no real robot",
            "artifact_root_hashes": {name: file_hash(output / name) for name in
                                     ("manifest.json", "pools_frozen.json", "searches_frozen.json", "heldout/results.json")},
            "source_snapshots_and_raw_pool_binding_verified": True,
            "proposals": proposals, "accepted_entries": len(candidates),
            "distinct_accepted_recipes": len({digest(recipe(c["candidate"])) for c in candidates}),
            "searches": search_summaries, "accounting": counts,
            "total_actual_rollouts": sum(c["cost"]["rollouts"] for c in counts.values()),
            "safety_checks": safety,
            "heldout": {"physical_baseline_executions": sum(r["method"] == "baseline" for r in executed),
                        "baseline_successes": sum(bool(r["success"]) for r in executed if r["method"] == "baseline"),
                        "physical_repaired_executions": sum(r["method"] != "baseline" for r in executed),
                        "abstained_deployment_records": len(abstentions),
                        "note": "abstention rows are not physical failures or independent tests"},
            "insufficient_evidence": {"raw_intent_abstain": intended_abstention,
                                      "parser_status": empty_parsed["status"], "parser_reason": empty_parsed.get("reason"),
                                      "generation_tokens": empty_call.get("generation_tokens")},
            "interpretation": "No speed advantage may be claimed unless a repair is confirmed; failed search cost is not time-to-success"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new audit output; never overwrite historical evidence")
    result = audit(args.run)
    save(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
