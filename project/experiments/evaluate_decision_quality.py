"""Offline decision-output acceptance: base vs trained adapter vs deterministic rule.

Scope: does the model emit legal, faithful, decision-bearing structured output
(not just text)? This is a dev-asset checkpoint on the 16 acceptance-slice
samples, NOT an OOD generalization test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

from blocks_p0 import save
from collect_pick_groups import file_hash
from export_pick_groups import diagnose_target, update_target
from train_pick_groups_lora import load_manifest

INPUT_ONLY_KEYS = {"model", "source", "before", "after", "api_trace", "exception", "check",
                   "physical_metrics", "evidence_ids", "observed", "baseline", "intervention",
                   "intervention_observed", "proposed_intervention"}

CORE_KEYS = {
    "diagnosis": {"decision", "physical_pick_success", "contract_consistent"},
    "forecast": {"physical_pick_success", "contract_consistent", "joint_success"},
    "update": {"decision", "physical_improved", "contract_improved"},
}


def parse_json(text):
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    candidates = [text]
    fence = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.S)
    if fence:
        candidates.insert(0, fence.group(1).strip())
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except (ValueError, TypeError):
            continue
    return None


def extract_payload(human_value):
    """Recover the structured input payload embedded after the task-type marker."""
    try:
        after = human_value.split("任务类型=", 1)[1]
        payload_json = after.split("\n", 1)[1]
        return json.loads(payload_json)
    except (IndexError, ValueError, TypeError):
        return None


def rule_decision(kind, payload):
    if kind == "diagnosis" and payload and "observed" in payload:
        return diagnose_target(payload["observed"])
    if kind == "update" and payload and "baseline" in payload and "intervention_observed" in payload:
        return update_target(payload["baseline"], payload["intervention_observed"])
    if kind == "forecast" and payload and "observed" in payload:
        base = payload["observed"]
        physical = base.get("physical_pick_success")
        contract = base.get("contract_consistent")
        return {"physical_pick_success": physical, "contract_consistent": contract,
                "joint_success": bool(physical and contract),
                "scope": "no_change_trivial_baseline"}
    return None


def score_output(kind, target, parsed):
    score = {"json_valid": parsed is not None, "core_keys_present": False,
             "decision_match": None, "outcome_match": False, "echoed_input_fields": [],
             "top_level_keys": []}
    if parsed is None:
        return score
    score["top_level_keys"] = sorted(parsed.keys())
    score["core_keys_present"] = CORE_KEYS[kind] <= set(parsed)
    score["echoed_input_fields"] = sorted(set(parsed) & INPUT_ONLY_KEYS)
    if kind == "forecast":
        score["outcome_match"] = all(parsed.get(k) == target.get(k) for k in CORE_KEYS[kind])
    elif kind == "diagnosis":
        score["decision_match"] = parsed.get("decision") == target.get("decision")
        score["outcome_match"] = (parsed.get("physical_pick_success") == target.get("physical_pick_success")
                                  and parsed.get("contract_consistent") == target.get("contract_consistent"))
    elif kind == "update":
        score["decision_match"] = parsed.get("decision") == target.get("decision")
        score["outcome_match"] = (parsed.get("physical_improved") == target.get("physical_improved")
                                  and parsed.get("contract_improved") == target.get("contract_improved"))
    return score


def generate(model, processor, dataset, prompt, images, max_tokens=600):
    from mlx_vlm import generate as gen
    started = time.monotonic()
    response = gen(model, processor, prompt, image=[str(dataset / p) for p in images],
                   max_tokens=max_tokens, temperature=0., verbose=False)
    return {"raw": response.text, "tokens": int(response.generation_tokens),
            "finish_reason": getattr(response, "finish_reason", None),
            "wall_seconds": time.monotonic() - started}


def run_model(model, processor, dataset, samples, output, max_tokens):
    rows = []
    for entry in samples:
        result = {"sample_id": entry["sample_id"], "kind": entry["kind"], "split": entry["split"]}
        try:
            result["generation"] = generate(model, processor, dataset, entry["prompt"], entry["images"], max_tokens)
            parsed = parse_json(result["generation"]["raw"])
            result["score"] = score_output(entry["kind"], entry["target_json"], parsed)
            result["parsed"] = parsed
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(result)
        print("GEN", entry["kind"], entry["split"], result["score"]["json_valid"],
              result["score"]["decision_match"], result["score"]["outcome_match"],
              result["score"]["echoed_input_fields"], flush=True)
    return rows


def summarize(rows):
    by_kind = {}
    for row in rows:
        kind = row["kind"]
        bucket = by_kind.setdefault(kind, {"n": 0, "json_valid": 0, "core_keys_present": 0,
                                          "decision_match": 0, "decision_checked": 0,
                                          "outcome_match": 0, "echo_any": 0})
        score = row.get("score", {})
        bucket["n"] += 1
        bucket["json_valid"] += bool(score.get("json_valid"))
        bucket["core_keys_present"] += bool(score.get("core_keys_present"))
        bucket["outcome_match"] += bool(score.get("outcome_match"))
        bucket["echo_any"] += bool(score.get("echoed_input_fields"))
        if score.get("decision_match") is not None:
            bucket["decision_checked"] += 1
            bucket["decision_match"] += bool(score["decision_match"])
    return by_kind


def load_gen_model(snapshot, adapter=None):
    import mlx.core as mx
    import mlx_vlm.models.rope_utils as rope
    from mlx_vlm import load
    rope._HAS_METAL = False
    model, _ = load(snapshot, adapter_path=str(adapter) if adapter else None,
                    processor_config={"local_files_only": True})
    for _, module in model.named_modules():
        if hasattr(module, "fused_apply"):
            module.fused_apply = False
            module._compiled_apply = {}
    model.eval()
    mx.eval(model.parameters())
    return model


def rule_rows(samples):
    rows = []
    for entry in samples:
        rule = entry["rule"]
        if rule is None:
            continue
        rows.append({"sample_id": entry["sample_id"], "kind": entry["kind"], "split": entry["split"],
                     "score": score_output(entry["kind"], entry["target_json"], rule),
                     "parsed": rule, "generation": {"raw": json.dumps(rule, ensure_ascii=False), "source": "deterministic_rule"}})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/pick_lora_acceptance_20260928_v2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--method", choices=("rule", "base", "adapter", "all"), default="all")
    args = parser.parse_args()
    args.run = args.run.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / f"{args.method}.json").exists():
        parser.error("Method result already exists; use a new output dir or method")

    manifest = load_manifest(args.run)
    dataset = Path(manifest["dataset"])
    index = json.loads((dataset / "sample_index.json").read_text())
    rows = {split: [json.loads(line) for line in (dataset / f"{split}.jsonl").read_text().splitlines()]
            for split in {m["split"] for m in index}}
    by_id = {m["sample_id"]: rows[m["split"]][m["line"]] for m in index}

    samples = []
    for entry in manifest["samples"]:
        record = by_id[entry["sample_id"]]
        human = record["conversations"][1]["value"]
        target = json.loads(record["conversations"][-1]["value"])
        payload = extract_payload(human)
        samples.append({"sample_id": entry["sample_id"], "kind": entry["kind"], "split": entry["split"],
                        "prompt": entry["prompt"], "images": entry["images"], "target_json": target,
                        "payload": payload, "rule": rule_decision(entry["kind"], payload)})

    base_result = {"run": str(args.run), "run_manifest_sha256": file_hash(args.run / "manifest.json"),
                   "sample_count": len(samples), "max_tokens": args.max_tokens,
                   "scope": "dev-asset decision-output acceptance; not OOD generalization",
                   "samples_meta": [{"sample_id": s["sample_id"], "kind": s["kind"], "split": s["split"]} for s in samples]}
    save(args.output / "samples.json", base_result)

    snapshot = Path(manifest["snapshot"])
    from mlx_vlm.utils import load_processor
    processor = load_processor(snapshot, add_detokenizer=True,
                               eos_token_ids=None, local_files_only=True,
                               min_pixels=128 * 28 * 28, max_pixels=256 * 28 * 28)

    if args.method in ("rule", "all"):
        rule = rule_rows(samples)
        save(args.output / "rule.json", {"rows": rule, "summary": summarize(rule)})
        print("=== rule ===")
        print(json.dumps(summarize(rule), ensure_ascii=False, indent=2))

    import mlx.core as mx
    if args.method in ("base", "all"):
        model = load_gen_model(snapshot, None)
        rows = run_model(model, processor, dataset, samples, args.output, args.max_tokens)
        save(args.output / "base.json", {"rows": rows, "summary": summarize(rows)})
        del model
        mx.clear_cache()
        print("=== base_unadapted ===")
        print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))

    if args.method in ("adapter", "all"):
        model = load_gen_model(snapshot, args.run / "adapter")
        rows = run_model(model, processor, dataset, samples, args.output, args.max_tokens)
        save(args.output / "adapter.json", {"rows": rows, "summary": summarize(rows)})
        del model
        mx.clear_cache()
        print("=== trained_12step_adapter ===")
        print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
