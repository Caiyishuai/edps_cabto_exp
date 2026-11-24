"""Causally scoped diagnosis/update/forecast examples from frozen pick groups."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import uuid

from PIL import Image
import numpy as np

from blocks_p0 import diagnose, digest, save
from collect_pick_groups import file_hash

SYSTEM = ("你分析单次机器人抓取。输入状态与传感信息可靠，但动作契约和程序可能有缺陷。"
          "区分物理执行结果与符号契约一致性；只引用已提供证据，不把一次失败称为物理不可能。"
          "输出简短结构化JSON；没有经过独立验证的修订不得标为可部署。")


def checked(root, path):
    value = (Path(root) / path).resolve()
    if not value.is_relative_to(Path(root).resolve()):
        raise ValueError("Artifact path outside root")
    return value


def public_trial(result):
    fields = ("model", "source", "before", "after", "api_trace", "exception", "check",
              "physical_pick_success", "contract_consistent", "physical_metrics")
    public = {key: deepcopy(result[key]) for key in fields}
    for key in ("before", "after"):
        public[key].pop("active_welds", None)
    public["evidence_ids"] = {"E1": "before", "E2": "after", "E3": "api_trace", "E4": "check"}
    return public


def diagnose_target(public):
    physical = public["physical_pick_success"]
    contract = public["contract_consistent"]
    failed = [row["method"] for row in public["api_trace"] if row["status"] == "failed"]
    base = {"failed_stage": failed[-1] if failed else None,
            "physical_pick_success": physical, "contract_consistent": contract,
            "evidence_refs": ["E2", "E3", "E4"], "deployable_repair": False}
    if physical and contract:
        return {**base, "decision": "no_repair_on_this_trial", "hypotheses": [],
                "scope": "single_observed_pick_not_whole_task_or_universal_skill"}
    if physical:
        return {**base, "decision": "inspect_contract_effects",
                "hypotheses": [{"kind": "contract_transition_mismatch", "support": "observed", "unique_cause": False}],
                "next_test": {"kind": "contract_only_recheck_then_matched_execution", "freeze": ["policy", "initial_context", "observer"]}}
    return {**base, "decision": "request_discriminating_trial",
            "hypotheses": [{"kind": "policy_or_controller_context_mismatch", "support": "unconfirmed"},
                           {"kind": "additional_contract_defect", "support": "not_excluded"}],
            "next_test": {"kind": "bounded_policy_parameter_or_code_test", "freeze": ["model", "initial_context", "observer"]},
            "scope": "no_unique_cause_from_one_failed_execution"}


def intervention_descriptor(base, candidate):
    added = sorted(set(candidate["model"]["del"]) - set(base["model"]["del"]))
    removed = sorted(set(base["model"]["del"]) - set(candidate["model"]["del"]))
    return {"model_delete_added": added, "model_delete_removed": removed,
            "candidate_model": deepcopy(candidate["model"]), "candidate_source": candidate["source"],
            "policy_changed": candidate["source"] != base["source"],
            "reset_mode": "matched_initial_context_not_failed_state_recovery"}


def update_target(base_public, next_public):
    physical_gain = not base_public["physical_pick_success"] and next_public["physical_pick_success"]
    contract_gain = not base_public["contract_consistent"] and next_public["contract_consistent"]
    success = next_public["physical_pick_success"] and next_public["contract_consistent"]
    return {"decision": "candidate_supported_pending_independent_validation" if success else
            "partial_progress" if physical_gain or contract_gain else "no_improvement_continue_or_abstain",
            "physical_improved": physical_gain, "contract_improved": contract_gain,
            "observed_physical_success": next_public["physical_pick_success"],
            "observed_contract_consistency": next_public["contract_consistent"],
            "evidence_refs": ["baseline:E2", "baseline:E4", "intervention:E2", "intervention:E4"],
            "deployable_repair": False, "scope": "one_matched_intervention_not_unique_root_cause"}


def frame_selection(result, terminal_only=False):
    frames = result["frames"]
    if not frames or frames[0]["stage"] != "initial":
        raise ValueError("Missing initial frame")
    if terminal_only:
        return [frames[-1]]
    middle = next((frame for frame in frames if frame["stage"] == "descend"), frames[0])
    indices = sorted({frames[0]["index"], middle["index"], frames[-1]["index"]})
    return [frame for frame in frames if frame["index"] in indices]


def make_record(kind, payload, target, attachments, output, run):
    images, metadata = [], []
    prefix = []
    trial_order = list(dict.fromkeys(trial["trial_id"] for _, trial, _ in attachments))
    for event in attachments:
        result_path, trial, frame = event
        role = "baseline" if trial["trial_id"] == trial_order[0] else "intervention"
        for media in frame["images"]:
            image_path = checked(result_path.parent, media["path"])
            if file_hash(image_path) != media["sha256"]:
                raise ValueError("Source image changed")
            relative = f"media/{media['sha256']}.png"
            target_path = output / relative
            if not target_path.exists():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(image_path, target_path)
            prefix.append(f"{len(images)+1}. {role} / {frame['stage']} / {media['camera']} / t={media['sim_time']:.3f}: <image>")
            images.append(relative)
            metadata.append({"trial_id": trial["trial_id"], "frame_index": frame["index"], "stage": frame["stage"],
                             "sim_time": media["sim_time"], "sha256": media["sha256"], "camera": media["camera"]})
    question = "\n".join(prefix) + "\n任务类型=" + kind + "\n" + json.dumps(payload, ensure_ascii=False)
    record = {"conversations": [{"from": "system", "value": SYSTEM}, {"from": "human", "value": question},
                                {"from": "gpt", "value": json.dumps(target, ensure_ascii=False)}], "images": images}
    return record, metadata


def export(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Use a new dataset output")
    frozen = json.loads((run / "collection_frozen.json").read_text())
    if file_hash(run / "manifest.json") != frozen["manifest_sha256"]:
        raise ValueError("Manifest mismatch")
    manifest = json.loads((run / "manifest.json").read_text())
    for name, sha in manifest["source_hashes"].items():
        if file_hash(checked(run / "source_snapshot", name)) != sha:
            raise ValueError("Source snapshot mismatch")
    output.mkdir(parents=True)
    datasets = defaultdict(list)
    index, group_summaries = [], []
    for group_entry in frozen["groups"]:
        group_path = checked(run, group_entry["path"])
        if file_hash(group_path) != group_entry["sha256"]:
            raise ValueError("Group modified")
        group = json.loads(group_path.read_text())
        results = {}
        for branch, entry in group["trials"].items():
            path = checked(run, entry["path"])
            if file_hash(path) != entry["sha256"]:
                raise ValueError("Trial modified")
            results[branch] = (path, json.loads(path.read_text()))
        valid = group["paired_layer_invariance"] and all(r[1]["valid"] and r[1]["artifact_complete"] for r in results.values())
        summary = {"group_id": group["id"], "context_id": group["context_id"], "split": group["split"],
                   "scene": group["scene"], "object": group["object"], "valid": valid,
                   "condition_private": group["condition"],
                   "physical": {k: r[1]["physical_pick_success"] for k, r in results.items()},
                   "contract": {k: r[1]["contract_consistent"] for k, r in results.items()}}
        summary["response_class_not_unique_cause"] = diagnose({k: r[1]["joint_success"] for k, r in results.items()}) if valid else "invalid"
        group_summaries.append(summary)
        if not valid:
            continue
        base_path, base = results["none"]
        base_public = public_trial(base)
        def append(kind, payload, target, attachments, visible, future, provenance):
            record, media = make_record(kind, payload, target, attachments, output, run)
            line = len(datasets[group["split"]])
            datasets[group["split"]].append(record)
            cutoff = {result["trial_id"]: result["after"]["sim_time"] for _, result in visible}
            index.append({"sample_id": uuid.uuid4().hex, "split": group["split"], "line": line,
                          "kind": kind, "group_id": group["id"], "context_id": group["context_id"],
                          "input_trial_ids": list(cutoff), "cutoff_by_trial": cutoff,
                          "teacher_only_future_trial_ids": [result["trial_id"] for _, result in future],
                          "media": media, "target_provenance": provenance,
                          "record_sha256": digest(record)})
        selected_base = [(base_path, base, frame) for frame in frame_selection(base)]
        append("diagnosis", {"observed": base_public}, diagnose_target(base_public), selected_base,
               [(base_path, base)], [], "deterministic_evidence_projection_with_handwritten_conservative_suggestions")
        for branch in ("model_only", "policy_only", "cross_level"):
            path, candidate = results[branch]
            descriptor = intervention_descriptor(base, candidate)
            public_next = public_trial(candidate)
            forecast = {"physical_pick_success": candidate["physical_pick_success"],
                        "contract_consistent": candidate["contract_consistent"],
                        "joint_success": candidate["joint_success"],
                        "scope": "single_realized_intervention_outcome_not_calibrated_probability"}
            append("forecast", {"observed": base_public, "proposed_intervention": descriptor}, forecast,
                   selected_base, [(base_path, base)], [(path, candidate)], "paired_execution_hindsight_outcome")
            frames = [(base_path, base, frame) for frame in frame_selection(base, terminal_only=True)]
            middle = next((frame for frame in candidate["frames"] if frame["stage"] == "close_gripper"), candidate["frames"][0])
            frames += [(path, candidate, frame) for frame in candidate["frames"]
                       if frame["index"] in {middle["index"], candidate["frames"][-1]["index"]}]
            append("update", {"baseline": base_public, "intervention": descriptor, "intervention_observed": public_next},
                   update_target(base_public, public_next), frames, [(base_path, base), (path, candidate)], [],
                   "two_observed_trial_comparison_not_unseen_root_cause")
    for split, rows in datasets.items():
        (output / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    save(output / "sample_index.json", index)
    save(output / "group_audit.json", group_summaries)
    save(output / "dataset_info.json", {f"pick_{split}": {"file_name": f"{split}.jsonl", "formatting": "sharegpt",
                                                       "columns": {"messages": "conversations", "images": "images"},
                                                       "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt", "system_tag": "system"}}
                                       for split in datasets})
    summary = {"collection_freeze_sha256": file_hash(run / "collection_frozen.json"),
               "expected_group_splits": {g["id"]: g["split"] for g in manifest["groups"]},
               "expected_group_contexts": {g["id"]: g["context_id"] for g in manifest["groups"]},
               "groups": len(group_summaries), "valid_groups": sum(x["valid"] for x in group_summaries),
               "samples": {k: len(v) for k, v in datasets.items()}, "kinds": dict(Counter(x["kind"] for x in index)),
               "distinct_contexts": len({x["context_id"] for x in index}),
               "note": "same generated skill family across splits; validation is not unseen-scene evidence",
               "teacher": "deterministic facts and conservative suggested tests; outcome/update targets use actual paired execution",
               "training_readiness": "pipeline_acceptance_only_not_production_causal_supervision",
               "data_files": {p.name: file_hash(p) for p in output.glob("*.json*")}}
    save(output / "dataset_manifest.json", summary)
    return summary


def validate_inventory(manifest, index, datasets):
    expected = manifest.get("expected_group_splits", {})
    if not expected or not index:
        raise ValueError("Empty or unregistered dataset cannot pass")
    if manifest["valid_groups"] != len(expected) or manifest["groups"] != len(expected):
        raise ValueError("Missing or invalid collection groups")
    expected_splits = set(expected.values())
    if set(datasets) != expected_splits or any(not rows for rows in datasets.values()):
        raise ValueError("Missing or empty expected split")
    per_group = defaultdict(Counter)
    seen = set()
    for item in index:
        group = item["group_id"]
        if expected.get(group) != item["split"]:
            raise ValueError("Group split differs from collection manifest")
        if manifest["expected_group_contexts"].get(group) != item["context_id"]:
            raise ValueError("Group context differs from collection manifest")
        key = (item["split"], item["line"])
        if key in seen or not 0 <= item["line"] < len(datasets[item["split"]]):
            raise ValueError("Duplicate or invalid dataset line")
        seen.add(key)
        per_group[group][item["kind"]] += 1
    if set(per_group) != set(expected) or any(c != Counter(diagnosis=1, forecast=3, update=3) for c in per_group.values()):
        raise ValueError("Incomplete diagnosis/update/forecast coverage")
    if len(index) != sum(len(rows) for rows in datasets.values()):
        raise ValueError("Dataset lines not fully indexed")


def audit_dataset(output):
    output = Path(output).resolve()
    manifest = json.loads((output / "dataset_manifest.json").read_text())
    for name, sha in manifest["data_files"].items():
        if file_hash(output / name) != sha:
            raise ValueError("Dataset file modified")
    index = json.loads((output / "sample_index.json").read_text())
    datasets = {split: [json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines()]
                for split in manifest["samples"]}
    validate_inventory(manifest, index, datasets)
    groups, contexts, pixels = defaultdict(set), defaultdict(set), defaultdict(set)
    for meta in index:
        split = meta["split"]
        record = datasets[split][meta["line"]]
        if digest(record) != meta["record_sha256"]:
            raise ValueError("Sample digest mismatch")
        human = record["conversations"][1]["value"]
        if len(record["images"]) != human.count("<image>") or len(record["images"]) != len(meta["media"]):
            raise ValueError("Image placeholder/metadata mismatch")
        json.loads(record["conversations"][-1]["value"])
        for key in ("condition_private", "evaluation_only", "response_class_not_unique_cause", "source_snapshot", "group_id", "context_id"):
            if key in human:
                raise ValueError("Private metadata in student input")
        if meta["kind"] == "forecast" and set(meta["input_trial_ids"]) & set(meta["teacher_only_future_trial_ids"]):
            raise ValueError("Forecast future trial visible")
        for image, info in zip(record["images"], meta["media"]):
            if info["trial_id"] not in meta["input_trial_ids"] or info["sim_time"] > meta["cutoff_by_trial"][info["trial_id"]] + 1e-9:
                raise ValueError("Media outside causal observation cutoff")
            path = checked(output, image)
            if file_hash(path) != info["sha256"]:
                raise ValueError("Media modified")
            with Image.open(path) as im:
                data = np.asarray(im)
                if data.max() == data.min():
                    raise ValueError("Black or constant image")
                pixel_hash = hashlib.sha256(data.tobytes()).hexdigest()
            pixels[pixel_hash].add(split)
        groups[meta["group_id"]].add(split)
        contexts[meta["context_id"]].add(split)
    if any(len(x) != 1 for x in list(groups.values()) + list(contexts.values())):
        raise ValueError("Group/context leakage across splits")
    if any(len(x) != 1 for x in pixels.values()):
        raise ValueError("Identical image pixels cross splits")
    return {"passed": True, "samples": len(index), "groups": len(groups), "contexts": len(contexts),
            "unique_pixel_images": len(pixels), "group_isolation": True, "image_time_cutoffs": True,
            "future_intervention_frames_excluded": True, "input_private_keys_excluded": True,
            "multi_image_placeholders": True, "processor_mask_checked": False,
            "scope": "automated structural/data audit; suggested diagnosis labels still need human review"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = export(args.run, args.output)
    audit = audit_dataset(args.output)
    save(args.output / "audit.json", audit)
    print(json.dumps({"dataset": summary, "audit": audit}, ensure_ascii=False, indent=2))
