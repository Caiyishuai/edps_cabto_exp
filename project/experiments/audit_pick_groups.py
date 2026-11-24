"""Verify frozen raw group evidence, output coverage and example provenance."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from blocks_p0 import digest, save
from collect_pick_groups import file_hash
from core.cover_cabto_api import transition_check
from export_pick_groups import audit_dataset, checked


def audit(run, dataset):
    run, dataset = Path(run).resolve(), Path(dataset).resolve()
    frozen = json.loads((run / "collection_frozen.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    if file_hash(run / "manifest.json") != frozen["manifest_sha256"] or not frozen["source_unchanged"]:
        raise ValueError("Collection manifest/freeze invalid")
    expected = {g["id"]: g for g in manifest["groups"]}
    if not expected or len(frozen["groups"]) != len(expected):
        raise ValueError("Collection group count incomplete")
    outcomes, trial_ids, frame_hashes, seen = defaultdict(Counter), set(), set(), set()
    trials_count = frames_count = 0
    for item in frozen["groups"]:
        group_path = checked(run, item["path"])
        if file_hash(group_path) != item["sha256"]:
            raise ValueError("Group hash invalid")
        group = json.loads(group_path.read_text())
        if group["id"] in seen or group["id"] not in expected:
            raise ValueError("Group duplicate or unexpected")
        seen.add(group["id"])
        if any(group[k] != v for k, v in expected[group["id"]].items()):
            raise ValueError("Group differs from pre-registered context")
        if set(group["trials"]) != set(manifest["protocol"]["interventions"]):
            raise ValueError("Missing intervention branch")
        results = {}
        for branch, info in group["trials"].items():
            path = checked(run, info["path"])
            if file_hash(path) != info["sha256"]:
                raise ValueError("Trial hash invalid")
            r = json.loads(path.read_text())
            if not r["valid"] or not r["artifact_complete"] or r["trial_id"] in trial_ids:
                raise ValueError("Invalid or duplicate physical trial")
            trial_ids.add(r["trial_id"])
            trials_count += 1
            if (file_hash(path.parent / "scene.xml") != r["scene_sha256"]
                    or file_hash(path.parent / "policy.py") != r["source_sha256"]
                    or digest(json.loads((path.parent / "model.json").read_text())) != r["model_sha256"]):
                raise ValueError("Executed scene/model/source artifact mismatch")
            for filename in ("events.json", "tcp_joint_trajectory.json", "contact_summary.json"):
                value = json.loads((path.parent / filename).read_text())
                if not isinstance(value, list):
                    raise ValueError("Incomplete side artifact")
            expected_check = transition_check(r["model"], r["before"]["state"], r["after"]["state"], r["exception"])
            if expected_check != r["check"] or r["contract_consistent"] != expected_check["effect_ok"]:
                raise ValueError("Contract result does not match observations")
            if r["joint_success"] != (r["physical_pick_success"] and r["contract_consistent"]):
                raise ValueError("Physical and contract outcomes conflated")
            if r["physical_metrics"]["weld_violation"]:
                raise ValueError("Weld used in strict experiment")
            if json.loads((path.parent / "frames.json").read_text()) != r["frames"]:
                raise ValueError("Phase frame list differs from result")
            last_time = -1.
            for frame in r["frames"]:
                obs = frame["observation"]
                if not obs.get("derived_state_refreshed") or obs["sim_time"] < last_time:
                    raise ValueError("Unsynchronized or unordered observation")
                last_time = obs["sim_time"]
                if [m["camera"] for m in frame["images"]] != manifest["protocol"]["cameras"]:
                    raise ValueError("Missing camera")
                for media in frame["images"]:
                    image_path = checked(path.parent, media["path"])
                    if media["sim_time"] != obs["sim_time"] or file_hash(image_path) != media["sha256"]:
                        raise ValueError("Image identity/time mismatch")
                    with Image.open(image_path) as im:
                        values = np.asarray(im)
                        if values.max() == values.min():
                            raise ValueError("Invalid rendered image")
                        frame_hashes.add(hashlib.sha256(values.tobytes()).hexdigest())
                    frames_count += 1
            results[branch] = r
            key = (group["scene"], branch)
            outcomes[key].update(total=1, physical=int(r["physical_pick_success"]), contract=int(r["contract_consistent"]), joint=int(r["joint_success"]))
        if len({r["scene_sha256"] for r in results.values()}) != 1 or len({r["initial_qpos_sha256"] for r in results.values()}) != 1:
            raise ValueError("Paired initial scenes differ")
        for left, right, field in (("none", "model_only", "source_sha256"), ("none", "policy_only", "model_sha256"),
                                   ("model_only", "cross_level", "model_sha256"), ("policy_only", "cross_level", "source_sha256")):
            if results[left][field] != results[right][field]:
                raise ValueError("Non-target intervention layer changed")
    if seen != set(expected):
        raise ValueError("Missing expected groups")
    data_audit = audit_dataset(dataset)
    processor_path = dataset / "processor_audit.json"
    processor = json.loads(processor_path.read_text()) if processor_path.exists() else None
    if processor is not None and len(processor["examples"]) != data_audit["samples"]:
        raise ValueError("Processor did not inspect every sample")
    return {"raw_evidence_passed": True, "groups": len(seen), "contexts": len({g["context_id"] for g in expected.values()}),
            "physical_rollouts": trials_count, "raw_images": frames_count, "unique_raw_pixel_images": len(frame_hashes),
            "outcomes": {scene: {branch: dict(outcomes[(scene, branch)]) for branch in manifest["protocol"]["interventions"]}
                         for scene in ("cover", "blocks")},
            "dataset_audit": data_audit, "processor_passed": processor["passed"] if processor else None,
            "processor_examples": len(processor["examples"]) if processor else 0,
            "max_sequence_tokens": max((r["sequence_tokens"] for r in processor["examples"]), default=0) if processor else None,
            "training_started": False, "production_training_approved": False,
            "reasons_to_defer_training": ["only four scene-seed contexts and one program family", "diagnosis recommendations are handwritten heuristics",
                                         "no unseen-scene test or independent repair validation", "reference assistant mask not wired into actual trainer"],
            "roots": {"collection_freeze": file_hash(run / "collection_frozen.json"), "dataset_manifest": file_hash(dataset / "dataset_manifest.json")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new audit output")
    result = audit(args.run, args.dataset)
    save(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
