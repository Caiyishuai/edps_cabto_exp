"""Bounded multi-image LoRA training acceptance; not a generalization experiment."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

from blocks_p0 import ROOT, digest, save
from collect_pick_groups import file_hash
from export_pick_groups import audit_dataset, checked
from inspect_pick_groups_processor import prepare_supervision

MODEL_ID = "mlx-community/Qwen2.5-VL-3B-Instruct-bf16"
REVISION = "33765717028a7f0514ea1d213035cddf4b32c500"


def select_samples(dataset):
    index = json.loads((dataset / "sample_index.json").read_text())
    groups = json.loads((dataset / "group_audit.json").read_text())
    rows = {split: [json.loads(x) for x in (dataset / f"{split}.jsonl").read_text().splitlines()]
            for split in ("train", "validation")}
    selected = {"train": [], "validation": []}
    for split in selected:
        for scene, obj in (("cover", "apple"), ("blocks", "yellow")):
            context = [g for g in groups if g["scene"] == scene and g["object"] == obj and g["split"] == split]
            if split == "validation":
                context = [g for g in context if g["condition_private"] == "combined"]
            for group in context:
                entries = [m for m in index if m["group_id"] == group["group_id"]]
                wanted = [next(m for m in entries if m["kind"] == "diagnosis")]
                if group["condition_private"] == "combined":
                    if split == "train":
                        wanted.append(next(m for m in entries if m["kind"] == "forecast"))
                    wanted.append(next(m for m in entries if m["kind"] == "update"))
                for meta in wanted:
                    record = rows[split][meta["line"]]
                    if digest(record) != meta["record_sha256"]:
                        raise ValueError("Selected record differs from audited dataset")
                    selected[split].append({"metadata": meta, "record": record, "scene": scene,
                                            "selection_only_condition": group["condition_private"]})
    if len(selected["train"]) != 12 or len(selected["validation"]) != 4:
        raise ValueError("Acceptance slice must contain 12 train and 4 validation examples")
    if {x["metadata"]["context_id"] for x in selected["train"]} & {x["metadata"]["context_id"] for x in selected["validation"]}:
        raise ValueError("Training and validation contexts overlap")
    return selected


def source_hashes():
    paths = [Path(__file__), Path(__file__).with_name("inspect_pick_groups_processor.py"),
             Path(__file__).with_name("export_pick_groups.py"), Path(__file__).with_name("collect_pick_groups.py"),
             Path(__file__).with_name("blocks_p0.py")]
    return {str(p.resolve().relative_to(ROOT)): file_hash(p) for p in paths}


def load_manifest(output):
    manifest = json.loads((output / "manifest.json").read_text())
    for name, sha in manifest["source_hashes"].items():
        if file_hash(ROOT / name) != sha:
            raise ValueError(f"Acceptance source changed: {name}")
    dataset = Path(manifest["dataset"])
    for name, sha in manifest["dataset_files"].items():
        if file_hash(dataset / name) != sha:
            raise ValueError("Audited dataset changed")
    for entry in manifest["samples"]:
        if file_hash(checked(output, entry["batch_file"])) != entry["batch_sha256"]:
            raise ValueError("Prepared training batch changed")
    return manifest


def prepare(args):
    if args.output.exists():
        raise ValueError("Use a new acceptance output directory")
    dataset = args.dataset.resolve()
    audit_dataset(dataset)
    prior = json.loads((dataset / "processor_audit.json").read_text())
    if prior["passed"] is not True:
        raise ValueError("Dataset has not passed processor acceptance")
    selected = select_samples(dataset)
    args.output.mkdir(parents=True)
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor
    snapshot = Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
    processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True,
                                             min_pixels=128 * 28 * 28, max_pixels=256 * 28 * 28)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "dataset": str(dataset),
                "model_id": MODEL_ID, "revision": REVISION, "snapshot": str(snapshot),
                "dataset_files": {name: file_hash(dataset / name) for name in
                                  ("dataset_manifest.json", "train.jsonl", "validation.jsonl", "sample_index.json", "processor_audit.json")},
                "source_hashes": source_hashes(), "samples": [],
                "settings": {"rank": 8, "alpha": 16, "dropout": 0.0, "targets": ["q_proj", "v_proj"],
                             "learning_rate": 1e-5, "steps": args.steps, "max_tokens": 8192,
                             "seed": 2801, "gradient_clip_norm": 1.0, "gradient_checkpoint": True,
                             "memory_guideline_gib": 14, "cache_limit_gib": 0.5},
                "versions": {p: importlib.metadata.version(p) for p in ("mlx", "mlx-vlm", "transformers", "numpy")},
                "scope": "12-example short LoRA acceptance; validation is same-family small jitter, NOT OOD"}
    for name, sha in manifest["source_hashes"].items():
        p = args.output / "source_snapshot" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ROOT / name).read_bytes())
    for split, records in selected.items():
        for i, entry in enumerate(records):
            processed = prepare_supervision(entry["record"], dataset, processor, return_tensors=True)
            if processed["audit"]["sequence_tokens"] > manifest["settings"]["max_tokens"]:
                raise ValueError("No silent truncation of training data")
            folder = args.output / "batches"
            folder.mkdir(exist_ok=True)
            path = folder / f"{split}_{i:02d}.npz"
            np.savez_compressed(path, **processed["batch"])
            record = {"sample_id": entry["metadata"]["sample_id"], "split": split,
                      "group_id": entry["metadata"]["group_id"], "context_id": entry["metadata"]["context_id"],
                      "kind": entry["metadata"]["kind"], "scene": entry["scene"],
                      "batch_file": str(path.relative_to(args.output)), "batch_sha256": file_hash(path),
                      "audit": processed["audit"], "prompt": processed["prompt"], "target": processed["target"],
                      "images": entry["record"]["images"], "image_sha256": [file_hash(dataset / p) for p in entry["record"]["images"]]}
            manifest["samples"].append(record)
            print("PREPARE", split, i, record["kind"], processed["audit"]["sequence_tokens"], flush=True)
    save(args.output / "manifest.json", manifest)


def model_setup(manifest, adapter=None):
    import mlx.core as mx
    import mlx_vlm.models.rope_utils as rope
    from mlx_vlm import load
    from mlx_vlm.trainer.utils import get_peft_model, grad_checkpoint
    mx.set_memory_limit(int(manifest["settings"]["memory_guideline_gib"] * 1024**3))
    mx.set_cache_limit(int(manifest["settings"]["cache_limit_gib"] * 1024**3))
    mx.random.seed(manifest["settings"]["seed"])
    rope._HAS_METAL = False
    started = time.monotonic()
    model, _ = load(manifest["snapshot"], adapter_path=str(adapter) if adapter else None,
                    processor_config={"local_files_only": True})
    if adapter is None:
        settings = manifest["settings"]
        model = get_peft_model(model, settings["targets"], rank=settings["rank"], alpha=settings["alpha"],
                               dropout=settings["dropout"], verbose=False)
    count = 0
    for _, module in model.named_modules():
        if hasattr(module, "fused_apply"):
            module.fused_apply = False
            module._compiled_apply = {}
            count += 1
    if adapter is None and manifest["settings"]["gradient_checkpoint"]:
        grad_checkpoint(model.layers[0])
    mx.eval(model.parameters())
    return model, {"load_seconds": time.monotonic() - started, "unfused_rope_modules": count,
                   "memory_limit_is_guideline_not_hard_cap": True}


def trainable_map(model):
    from mlx.utils import tree_flatten
    return dict(tree_flatten(model.trainable_parameters()))


def array_digest(values):
    import mlx.core as mx
    result = {}
    for name, value in values.items():
        mx.eval(value)
        data = np.asarray(value.astype(mx.float32))
        result[name] = hashlib.sha256(data.tobytes()).hexdigest()
    return result


def batch_from_file(output, entry):
    import mlx.core as mx
    with np.load(checked(output, entry["batch_file"]), allow_pickle=False) as data:
        values = {k: mx.array(data[k]) for k in data.files}
    mask = np.asarray(values["attention_mask"])
    if values["input_ids"].shape[0] != 1 or not np.all(mask == 1):
        raise ValueError("This acceptance collator requires unpadded batch size one")
    labels = np.asarray(values.pop("labels"))
    valid = np.flatnonzero(labels[0] != -100)
    if not len(valid) or valid[0] < 1 or not np.array_equal(valid, np.arange(valid[0], labels.shape[1])):
        raise ValueError("Target tokens must be one complete assistant suffix")
    if len(valid) != entry["audit"]["target_tokens"] or valid[0] != entry["audit"]["prompt_tokens"]:
        raise ValueError("Batch labels differ from processor acceptance")
    values["labels"] = mx.array(labels)
    return values, int(valid[0])


def masked_ce(logits, labels):
    import mlx.core as mx
    import mlx.nn as nn
    valid = labels != -100
    safe = mx.where(valid, labels, mx.zeros_like(labels))
    ce = nn.losses.cross_entropy(logits.astype(mx.float32), safe, reduction="none")
    return mx.sum(mx.where(valid, ce, mx.zeros_like(ce))) / mx.maximum(mx.sum(valid), 1)


def supervised_logits(model, batch, target_start, *, cached=None):
    import mlx.core as mx
    kwargs = {"image_grid_thw": batch["image_grid_thw"]}
    if cached is not None:
        kwargs["cached_image_features"] = cached
    out = model(batch["input_ids"][:, :-1], batch["pixel_values"], mask=None, cache=None, **kwargs)
    logits = out.logits
    if logits.shape[1] != batch["input_ids"].shape[1] - 1:
        raise ValueError("Unexpected logits/token alignment; refuse padding workaround")
    return logits[:, target_start - 1:, :]


def loss_fn(model, batch, target_start, *, cached=None):
    logits = supervised_logits(model, batch, target_start, cached=cached)
    return masked_ce(logits, batch["labels"][:, target_start:])


def vision_features(model, batch):
    import mlx.core as mx
    dtype = model.vision_tower.patch_embed.proj.weight.dtype
    features = model.vision_tower(batch["pixel_values"].astype(dtype), batch["image_grid_thw"], output_hidden_states=False)
    features = mx.stop_gradient(features)
    mx.eval(features)
    return features


def evaluate(model, output, entries, stage):
    import mlx.core as mx
    model.eval()
    result = []
    for entry in entries:
        batch, target_start = batch_from_file(output, entry)
        loss = loss_fn(model, batch, target_start)
        mx.eval(loss)
        result.append({"sample_id": entry["sample_id"], "kind": entry["kind"], "loss": float(loss.item()),
                       "target_tokens": entry["audit"]["target_tokens"]})
        print("EVAL", stage, entry["kind"], result[-1]["loss"], flush=True)
        del loss, batch
        mx.clear_cache()
    return {"samples": result, "mean_loss": float(np.mean([r["loss"] for r in result])),
            "token_weighted_loss": sum(r["loss"] * r["target_tokens"] for r in result) / sum(r["target_tokens"] for r in result)}


def run_train(args):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map
    from mlx_vlm.trainer.utils import save_adapter
    manifest = load_manifest(args.output)
    result_path = args.output / "training_result.json"
    if result_path.exists():
        raise ValueError("Training result exists; do not overwrite")
    result = {"status": "started", "passed": False, "steps": [], "manifest_sha256": file_hash(args.output / "manifest.json"),
              "scope": "training mechanics only; not generalization or autonomous attribution accuracy"}
    save(result_path, result)
    start = time.monotonic()
    try:
        model, info = model_setup(manifest)
        result["model_setup"] = info
        train = [r for r in manifest["samples"] if r["split"] == "train"]
        val = [r for r in manifest["samples"] if r["split"] == "validation"]
        parameters = trainable_map(model)
        if not parameters or any(not n.startswith("language_model.") or not n.endswith(("lora_a", "lora_b")) for n in parameters):
            raise ValueError("Unexpected trainable base or vision parameters")
        if any(not any(f".{target}." in n for target in manifest["settings"]["targets"]) for n in parameters):
            raise ValueError("Unexpected adapter target")
        initial_hashes = array_digest(parameters)
        result["trainable"] = {"parameters": sum(v.size for v in parameters.values()),
                               "tensors": {k: list(v.shape) for k, v in parameters.items()},
                               "vision_frozen": True, "base_frozen": True}
        all_params = dict(tree_flatten(model.parameters()))
        frozen_refs = {k: all_params[k] for k in ("vision_tower.patch_embed.proj.weight", "language_model.model.embed_tokens.weight")}
        base_hashes = array_digest(frozen_refs)
        result["before_train"] = evaluate(model, args.output, train, "before/train")
        result["before_validation"] = evaluate(model, args.output, val, "before/validation")
        save(result_path, result)
        batch, target_start = batch_from_file(args.output, train[0])
        features = vision_features(model, batch)
        direct = float(loss_fn(model, batch, target_start).item())
        mx.clear_cache()
        cached_loss = float(loss_fn(model, batch, target_start, cached=features).item())
        mx.clear_cache()
        no_image = float(loss_fn(model, batch, target_start, cached=mx.zeros_like(features)).item())
        result["image_path_check"] = {"uncached_loss": direct, "cached_loss": cached_loss,
                                      "zero_feature_loss": no_image,
                                      "cache_absolute_difference": abs(direct - cached_loss),
                                      "zero_feature_absolute_difference": abs(direct - no_image),
                                      "interpretation": "feature-level dependency only, NOT evidence of visual diagnostic superiority"}
        if result["image_path_check"]["cache_absolute_difference"] > 1e-4:
            raise ValueError("Cached frozen-vision forward differs from direct multi-image forward")
        del batch, features, direct, cached_loss, no_image
        mx.clear_cache()
        optimizer = optim.Adam(learning_rate=manifest["settings"]["learning_rate"])
        train_loss_grad = nn.value_and_grad(model, lambda m, b, n, f: loss_fn(m, b, n, cached=f))
        model.train()
        for step in range(manifest["settings"]["steps"]):
            entry = train[step % len(train)]
            batch, target_start = batch_from_file(args.output, entry)
            features = vision_features(model, batch)
            began = time.monotonic()
            loss, grads = train_loss_grad(model, batch, target_start, features)
            flat = dict(tree_flatten(grads))
            norm = mx.sqrt(sum(mx.sum(g.astype(mx.float32) ** 2) for g in flat.values()))
            mx.eval(loss, norm, grads)
            loss_value, norm_value = float(loss.item()), float(norm.item())
            if not np.isfinite(loss_value) or not np.isfinite(norm_value) or norm_value <= 0:
                raise ValueError("Nonfinite loss or zero/nonfinite adapter gradient")
            scale = min(1., manifest["settings"]["gradient_clip_norm"] / max(norm_value, 1e-12))
            grads = tree_map(lambda g: g * scale, grads)
            optimizer.update(model, grads)
            mx.eval(model.trainable_parameters(), optimizer.state)
            row = {"step": step + 1, "sample_id": entry["sample_id"], "kind": entry["kind"], "loss": loss_value,
                   "gradient_norm_before_clip": norm_value, "clip_scale": scale,
                   "target_tokens": entry["audit"]["target_tokens"], "image_count": entry["audit"]["image_count"],
                   "wall_seconds": time.monotonic() - began, "peak_memory_gib": mx.get_peak_memory() / 1024**3}
            result["steps"].append(row)
            save(result_path, result)
            print("TRAIN", json.dumps(row), flush=True)
            del loss, grads, flat, norm, features, batch
            mx.clear_cache()
        model.eval()
        final_params = trainable_map(model)
        final_hashes = array_digest(final_params)
        changed = [name for name in final_hashes if final_hashes[name] != initial_hashes[name]]
        result["changed_adapter_tensors"] = len(changed)
        current_weights = dict(tree_flatten(model.parameters()))
        result["frozen_reference_weights_unchanged"] = array_digest({k: current_weights[k] for k in base_hashes}) == base_hashes
        if not changed or not result["frozen_reference_weights_unchanged"]:
            raise ValueError("Adapter failed to update or checked base weights changed")
        result["after_train"] = evaluate(model, args.output, train, "after/train")
        result["after_validation"] = evaluate(model, args.output, val, "after/validation")
        adapter_dir = args.output / "adapter"
        save_adapter(model, adapter_dir / "adapters.safetensors")
        saved = mx.load(str(adapter_dir / "adapters.safetensors"))
        result["saved_adapter_matches_memory"] = array_digest(saved) == final_hashes
        save(args.output / "adapter_hashes.json", final_hashes)
        batch, target_start = batch_from_file(args.output, train[0])
        reference = supervised_logits(model, batch, target_start).astype(mx.float32)
        mx.eval(reference)
        np.save(args.output / "reload_reference_logits.npy", np.asarray(reference[:, :2, :]))
        result["reload_reference_sample"] = train[0]["sample_id"]
        result["adapter_files"] = {p.name: file_hash(p) for p in adapter_dir.iterdir() if p.is_file()}
        result["training_loss_decreased"] = result["after_train"]["mean_loss"] < result["before_train"]["mean_loss"]
        result["passed"] = bool(result["saved_adapter_matches_memory"] and len(result["steps"]) == manifest["settings"]["steps"])
        result["status"] = "trained_pending_fresh_process_reload"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["total_wall_seconds"] = time.monotonic() - start
        result["peak_memory_gib"] = mx.get_peak_memory() / 1024**3
        save(result_path, result)


def verify(args):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    manifest = load_manifest(args.output)
    trained = json.loads((args.output / "training_result.json").read_text())
    if not trained["passed"] or trained["status"] != "trained_pending_fresh_process_reload":
        raise ValueError("Training did not pass mechanical acceptance")
    path = args.output / "reload_result.json"
    if path.exists():
        raise ValueError("Reload report exists")
    result = {"passed": False, "status": "started", "fresh_process": True}
    save(path, result)
    try:
        adapter = args.output / "adapter"
        for name, sha in trained["adapter_files"].items():
            if file_hash(adapter / name) != sha:
                raise ValueError("Saved adapter changed")
        model, info = model_setup(manifest, adapter)
        model.eval()
        expected = json.loads((args.output / "adapter_hashes.json").read_text())
        params = {name: value for name, value in tree_flatten(model.parameters()) if name in expected}
        result["adapter_hashes_match"] = array_digest(params) == expected
        entry = next(e for e in manifest["samples"] if e["sample_id"] == trained["reload_reference_sample"])
        batch, target_start = batch_from_file(args.output, entry)
        logits = supervised_logits(model, batch, target_start).astype(mx.float32)
        loss = loss_fn(model, batch, target_start)
        mx.eval(logits, loss)
        saved = np.load(args.output / "reload_reference_logits.npy", allow_pickle=False)
        actual = np.asarray(logits[:, :2, :])
        result["max_logit_difference"] = float(np.max(np.abs(actual - saved)))
        result["reloaded_loss"] = float(loss.item())
        reference_loss = next(r["loss"] for r in trained["after_train"]["samples"] if r["sample_id"] == entry["sample_id"])
        result["loss_difference"] = abs(result["reloaded_loss"] - reference_loss)
        result["setup"] = info
        result["passed"] = bool(result["adapter_hashes_match"] and result["max_logit_difference"] < 1e-4 and result["loss_difference"] < 1e-4)
        result["status"] = "complete"
        result["scope"] = "adapter persistence and fresh six-image teacher-forced inference; not generation accuracy or OOD"
        if not result["passed"]:
            raise ValueError("Fresh-process adapter reload did not reproduce logits/loss")
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save(path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "train", "verify"))
    parser.add_argument("--dataset", type=Path, default=ROOT / "outputs/pick_groups_sft_20260927_v3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=range(1, 25), default=12)
    parser.add_argument("--max-seconds", type=int, default=1800)
    args = parser.parse_args()
    args.output = args.output.resolve()
    def timeout(signum, frame):
        raise TimeoutError("Acceptance phase time budget exhausted")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.max_seconds)
    try:
        {"prepare": prepare, "train": run_train, "verify": verify}[args.phase](args)
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
