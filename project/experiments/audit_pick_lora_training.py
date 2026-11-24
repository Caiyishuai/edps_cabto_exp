"""Fresh-process audit for canonical direct multi-image LoRA training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

from blocks_p0 import save
from collect_pick_groups import file_hash
from train_pick_groups_lora import (array_digest, batch_from_file, load_manifest, loss_fn,
                                   model_setup, supervised_logits)


def audit(args):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_vlm import generate
    from mlx_vlm.utils import load_processor
    manifest = load_manifest(args.run)
    training = json.loads((args.run / "training_result.json").read_text())
    reload = json.loads((args.run / "reload_result.json").read_text())
    if not training["passed"] or not reload["passed"]:
        raise ValueError("Prior train/reload phase did not pass")
    if manifest["settings"].get("vision_training_path") != "direct_model_forward_no_feature_cache":
        raise ValueError("Cached-vision training is not certified by this audit")
    result = {"passed": False, "status": "started", "pid": os.getpid(), "ppid": os.getppid(),
              "created_at_unix": time.time(), "training_result_sha256": file_hash(args.run / "training_result.json"),
              "reload_result_sha256": file_hash(args.run / "reload_result.json"),
              "scope": "direct-model training audit; no optimizer steps; not diagnostic accuracy",
              "cache_policy": "feature-cache optimization rejected after prior failed gradient-equivalence check"}
    save(args.output, result)
    try:
        losses = [r["loss"] for key in ("before_train", "before_validation", "after_train", "after_validation")
                  for r in training[key]["samples"]]
        if not all(np.isfinite(losses)) or not all(np.isfinite(r["gradient_norm_before_clip"]) and r["gradient_norm_before_clip"] > 0 for r in training["steps"]):
            raise ValueError("Nonfinite metrics or invalid training gradients")
        saved_weights = mx.load(str(args.run / "adapter/adapters.safetensors"))
        if not all(bool(mx.all(mx.isfinite(x)).item()) for x in saved_weights.values()):
            raise ValueError("Nonfinite trained adapter")
        result["all_saved_adapter_values_finite"] = True
        result["all_losses_and_gradient_norms_finite"] = True
        result["trainable_target_coverage"] = {target: sum(f".{target}." in name for name in saved_weights)
                                               for target in ("q_proj", "v_proj")}
        if not all(result["trainable_target_coverage"].values()):
            raise ValueError("Missing requested adapter family")
        model, setup = model_setup(manifest, args.run / "adapter")
        model.eval()
        flat = dict(tree_flatten(model.parameters()))
        loaded = {k: flat[k] for k in saved_weights}
        result["fresh_loaded_adapter_matches"] = array_digest(loaded) == array_digest(saved_weights)
        if not result["fresh_loaded_adapter_matches"]:
            raise ValueError("Loaded adapter differs")
        entry = next(e for e in manifest["samples"] if e["sample_id"] == training["reload_reference_sample"])
        batch, n = batch_from_file(args.run, entry)
        direct = supervised_logits(model, batch, n)
        mx.eval(direct)
        reference = np.asarray(direct[:, :2].astype(mx.float32))
        del direct
        mx.clear_cache()
        changed = dict(batch)
        ids = np.asarray(batch["input_ids"]).copy()
        ids[0, -3] = 100 if ids[0, -3] != 100 else 101
        changed["input_ids"] = mx.array(ids)
        causal = supervised_logits(model, changed, n)
        mx.eval(causal)
        result["future_token_change_earlier_logits_max_difference"] = float(np.max(np.abs(np.asarray(causal[:, :2].astype(mx.float32)) - reference)))
        del causal, changed
        baseline_loss = float(loss_fn(model, batch, n).item())
        zero_batch = {**batch, "pixel_values": mx.zeros_like(batch["pixel_values"])}
        zero_loss = float(loss_fn(model, zero_batch, n).item())
        result["zero_image_pixels_loss_delta"] = abs(zero_loss - baseline_loss)
        result["image_check_scope"] = "real pixel-path numerical dependence, not proof of diagnostic superiority"
        result["mechanics_passed"] = bool(result["future_token_change_earlier_logits_max_difference"] <= 1e-4
                                          and result["zero_image_pixels_loss_delta"] > 1e-6)
        save(args.output, result)
        del batch, zero_batch, flat, loaded, saved_weights
        mx.clear_cache()
        processor = load_processor(Path(manifest["snapshot"]), add_detokenizer=True,
                                   eos_token_ids=model.config.eos_token_id, local_files_only=True,
                                   min_pixels=128 * 28 * 28, max_pixels=256 * 28 * 28)
        before = time.monotonic()
        try:
            response = generate(model, processor, entry["prompt"],
                                image=[str(Path(manifest["dataset"]) / p) for p in entry["images"]],
                                max_tokens=192, temperature=0., verbose=False)
            result["generation_smoke"] = {"raw": response.text, "tokens": int(response.generation_tokens),
                                           "finish_reason": getattr(response, "finish_reason", None),
                                           "image_count": len(entry["images"]), "wall_seconds": time.monotonic() - before}
            try:
                json.loads(response.text)
                result["generation_smoke"]["valid_json"] = True
            except (ValueError, TypeError):
                result["generation_smoke"]["valid_json"] = False
        except Exception as exc:
            result["generation_smoke"] = {"error": f"{type(exc).__name__}: {exc}", "wall_seconds": time.monotonic() - before}
        result["passed"] = bool(result["mechanics_passed"] and "raw" in result["generation_smoke"])
        result["status"] = "complete"
        result["peak_memory_gib"] = mx.get_peak_memory() / 1024**3
        if not result["passed"]:
            raise ValueError("Direct-model audit or generation invocation failed")
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save(args.output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.run = args.run.resolve()
    if args.output.exists():
        parser.error("Use a new audit output")
    def timeout(signum, frame):
        raise TimeoutError("Real-model audit time budget exhausted")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(600)
    audit(args)
