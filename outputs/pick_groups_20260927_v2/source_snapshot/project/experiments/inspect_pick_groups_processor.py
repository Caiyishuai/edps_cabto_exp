"""Offline processor-only inspection of image expansion and assistant supervision."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from blocks_p0 import save
from collect_pick_groups import file_hash
from export_pick_groups import audit_dataset, checked


def prepare_supervision(record, dataset, processor):
    images = [Image.open(checked(dataset, path)).convert("RGB") for path in record["images"]]
    try:
        messages = []
        for turn in record["conversations"]:
            role = {"system": "system", "human": "user", "gpt": "assistant"}[turn["from"]]
            if role == "user":
                content = []
                parts = turn["value"].split("<image>")
                for i, part in enumerate(parts):
                    if part:
                        content.append({"type": "text", "text": part})
                    if i < len(parts) - 1:
                        content.append({"type": "image"})
            else:
                content = [{"type": "text", "text": turn["value"]}]
            messages.append({"role": role, "content": content})
        if sum(t["type"] == "image" for m in messages for t in m["content"]) != len(images):
            raise ValueError("Conversation/media alignment invalid")
        prompt = processor.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoded = processor(text=[full_text], images=images, return_tensors="np", padding=False)
        prefix = processor(text=[prompt], images=images, return_tensors="np", padding=False)
        tokens, prefix_tokens = np.asarray(encoded["input_ids"])[0], np.asarray(prefix["input_ids"])[0]
        n = len(prefix_tokens)
        if not np.array_equal(tokens[:n], prefix_tokens) or len(tokens) <= n:
            raise ValueError("Template is not a strict prefix; cannot define assistant mask")
        labels = tokens.copy()
        labels[:n] = -100
        mask = np.asarray(encoded.get("attention_mask", np.ones_like(encoded["input_ids"])))
        labels[mask[0] == 0] = -100
        tokenizer = processor.tokenizer
        for marker in ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>"):
            identifier = tokenizer.convert_tokens_to_ids(marker)
            if identifier is not None and np.any(labels == identifier):
                raise ValueError("Visual marker receives assistant supervision")
        grid = np.asarray(encoded["image_grid_thw"])
        if grid.shape[0] != len(images):
            raise ValueError("Processor lost images")
        decoded = tokenizer.decode(tokens[n:], skip_special_tokens=True).strip()
        expected = record["conversations"][-1]["value"].strip()
        if decoded != expected:
            raise ValueError("Supervised token span does not match the target answer")
        return {"image_count": len(images), "sequence_tokens": len(tokens), "prompt_tokens": n,
                "target_tokens": int(np.sum(labels != -100)), "image_grid_thw": grid.tolist(),
                "assistant_only_mask": True, "prompt_and_visual_tokens_ignored": True,
                "target_decodes_exactly": True, "truncated": False}
    finally:
        for image in images:
            image.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", default="mlx-community/Qwen2.5-VL-3B-Instruct-bf16")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new processor audit path")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor
    audit_dataset(args.dataset)
    model = Path(snapshot_download(args.model, local_files_only=True))
    processor = AutoProcessor.from_pretrained(str(model), local_files_only=True,
                                             min_pixels=128 * 28 * 28, max_pixels=256 * 28 * 28)
    metadata = json.loads((args.dataset / "sample_index.json").read_text())
    rows = {split: [json.loads(line) for line in (args.dataset / f"{split}.jsonl").read_text().splitlines()]
            for split in {m["split"] for m in metadata}}
    selected = metadata
    result = {"model_id": args.model, "snapshot": str(model), "revision": model.name,
              "weights_loaded": False, "training_started": False, "examples": [],
              "processor_files": {p.name: file_hash(p) for p in model.iterdir() if p.is_file() and p.suffix in (".json", ".jinja")},
              "scope": "processor and reference mask validation; existing MLX trainer collator is NOT certified"}
    for m in selected:
        report = prepare_supervision(rows[m["split"]][m["line"]], args.dataset, processor)
        result["examples"].append({"sample_id": m["sample_id"], "split": m["split"], "kind": m["kind"], **report})
        print(m["split"], m["kind"], report, flush=True)
    result["passed"] = bool(selected and all(r["sequence_tokens"] <= args.max_tokens for r in result["examples"]))
    result["max_tokens"] = args.max_tokens
    save(args.output, result)
    if not result["passed"]:
        raise SystemExit("Sequence length budget exceeded; no silent truncation")


if __name__ == "__main__":
    main()
