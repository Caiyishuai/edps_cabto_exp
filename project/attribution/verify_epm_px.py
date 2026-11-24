"""CABTO-EPM v2 验证：手动自回归（训练一致 prompt），从 RGB 俯视图推理像素坐标。

用 apply_chat_template(add_generation_prompt=True) + prepare_inputs 构造与训练一致的 prompt，
手动 argmax 贪心自回归（绕开 mlx_vlm.generate() 对 Qwen2.5-VL 的 prompt 起点错位 bug），
计算像素误差 + 用 intersect_plane 反投影的 3D 定位误差。
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

import mlx_vlm.models.rope_utils as rope_utils
rope_utils._HAS_METAL = False

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.single_arm_rebuilt import RebuiltSingleArm  # noqa: E402
from core.cover_cabto_api import intersect_plane  # noqa: E402

MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-bf16"
ADAPTER = "/Users/yishuaicai/mywork/edps_cabto_exp/outputs/epm_px_data/adapters"
SAMPLES = "/Users/yishuaicai/mywork/edps_cabto_exp/outputs/epm_px_data/samples.json"
VAL_JSONL = "/Users/yishuaicai/mywork/edps_cabto_exp/outputs/epm_px_data/val.jsonl"
SCENE = "/Users/yishuaicai/mywork/edps_cabto_exp/project/scenes/cover_kitchen_sort_v2.json"
NUM = 20

PAT = re.compile(r"(\w+)\s*[:：]\s*\(?\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)?")


def parse_coords(text):
    out = {}
    for line in text.splitlines():
        m = PAT.search(line)
        if m:
            out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return out


def manual_generate(model, processor, config, question, img_path, max_tokens=60):
    image_token_index = config["image_token_id"]
    eos = config["eos_token_id"]
    tok = processor.tokenizer
    messages = [{"role": "user", "content": question}]
    prompt = apply_chat_template(processor, config, messages, add_generation_prompt=True, num_images=1)
    inputs = prepare_inputs(
        processor, images=[img_path], prompts=[prompt],
        image_token_index=image_token_index, resize_shape=None,
    )
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    attention_mask = inputs["attention_mask"]
    data_kwargs = {k: v for k, v in inputs.items() if k not in ["input_ids", "pixel_values", "attention_mask"]}

    generated = []
    for _ in range(max_tokens):
        outputs = model(input_ids, pixel_values, attention_mask, **data_kwargs)
        next_token = int(mx.argmax(outputs.logits[0, -1]))
        if next_token == eos:
            break
        generated.append(next_token)
        input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)
        attention_mask = mx.concatenate([attention_mask, mx.ones((1, 1))], axis=1)
    return tok.decode(generated, skip_special_tokens=False)


def main():
    print("loading model + adapter ...", flush=True)
    model, processor = load(MODEL, adapter_path=ADAPTER, processor_config={"trust_remote_code": True})
    config = model.config.__dict__
    for _, m in model.named_modules():
        if hasattr(m, "fused_apply"):
            m.fused_apply = False
            m._compiled_apply = {}

    config_json = json.loads(Path(SCENE).read_text())
    runtime = RebuiltSingleArm(config_json, seed=0, render=True)
    env = runtime.env

    samples_data = json.loads(Path(SAMPLES).read_text())
    by_image = {Path(s["image"]).name: s for s in samples_data["samples"]}

    rows = [json.loads(l) for l in Path(VAL_JSONL).read_text().splitlines()]
    rows = rows[:NUM]

    px_errs, d3_errs = [], []
    size = 720
    for i, row in enumerate(rows):
        img = row["images"][0]
        img_rel = Path(img).name
        truth_px = parse_coords(row["answer"])
        text = manual_generate(model, processor, config, row["question"], img)
        pred_px = parse_coords(text)

        s3d = by_image.get(img_rel) or {}
        for name, t in truth_px.items():
            if name not in pred_px:
                print(f"[{i}] {name}: 真值 {t}  预测缺失", flush=True)
                continue
            p = pred_px[name]
            pe = float(np.linalg.norm(np.array(p) - np.array(t)))
            px_errs.append(pe)
            line = f"[{i}] {name}: 真值 {t} 预测 {p} 像素误差 {pe:.1f}px"
            if name in s3d.get("objects_3d", {}):
                z = s3d["objects_3d"][name][2]
                pred3d = intersect_plane(env, np.array(p), "overview", size, z)
                truth3d = s3d["objects_3d"][name]
                de = float(np.linalg.norm(pred3d[:2] - np.array(truth3d[:2])))
                d3_errs.append(de)
                line += f" -> 3D {de*100:.2f}cm"
            print(line, flush=True)

    print("\n" + "=" * 50, flush=True)
    if px_errs:
        print(f"平均像素误差: {np.mean(px_errs):.1f} px  (中位 {np.median(px_errs):.1f} px, 共 {len(px_errs)} 个)", flush=True)
    if d3_errs:
        print(f"平均 3D 定位误差: {np.mean(d3_errs)*100:.2f} cm  (中位 {np.median(d3_errs)*100:.2f} cm)", flush=True)

    runtime.close()


if __name__ == "__main__":
    main()
