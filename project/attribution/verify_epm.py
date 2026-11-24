"""CABTO-EPM 验证：加载 adapter，从 RGB 图推理物体 3D 坐标，与真值对比误差。"""
import json
import re
import sys
from pathlib import Path

import mlx_vlm.models.rope_utils as rope_utils  # noqa: E402  (推理 forward 无需禁用，但保持一致)

from mlx_vlm import generate, load  # noqa: E402

MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-bf16"
ADAPTER = "/Users/yishuaicai/mywork/edps_cabto_exp/outputs/epm_data/adapters"
VAL_JSONL = "/Users/yishuaicai/mywork/edps_cabto_exp/outputs/epm_data/val.jsonl"
NUM = 8

PROMPT = (
    "这是桌面场景的图片。输出桌面上每个物体的 3D 世界坐标 (x, y, z)，"
    "单位米，格式为「物体名: (x, y, z)」，每行一个物体，不要输出其他内容。"
)

PAT = re.compile(r"(\w+)\s*[:：]\s*\(?\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)?")


def parse_answer(text):
    out = {}
    for line in text.splitlines():
        m = PAT.search(line)
        if m:
            out[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return out


def parse_truth(text):
    out = {}
    for line in text.splitlines():
        m = PAT.search(line)
        if m:
            out[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return out


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def main():
    print("loading model + adapter ...", flush=True)
    model, processor = load(MODEL, adapter_path=ADAPTER, processor_config={"trust_remote_code": True})

    rows = [json.loads(l) for l in Path(VAL_JSONL).read_text().splitlines()]
    rows = rows[:NUM]

    total_err = 0.0
    total_n = 0
    for i, row in enumerate(rows):
        img = row["images"][0]  # overview 图
        truth = parse_truth(row["answer"])
        out = generate(model, processor, PROMPT, image=[img], max_tokens=128)
        text = out.text if hasattr(out, "text") else str(out)
        pred = parse_answer(text)
        print(f"\n===== 样本 {i} =====", flush=True)
        print("图片:", Path(img).name, flush=True)
        print("模型输出:", repr(text), flush=True)
        for name, t in truth.items():
            if name in pred:
                e = dist(t, pred[name])
                total_err += e
                total_n += 1
                print(f"  {name}: 真值 {t}  预测 {pred[name]}  误差 {e*100:.2f} cm", flush=True)
            else:
                print(f"  {name}: 真值 {t}  预测 缺失", flush=True)

    if total_n:
        print(f"\n平均 3D 误差: {total_err/total_n*100:.2f} cm (共 {total_n} 个物体)", flush=True)


if __name__ == "__main__":
    main()
