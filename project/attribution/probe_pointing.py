#!/usr/bin/env python3
"""打点精度对比：3B vs 9B，回答「感知瓶颈是规模还是机制」。

绕过核心代码只认 JSON 的 parse_pixel，用鲁棒解析（JSON + 纯文本 `(x,y)` 都认），
对每个目标物体渲染真图 → VLM 打点 → 对比 oracle 投影，报告像素误差。

用法
----
    python3 probe_pointing.py blocks --models mlx-community/Qwen2.5-VL-3B-Instruct-4bit mlx-community/Qwen3.5-9B-4bit
    python3 probe_pointing.py cover  --models mlx-community/Qwen2.5-VL-3B-Instruct-4bit mlx-community/Qwen3.5-9B-4bit
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # project/
PACKAGE = ROOT.parent
sys.path.insert(0, str(ROOT))

from core.single_arm_rebuilt import RebuiltSingleArm   # noqa: E402
from core.cover_cabto_api import project                # noqa: E402
from core.cover_cabto_grounding import LocalModel       # noqa: E402
from probe_oracle import TASKS                          # noqa: E402


def robust_parse_point(raw):
    """鲁棒解析打点坐标，兼容多种输出格式：
    JSON point/point_2d、XML <points x1 y1>、纯文本 (x,y)/[x,y]、"x, y"。
    """
    if not raw:
        return None
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    text = re.sub(r"```(?:json)?|```", "", text).strip()
    # 1) JSON 字段 point / point_2d
    m = re.search(r'"point(?:_2d|_3d)?"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)', text)
    if m:
        return float(m.group(1)), float(m.group(2))
    # 2) XML <points x1=".." y1="..">
    m = re.search(r'x1\s*=\s*"([\d.]+)"[^>]*y1\s*=\s*"([\d.]+)"', text)
    if m:
        return float(m.group(1)), float(m.group(2))
    # 3) 括号/方括号 (x, y) / [x, y]
    m = re.search(r'[\(\[]\s*([\d.]+)\s*,\s*([\d.]+)\s*[\)\]]', text)
    if m:
        return float(m.group(1)), float(m.group(2))
    # 4) 逗号分隔的纯数字对，用负向前后瞻避免属性名（x1="361" 里的 1）
    m = re.search(r'(?<![\w.])(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)(?![\w.])', text)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def run(task, model_ids, output_dir):
    config = json.loads(TASKS[task].read_text())
    if task == "blocks":
        objects = list(config["blocks"].keys())
    else:
        objects = list(config["items"].keys()) + list(config["supports"].keys())

    runtime = RebuiltSingleArm(config, seed=0, render=True)
    env = runtime.env
    size = env.img_size
    cam = "front"
    img_dir = (output_dir or ROOT / "attribution" / "out" / task / "pointing")
    img_dir.mkdir(parents=True, exist_ok=True)

    # 预渲染每个物体的真图 + oracle 投影（只渲染一次，两个模型共用）
    shots = {}
    for name in objects:
        img = env.render_cam(cam)
        p = img_dir / f"{name}.png"
        from PIL import Image
        Image.fromarray(img).save(p)
        world = runtime.pos(name)
        gt_uv = project(env, np.asarray(world), cam, size)
        shots[name] = {"img_path": str(p), "gt_uv": gt_uv}

    print(f"\n=== 打点精度对比 | task={task} | 相机={cam} ===")
    print(f"{'模型':<45}{'物体':<8}{'打点':<14}{'真值':<14}{'误差(px)'}")
    summary = {}
    for model_id in model_ids:
        client = LocalModel(model_id)
        errs = []
        for name, s in shots.items():
            call = client.call(f"Point to the {name}. Output the pixel coordinate.",
                               [s["img_path"]], max_tokens=180)
            uv = robust_parse_point(call.get("raw", ""))
            gt = s["gt_uv"]
            if uv is None:
                err = None
                short = model_id.split("/")[-1]
                print(f"{short:<45}{name:<8}{'解析失败':<14}{np.round(gt,0).tolist()!s:<14}{'—'}")
            else:
                err = float(np.linalg.norm(np.array(uv) - np.array(gt)))
                short = model_id.split("/")[-1]
                print(f"{short:<45}{name:<8}{str(uv):<14}{str(np.round(gt,0).tolist()):<14}{err:.1f}")
                errs.append(err)
        if errs:
            summary[model_id] = {
                "mean_px": float(np.mean(errs)),
                "max_px": float(np.max(errs)),
                "parsed": len(errs),
                "total": len(objects),
            }
        print(f"  → {model_id.split('/')[-1]}: 解析成功 {len(errs)}/{len(objects)}，"
              f"平均误差 {np.mean(errs):.1f}px，最大 {np.max(errs):.1f}px\n")

    outdir = output_dir or ROOT / "attribution" / "out" / task
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "probe_pointing.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    runtime.close()


def main():
    ap = argparse.ArgumentParser(description="打点精度对比：3B vs 9B")
    ap.add_argument("task", choices=["cover", "blocks"])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    run(args.task, args.models, args.output)


if __name__ == "__main__":
    main()
