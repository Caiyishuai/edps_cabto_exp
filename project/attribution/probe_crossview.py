#!/usr/bin/env python3
"""诊断 cover 的两视角（overview/front）定位不一致。

对每个物体，用 VLM 在两个视角各打一次点，拆解 cross-view disagreement 的来源：
  1. 打点像素 vs oracle 投影（每视角像素误差）
  2. 反投影世界坐标 vs oracle 真值（每视角世界误差）
  3. cross-view disagreement（两视角反投影差）

判断：是「模型打点偏」还是「反投影几何/高度假设」导致的。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent
sys.path.insert(0, str(ROOT))

from core.single_arm_rebuilt import RebuiltSingleArm   # noqa: E402
from core.cover_cabto_api import project, intersect_plane  # noqa: E402
from core.cover_cabto_grounding import LocalModel       # noqa: E402
from probe_oracle import TASKS                          # noqa: E402

DESC = {
    "shrimp": "orange shrimp at the front of the table",
    "apple": "red apple at the front of the table",
    "potato": "brown potato at the front of the table",
}


def robust_parse(raw):
    if not raw:
        return None
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    text = re.sub(r"```(?:json)?|```", "", text).strip()
    m = re.search(r'"point(?:_2d|_3d)?"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)', text)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r'\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)', text)
    if m:
        return float(m.group(1)), float(m.group(2))
    nums = [float(n) for n in re.findall(r'\d+(?:\.\d+)?', text)]
    if len(nums) >= 2:
        return nums[0], nums[1]
    return None


def run(model_id, output_dir):
    config = json.loads(TASKS["cover"].read_text())
    runtime = RebuiltSingleArm(config, seed=0, render=True)
    env = runtime.env
    size = env.img_size
    client = LocalModel(model_id)
    img_dir = output_dir or ROOT / "attribution" / "out" / "cover" / "crossview"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== cross-view 诊断 | cover | {model_id} ===")
    for obj in DESC:
        true_xy = runtime.pos(obj)[:2]
        z = runtime.initial[obj][2] + config["grip_offset_z"][obj]   # grasp 高度
        ests, errs_px = {}, {}
        for cam in ("overview", "front"):
            img = env.render_cam(cam)
            p = img_dir / f"{obj}_{cam}.png"
            from PIL import Image
            Image.fromarray(img).save(p)
            gt_uv = project(env, np.r_[true_xy, z], cam, size)
            call = client.call(f"Point to the {DESC[obj]}. Output the pixel coordinate.",
                               [str(p)], max_tokens=180)
            uv = robust_parse(call.get("raw", ""))
            if uv is None:
                ests[cam] = None
                errs_px[cam] = None
                print(f"  {obj} [{cam}] 解析失败 raw={call.get('raw','')[:80]!r}")
                continue
            est = intersect_plane(env, np.asarray(uv), cam, size, z)
            px_err = float(np.linalg.norm(np.array(uv) - gt_uv))
            world_err = float(np.linalg.norm(est[:2] - true_xy))
            ests[cam] = est
            errs_px[cam] = px_err
            print(f"  {obj} [{cam:<9}] 打点={str(uv):<14} 真值px={np.round(gt_uv,0).tolist()!s:<14} "
                  f"px误差={px_err:.1f} 世界误差={world_err:.3f}m")
        if ests["overview"] is not None and ests["front"] is not None:
            d = float(np.linalg.norm(ests["overview"][:2] - ests["front"][:2]))
            print(f"      → cross-view disagreement = {d:.3f}m")
        print()

    runtime.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    run(args.model, args.output)


if __name__ == "__main__":
    main()
