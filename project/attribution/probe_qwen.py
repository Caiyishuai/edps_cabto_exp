#!/usr/bin/env python3
"""探针 3：VLM 策略 + qwen 定位 —— 解耦「感知错」，输出完整三路归因。

在前两个探针基础上加第三路（qwen 定位），对每个动作模型跑三路对照：

    探针 1（oracle 上限）       : RebuiltSingleArm.pick/place/home（真值位姿）
    探针 2（VLM 策略+oracle 定位）: 生成的 def policy(api) 代码 + oracle 定位
    探针 3（VLM 策略+qwen 定位）  : 生成的 def policy(api) 代码 + Qwen-VL 打点定位

完整归因决策树：

    probe1 失败                                  → pre 错（物理不可达）
    probe1 成功 & probe2 失败                    → 策略弱（代码能力不足）
    probe1 成功 & probe2 成功 & probe3 失败      → 感知错（qwen 定位误差）
    三路全成功                                   → OK

用法
----
    python3 probe_qwen.py cover
    python3 probe_qwen.py blocks [--model mlx-community/Qwen2.5-VL-3B-Instruct-4bit]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # project/
PACKAGE = ROOT.parent                                # edps_cabto_exp/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PACKAGE / "CABTO/exp4_bt_tasks/stage3_cabto"))  # policy_codegen

from core.cover_cabto_api import FineCover, API_DOCS          # noqa: E402
from core.remaining_blocks_api import FineBlocks              # noqa: E402
from core.cover_cabto_grounding import LocalModel              # noqa: E402
from policy_codegen import PolicyProgram                       # noqa: E402
from probe_oracle import probe_model as probe1, TASKS          # noqa: E402
from probe_vlm import POLICIES, probe2_vlm, probe1_effect_check  # noqa: E402

DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"


def make_env_qwen(task, config, out, client):
    """构造 qwen 感知的 env（client 必须是已加载的 VLM）。

    注意：qwen 感知要 render_cam 出真实图片，必须 render=True（离屏渲染，
    不弹窗）。render=False 会让 render_cam 返回全黑图，导致模型看到空场景。
    """
    if task == "blocks":
        return FineBlocks(config, seed=0, render=True, perception="qwen", out=out, client=client)
    return FineCover(config, seed=0, render=True, perception="qwen", out=out, client=client)


def probe3_qwen(task, config, model, source, pairs, out, client):
    """探针 3：VLM 策略代码 + qwen 定位执行。"""
    name = model["name"]
    args = model.get("args", {})
    env = make_env_qwen(task, config, out, client)
    try:
        # 前置状态用 oracle 上限（隔离目标动作的感知误差，不把前置 pick 的感知误差算进来）
        if name == "place":
            env.r.pick(args["object"])
        elif name == "home":
            for obj, sup in pairs:
                env.r.pick(obj)
                env.r.place(obj, sup)

        before = sorted(env.state())
        exception = None
        try:
            api = env.bind(model)
            program = PolicyProgram(source, list(API_DOCS[name].keys()))
            program.validate_api(api)
            program.run(api, call_limit=60)
        except Exception as exc:
            exception = f"{type(exc).__name__}: {exc}"

        after = sorted(env.state())
        v = probe1_effect_check(model, before, after, exception)
        return {
            "model": name, "args": args,
            "before": before, "after": after,
            "effect_ok": exception is None and not any(v.values()),
            "violations": v, "exception": exception,
        }
    finally:
        env.close()


def run_task(task, model_id, output_dir):
    config = json.loads(TASKS[task].read_text())
    pairs = config["pairs"]
    accepted = json.loads(POLICIES[task].read_text())["accepted"]

    print(f"加载 VLM: {model_id} ...")
    client = LocalModel(model_id)

    rows = []
    for action_id, rec in accepted.items():
        model = rec["model"]
        source = rec["source"]
        out = (output_dir or ROOT / "attribution" / "out" / task / "qwen")

        p1 = probe1(config, model, pairs, 0)
        p2 = probe2_vlm(task, config, model, source, pairs, out)
        p3 = probe3_qwen(task, config, model, source, pairs, out, client)

        if not p1["effect_ok"]:
            attr = "pre错(物理不可达)"
        elif not p2["effect_ok"]:
            attr = "策略弱(代码能力不足)"
        elif not p3["effect_ok"]:
            attr = "感知错(qwen定位误差)"
        else:
            attr = "OK(全链路通)"

        rows.append({
            "action_id": action_id,
            "probe1_oracle_ok": p1["effect_ok"],
            "probe2_vlm_oracle_ok": p2["effect_ok"],
            "probe3_vlm_qwen_ok": p3["effect_ok"],
            "attribution": attr,
            "probe3_violations": p3["violations"],
            "probe3_exception": p3["exception"],
        })

    payload = {"task": task, "model_id": model_id, "rows": rows}
    outdir = output_dir or ROOT / "attribution" / "out" / task
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "probe_qwen.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"\n=== 探针3：三路归因 | task={task} | model={model_id} ===")
    print(f"{'action_id':<32}{'p1(oracle)':<11}{'p2(vlm+oracle)':<15}{'p3(vlm+qwen)':<13}{'归因'}")
    for r in rows:
        print(f"{r['action_id']:<32}{str(r['probe1_oracle_ok']):<11}"
              f"{str(r['probe2_vlm_oracle_ok']):<15}{str(r['probe3_vlm_qwen_ok']):<13}{r['attribution']}")
        if not r["probe3_vlm_qwen_ok"]:
            print(f"      violations={r['probe3_violations']} exc={r['probe3_exception']}")
    print(f"\n结果写入: {outdir / 'probe_qwen.json'}")


def main():
    ap = argparse.ArgumentParser(description="探针3：VLM策略 + qwen定位，三路归因")
    ap.add_argument("task", choices=["cover", "blocks"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    run_task(args.task, args.model, args.output)


if __name__ == "__main__":
    main()
