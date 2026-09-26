#!/usr/bin/env python3
"""探针 2：VLM 策略 + oracle 定位 —— 解耦「策略弱 vs 感知错」。

与探针 1（oracle 上限策略）对照，对每个动作模型同时跑两条路：

    探针 1（oracle 上限）: RebuiltSingleArm.pick/place/home（直接用真值位姿）
    探针 2（VLM 策略）  : 模型生成的 `def policy(api)` 代码 + oracle 定位

归因决策（这是「归因混淆」解耦的核心）：

    probe1 失败                    → pre 错（物理不可达，该修动作模型）
    probe1 成功 & probe2 失败      → 策略弱（代码/参数能力不足）
    probe1 成功 & probe2 成功      → 动作模型 + 策略都 OK（若之后换 qwen 定位失败，
                                     那才是感知错，由探针 3 处理）

本探针固定感知为 oracle 定位，单独隔离「策略代码能力」这一个变量。

用法
----
    python3 probe_vlm.py cover
    python3 probe_vlm.py blocks
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
from policy_codegen import PolicyProgram                       # noqa: E402
from probe_oracle import probe_model as probe1, TASKS          # noqa: E402

# VLM 生成策略代码（accepted 字段）来源。
POLICIES = {
    "cover": PACKAGE / "outputs/cover_cabto_fix/sampling_9b_complete/policies.json",
    "blocks": PACKAGE / "outputs/remaining_cabto/blocks/sampling/policies.json",
}


def make_env(task, config, out):
    """构造高层 env（perception=oracle 固定感知，client=None）。"""
    if task == "blocks":
        return FineBlocks(config, seed=0, render=False, perception="oracle", out=out, client=None)
    return FineCover(config, seed=0, render=False, perception="oracle", out=out, client=None)


def probe2_vlm(task, config, model, source, pairs, out):
    """探针 2：用 VLM 生成的策略代码 + oracle 定位执行，返回一致性结论。"""
    name = model["name"]
    args = model.get("args", {})
    env = make_env(task, config, out)
    try:
        # ---- 构造满足 pre 的前置状态（用 oracle 上限动作）----
        if name == "place":
            env.r.pick(args["object"])                   # -> holding(obj)
        elif name == "home":
            for obj, sup in pairs:
                env.r.pick(obj)
                env.r.place(obj, sup)                    # -> empty + all at_target

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
        # 用与 probe_oracle 相同的关键违规判定。
        v = probe1_effect_check(model, before, after, exception)
        return {
            "model": name, "args": args,
            "before": before, "after": after,
            "effect_ok": exception is None and not any(v.values()),
            "violations": v, "exception": exception,
        }
    finally:
        env.close()


def probe1_effect_check(model, before, after, exception):
    """关键违规判定（与 probe_oracle.effect_check 同款，这里内联避免循环 import）。"""
    from core.cover_cabto_api import transition_check
    v = transition_check(model, set(before), set(after), exception)["violations"]
    return {k: v[k] for k in ("missing_pre", "missing_add", "remaining_del", "missing_preserved_pre")}


def run_task(task, output_dir):
    config = json.loads(TASKS[task].read_text())
    pairs = config["pairs"]
    accepted = json.loads(POLICIES[task].read_text())["accepted"]

    rows = []
    for action_id, rec in accepted.items():
        model = rec["model"]
        source = rec["source"]
        seed = 0

        # 探针 1：oracle 上限
        p1 = probe1(config, model, pairs, seed)

        # 探针 2：VLM 策略 + oracle 定位
        out = (output_dir or ROOT / "attribution" / "out" / task / "vlm")
        p2 = probe2_vlm(task, config, model, source, pairs, out)

        # 归因
        if not p1["effect_ok"]:
            attr = "pre错(物理不可达)"
        elif not p2["effect_ok"]:
            attr = "策略弱(代码能力不足)"
        else:
            attr = "OK(动作模型+策略都通)"

        rows.append({
            "action_id": action_id,
            "probe1_oracle_ok": p1["effect_ok"],
            "probe2_vlm_ok": p2["effect_ok"],
            "attribution": attr,
            "probe2_violations": p2["violations"],
            "probe2_exception": p2["exception"],
        })

    payload = {"task": task, "rows": rows}
    outdir = output_dir or ROOT / "attribution" / "out" / task
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "probe_vlm.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"\n=== 探针2：VLM策略+oracle定位 | task={task} ===")
    print(f"{'action_id':<32}{'probe1(oracle)':<16}{'probe2(vlm)':<12}{'归因'}")
    for r in rows:
        print(f"{r['action_id']:<32}{str(r['probe1_oracle_ok']):<16}{str(r['probe2_vlm_ok']):<12}{r['attribution']}")
        if not r["probe2_vlm_ok"]:
            print(f"      violations={r['probe2_violations']} exc={r['probe2_exception']}")
    print(f"\n结果写入: {outdir / 'probe_vlm.json'}")


def main():
    ap = argparse.ArgumentParser(description="探针2：VLM策略 + oracle定位")
    ap.add_argument("task", choices=["cover", "blocks"])
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    run_task(args.task, args.output)


if __name__ == "__main__":
    main()
