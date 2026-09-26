#!/usr/bin/env python3
"""探针 1：oracle 上限策略可达性 —— 归因「pre 错 vs 策略弱」的第一步。

背景
----
CABTO 第 2 步（VLM 实现动作 + 查一致性）的归因混淆：执行失败时，无法区分
「动作模型 pre 错了（物理不可达）」还是「策略实现能力不足（可达但策略做不到）」。

本脚本用底层 `RebuiltSingleArm` 内置的 oracle 上限策略（直接用真值位姿的
pick/place/home）做对照实验：

    - oracle 上限也失败  → pre 错（物理不可达），该修动作模型
    - oracle 上限成功    → pre 对（物理可达），后续与 VLM 策略对照才判「策略弱」

零模型、零训练、零 VLM，只需 MuJoCo 仿真 + oracle 位姿。跑通后得到每个
动作模型的「oracle 可达性」基准，是后续「VLM 策略失败归因」和「造训练数据」
的前置。

用法
----
    python3 probe_oracle.py cover          # cover 场景，seed=0（无 jitter）
    python3 probe_oracle.py blocks --seeds 0 1 2   # 多 seed（有 jitter 扰动）
    python3 probe_oracle.py cover --output out/cover
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # project/
PACKAGE = ROOT.parent                                # edps_cabto_exp/
sys.path.insert(0, str(ROOT))

from core.single_arm_rebuilt import RebuiltSingleArm  # noqa: E402
from core.cover_cabto_api import transition_check        # noqa: E402


# 标准动作模型 schema（与 cover models.json 的 schemas 一致，物理正确的参考）。
SCHEMAS = {
    "pick": {
        "pre": ["empty()", "at_source(OBJECT)"],
        "add": ["holding(OBJECT)"],
        "del": ["empty()", "at_source(OBJECT)", "home_completed()"],
    },
    "place": {
        "pre": ["holding(OBJECT)"],
        "add": ["at_target(OBJECT)", "empty()"],
        "del": ["holding(OBJECT)", "home_completed()"],
    },
    "home": {
        "pre": ["empty()"],
        "add": ["home_completed()"],
        "del": [],
    },
}

TASKS = {
    "cover": ROOT / "scenes/cover_kitchen_sort_v2.json",
    "blocks": ROOT / "scenes/blocks_rebuilt_v1.json",
}


def build_reference_models(config):
    """从 config 的 pairs 生成标准参考动作模型（pick/place × 每对 + home）。"""
    pairs = config["pairs"]
    models = []
    for obj, support in pairs:
        for kind in ("pick", "place"):
            sc = SCHEMAS[kind]
            models.append({
                "name": kind,
                "args": {"object": obj, "support": support},
                "pre": [a.replace("OBJECT", obj) for a in sc["pre"]],
                "add": [a.replace("OBJECT", obj) for a in sc["add"]],
                "del": [a.replace("OBJECT", obj) for a in sc["del"]],
            })
    models.append({"name": "home", "args": {}, **{k: list(v) for k, v in SCHEMAS["home"].items()}})
    return models


def effect_check(model, before, after, exception):
    """用完整 STRIPS 转移检查一致性。

    只把「与动作模型声称直接相关」的违规当作不一致，忽略碰撞噪声导致的
    无关谓词漂移（unexpected_added / unexpected_removed）。关键违规：
      - missing_pre           : 前置没满足（不该执行）
      - missing_add           : 声称的 add 没达成
      - remaining_del         : 声称的 del 没消失
      - missing_preserved_pre : pre 中该保持的谓词执行后消失（del 漏了某项）
    """
    chk = transition_check(model, set(before), set(after), exception)
    v = chk["violations"]
    key = {
        "missing_pre": v["missing_pre"],
        "missing_add": v["missing_add"],
        "remaining_del": v["remaining_del"],
        "missing_preserved_pre": v["missing_preserved_pre"],
    }
    ok = exception is None and not any(key.values())
    return ok, key


def probe_model(config, model, pairs, seed):
    """对单个动作模型跑 oracle 上限策略，返回可达性结论。"""
    name = model["name"]
    args = model.get("args", {})
    env = RebuiltSingleArm(config, seed=seed)
    try:
        # ---- 构造满足 pre 的前置状态（全部用 oracle 上限动作）----
        if name == "place":
            env.pick(args["object"])                     # -> holding(obj)
        elif name == "home":
            for obj, sup in pairs:
                env.pick(obj)
                env.place(obj, sup)                      # -> empty + all at_target

        before = sorted(env.state())
        pre_holds = set(model["pre"]) <= set(before)

        # ---- 执行 oracle 上限策略 ----
        exception = None
        try:
            if name == "pick":
                env.pick(args["object"])
            elif name == "place":
                env.place(args["object"], args["support"])
            elif name == "home":
                env.home()
            else:
                raise ValueError(f"unknown action {name}")
        except Exception as exc:
            exception = f"{type(exc).__name__}: {exc}"

        after = sorted(env.state())
        effect_ok, violations = effect_check(model, before, after, exception)

        return {
            "model": name,
            "args": args,
            "pre": model["pre"],
            "add": model["add"],
            "del": model["del"],
            "pre_holds": pre_holds,
            "before": before,
            "after": after,
            "effect_ok": effect_ok,
            "violations": violations,
            "exception": exception,
        }
    finally:
        env.close()


def run_task(task, seeds, output_dir):
    config = json.loads(TASKS[task].read_text())
    models = build_reference_models(config)
    pairs = config["pairs"]

    results = []
    for seed in seeds:
        for model in models:
            rec = probe_model(config, model, pairs, seed)
            rec["seed"] = seed
            rec["task"] = task
            results.append(rec)

    # ---- 汇总 ----
    summary = {}
    for rec in results:
        key = (rec["seed"], rec["model"], tuple(sorted(rec["args"].items())))
        summary.setdefault(key, []).append(rec["effect_ok"])
    rows = []
    for (seed, name, args), oks in summary.items():
        rows.append({
            "seed": seed, "model": name, "args": dict(args),
            "all_ok": all(oks), "ok": bool(oks[0]),
        })
    all_ok = all(r["effect_ok"] for r in results)

    payload = {
        "task": task,
        "seeds": seeds,
        "models": models,
        "results": results,
        "all_oracle_reachable": all_ok,
    }
    out = output_dir or ROOT / "attribution" / "out" / task
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe_oracle.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # ---- 打印 ----
    print(f"\n=== 探针1：oracle 上限可达性 | task={task} | seeds={seeds} ===")
    print(f"{'seed':<5}{'model':<7}{'args':<30}{'pre_holds':<10}{'effect_ok'}")
    for r in results:
        arg_s = json.dumps(r["args"], ensure_ascii=False)
        print(f"{r['seed']:<5}{r['model']:<7}{arg_s:<30}{str(r['pre_holds']):<10}{r['effect_ok']}")
        if not r["effect_ok"]:
            print(f"      violations={r['violations']} exc={r['exception']}")
    print(f"\n全部动作 oracle 可达: {all_ok}")
    print(f"结果写入: {out / 'probe_oracle.json'}")


def main():
    ap = argparse.ArgumentParser(description="探针1：oracle 上限策略可达性")
    ap.add_argument("task", choices=["cover", "blocks"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="要跑的 seed 列表（0=无 jitter，非0=有 jitter 扰动）")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    run_task(args.task, args.seeds, args.output)


if __name__ == "__main__":
    main()
