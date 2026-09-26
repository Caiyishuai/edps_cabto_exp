#!/usr/bin/env python3
"""CABTO on LIBERO 命令行入口。

用法（在 libero_repro 目录，用 .venv/bin/python）：
  python run_cabto.py plan <suite> [--output DIR]                 # 符号规划验证整个 suite
  python run_cabto.py plan all [--output DIR]                     # 验证全部 130 任务
  python run_cabto.py prepare <suite> --task TASK [--output DIR]  # 构造动作库 + 规划校验
  python run_cabto.py sample  <suite> --task TASK --models M [--backend reference|llm] --output DIR
  python run_cabto.py replay  <suite> --task TASK --models M --policies P --output DIR

示例：
  .venv/bin/python run_cabto.py plan libero_object --output runs/plan_object
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cabto.bddl import SUITES, list_tasks  # noqa: E402
from cabto import pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="CABTO on LIBERO single-arm benchmark")
    ap.add_argument("phase", choices=["plan", "prepare", "sample", "replay"])
    ap.add_argument("suite", help=f"suite 名（{SUITES}）或 all")
    ap.add_argument("--task", help="单任务名（prepare/sample/replay 必填）")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--models", type=Path)
    ap.add_argument("--policies", type=Path)
    ap.add_argument("--backend", default="reference", choices=["reference", "llm"])
    ap.add_argument("--refine", action="store_true", help="prepare 时用本地 LLM 修订（需 mlx-vlm）")
    args = ap.parse_args()

    out = args.output or Path("runs") / args.phase / (args.suite if args.suite != "all" else "all")
    out.mkdir(parents=True, exist_ok=True)

    if args.phase == "plan":
        suites = SUITES if args.suite == "all" else [args.suite]
        summary = {"suites": {}}
        for s in suites:
            summary["suites"][s] = pipeline.plan(s, out / s)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        solved = sum(v["solved"] for v in summary["suites"].values())
        total = sum(v["total"] for v in summary["suites"].values())
        print(f"PLAN  {solved}/{total} 任务符号可达")
        return 0 if solved == total else 2

    if args.phase == "prepare":
        if not args.task:
            ap.error("prepare 需要 --task")
        r = pipeline.prepare(args.suite, args.task, out, refine=args.refine)
        print("PREPARE", args.task, "reached_goal=", r["check"]["reached_goal"])
        return 0 if r["check"]["reached_goal"] else 2

    if args.phase == "sample":
        if not (args.task and args.models):
            ap.error("sample 需要 --task 与 --models")
        r = pipeline.sample(args.suite, args.task, args.models, out, backend=args.backend)
        print("SAMPLE", args.task, "final_reached_goal=", r["final_reached_goal"])
        return 0 if r["final_reached_goal"] else 2

    if args.phase == "replay":
        if not (args.task and args.models and args.policies):
            ap.error("replay 需要 --task/--models/--policies")
        r = pipeline.replay(args.suite, args.task, args.models, args.policies, out)
        print("REPLAY", args.task, "success=", r["success"])
        return 0 if r["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
