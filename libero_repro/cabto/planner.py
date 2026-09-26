"""形式化 BT 规划 + 符号可达性验证（复用 CABTO 的 formal_bt 内核）。

入口 `plancheck`：给定一个 Problem 的 init/goal 与 STRIPS 动作库，用
`formal_bt.ModelLibrary.build` 合成成本最小行为树，再 `symbolic_dry_run` 做有限符号重放，
判定目标是否可达。这是纯符号验证，不涉及物理执行。

`formal_bt` 内核位于 CABTO 上游工作区的 `CABTO/exp4_bt_tasks/stage3_cabto`，BT 扩展内核
位于 `BTExpansion-demo`（依赖 py_trees / tabulate，已装入 libero_repro/.venv）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from . import STAGE3_CABTO
from .atoms import Problem

if str(STAGE3_CABTO) not in sys.path:
    sys.path.insert(0, str(STAGE3_CABTO))

from formal_bt import ModelLibrary, action_id, symbolic_dry_run, validate_model  # noqa: E402


def plancheck(problem: Problem, models: List[Dict], out: Optional[Path] = None) -> Dict:
    """对一个任务做形式化 BT 规划 + 符号重放，返回结构化结果。"""
    for m in models:
        validate_model(m)
    library = ModelLibrary(models)
    start = problem.init
    goal = problem.goal
    try:
        bt = library.build(start, goal)
        result = symbolic_dry_run(bt, start, goal, 256)
    except Exception as exc:
        return {
            "problem": problem.id,
            "reached_goal": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "steps": 0,
            "action_ids": [],
            "model_count": len(models),
            "goal": sorted(goal),
            "scope": "symbolic_only",
        }
    report = {
        "problem": problem.id,
        "reached_goal": result["reached_goal"],
        "reason": result["reason"],
        "steps": result["steps"],
        "action_ids": result["action_ids"],
        "model_count": len(models),
        "goal": sorted(goal),
        "scope": "symbolic_only",
    }
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "tree.json").write_text(bt.export())
        (out / "tree.dot").write_text(bt.export("dot"))
        report["tree_nodes"] = len(json.loads(bt.export())["nodes"])
    return report


def verify_suite(suite: str, out: Optional[Path] = None) -> Dict:
    """对某个 suite 的全部任务做符号规划验证，返回汇总。"""
    from .bddl import load_suite
    from .atoms import build_problem
    from .library import reference_models

    reports = []
    for task in load_suite(suite):
        problem = build_problem(task)
        models = reference_models(problem)
        out_dir = out / problem.task.name if out is not None else None
        reports.append(plancheck(problem, models, out_dir))

    summary = {
        "suite": suite,
        "total": len(reports),
        "solved": sum(r["reached_goal"] for r in reports),
        "reports": reports,
        "scope": "symbolic_only",
    }
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
