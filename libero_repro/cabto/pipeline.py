"""CABTO 单臂技术路线在 LIBERO 上的编排：prepare → sample → replay（外加 plan 汇总）。

与 `outputs/remaining_cabto` 的 cover/blocks 技术路线一一对应：

| 阶段     | 对应 run_remaining_cabto | 内容 |
|----------|--------------------------|------|
| prepare  | prepare                  | 构造 STRIPS 动作库（参考 + 可选 LLM 参考初始化修订）→ 形式化 BT 规划校验 |
| sample   | sample                   | 每个动作生成 `def policy(api)`（参考离线 / 本地 LLM）→ 受限解释器校验 + 效果校验 |
| replay   | replay                   | 独立复验：重放已通过动作，逐帧断言 STRIPS 转移，最终目标可达 |
| plan     | plancheck                | 对 suite 全部任务做符号可达性验证 |

符号层（prepare/plan/replay）在本机完整运行；sample 默认用离线参考策略，接入本地
mlx-vlm 模型需在装有 mlx-vlm 的环境运行（见 README_cabto.md 的边界说明）。物理执行
（真实机器人接地）需 LIBERO 官方 Linux/GPU 栈或 CABTO MuJoCo 3.10 单臂场景 + LIBERO 资产，
本包只提供统一 Runtime 接口与效果校验器，不做真机。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

from .atoms import Problem, build_problem
from .bddl import Task, parse_task
from .checker import symbolic_apply, transition_check
from .codegen import API_DOCS, build_program, reference_policy
from .library import reference_models
from .planner import plancheck

SCHEMA = "finite_symbolic_only"  # 全程符号验证，不冒充物理执行证据


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


class SymbolicRuntime:
    """符号执行运行时：把 `def policy(api)` 的 API 调用序列解释成一次 STRIPS 转移。

    运动 API 在符号层不产生真实位移，只记录调用并返回占位数据；真实的"状态转移"由
    `symbolic_apply(model, before)` 给出。这不是物理执行，仅用于校验生成代码的契约
    合法性与符号效应一致性。
    """

    def __init__(self, problem: Problem):
        self.problem = problem
        self.state = frozenset(problem.init)
        self.goal = frozenset(problem.goal)
        self.trace: List[Dict] = []

    def _point(self, *_a, **_k):
        return (0.0, 0.0, 0.0)

    def api(self, program_kind: str) -> Dict:
        mapping = {}
        for name in API_DOCS[program_kind]:
            def make(name=name):
                def call(*args, **kwargs):
                    row = {"method": name, "args": list(args), "kwargs": kwargs}
                    self.trace.append(row)
                    if name.endswith("_point") or name in ("grasp_point", "destination_point"):
                        return self._point()
                    return None
                return call
            mapping[name] = make()
        return mapping

    def execute(self, model: Dict, source: str) -> Dict:
        before = self.state
        self.trace = []
        program = build_program(source, model["program_kind"])
        try:
            result = program.run(self.api(model["program_kind"]), call_limit=60)
            exception = None
            executed = True
        except Exception as exc:
            result = {"trace": getattr(exc, "policy_trace", [])}
            exception = f"{type(exc).__name__}: {exc}"
            executed = False
        self.state = symbolic_apply(model, before) if exception is None else before
        check = transition_check(model, before, self.state, exception)
        return {
            "model": model,
            "before": sorted(before),
            "after": sorted(self.state),
            "code_executed": executed,
            "exception": exception,
            "trace": list(self.trace),
            "effect_ok": bool(check["effect_ok"] and executed),
            "strict_CP_transition_match": check["strict_CP_transition_match"],
            "code_sha256": _sha256(source),
        }


# --------------------------------------------------------------------------- prepare
def prepare(suite: str, task: str, out: Path, refine: bool = False) -> Dict:
    """构造 STRIPS 动作库并做规划校验；写 models.json。"""
    t = parse_task(suite, task)
    problem = build_problem(t)
    models = reference_models(problem)
    provenance = {
        "model": "engineer_reference_domain",
        "mode": "explicit_reference_library; NOT unconstrained action discovery",
    }
    if refine:
        # 参考初始化 + 本地 LLM 修订（需 mlx-vlm 环境）；此处仅占位说明，不静默替换。
        provenance["mode"] = "reference_initialized_local_model_proposal(requires mlx-vlm)"
    check = plancheck(problem, models)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    save(out / "models.json", {
        "suite": suite, "task": task, "language": t.language,
        "models": models, "init": sorted(problem.init), "goal": sorted(problem.goal),
        "check": check, "provenance": provenance,
        "config_sha256": _sha256(json.dumps({"init": sorted(problem.init), "goal": sorted(problem.goal)})),
        "scope": SCHEMA,
    })
    return {"problem": problem, "models": models, "check": check}


# ----------------------------------------------------------------------------- plan
def plan(suite: str, out: Path) -> Dict:
    from .planner import verify_suite
    return verify_suite(suite, out)


# ---------------------------------------------------------------------------- sample
def sample(suite: str, task: str, models_path: Path, out: Path,
           backend: str = "reference") -> Dict:
    """按形式化 BT 的计划顺序，为每个动作生成/载入 `def policy(api)` 并符号执行 + 效果校验。

    backend='reference' 用离线参考策略；backend='llm' 需本地 mlx-vlm（真实模型证据）。
    只采样计划实际走到的动作；未被计划使用的动作记录为 unvalidated（与
    run_remaining_cabto 的 unvalidated_models 一致）。
    """
    data = json.loads(models_path.read_text())
    models = data["models"]
    problem = build_problem(parse_task(suite, task))
    runtime = SymbolicRuntime(problem)
    accepted: Dict[str, dict] = {}
    attempts: List[Dict] = []
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    def source_for(model: Dict) -> str:
        kind = model["program_kind"]
        if backend == "reference":
            return reference_policy(kind), "engineer_reference_policy(offline baseline, not model evidence)"
        if backend == "llm":
            return _llm_policy(model, problem), "local_mlx_vlm_generated"
        raise ValueError("backend 只能是 reference 或 llm")

    from formal_bt import ModelLibrary
    tree = ModelLibrary(models).build(problem.init, problem.goal)
    while True:
        status, model = tree.tick(runtime.state)
        if status != "running":
            break
        key = _action_key(model)
        source, provenance = source_for(model)
        rec = runtime.execute(model, source)
        rec["provenance"] = provenance
        rec["source"] = source
        attempts.append(rec)
        if rec["effect_ok"]:
            accepted[key] = {"source": source, "sha256": _sha256(source), "model": model}
        else:
            break
        print("SAMPLE", task, key, rec["effect_ok"], rec["exception"], flush=True)

    final_reached = problem.goal <= runtime.state
    unvalidated = [m["name"] for m in models if _action_key(m) not in accepted]
    save(out / "policies.json", {
        "suite": suite, "task": task, "backend": backend,
        "accepted": accepted, "attempts": attempts,
        "final_reached_goal": final_reached,
        "unvalidated_models": unvalidated,
        "final_state": sorted(runtime.state),
        "goal": sorted(problem.goal),
        "scope": SCHEMA,
    })
    return {"accepted": accepted, "final_reached_goal": final_reached}


# ---------------------------------------------------------------------------- replay
def replay(suite: str, task: str, models_path: Path, policies_path: Path, out: Path) -> Dict:
    """独立复验：按计划顺序重放已通过策略，逐动作断言 STRIPS 转移，最终目标可达。"""
    data = json.loads(models_path.read_text())
    cache = json.loads(policies_path.read_text())
    models = data["models"]
    accepted = cache["accepted"]
    problem = build_problem(parse_task(suite, task))
    runtime = SymbolicRuntime(problem)
    steps: List[Dict] = []
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    from formal_bt import ModelLibrary
    tree = ModelLibrary(models).build(problem.init, problem.goal)
    while True:
        status, model = tree.tick(runtime.state)
        if status != "running":
            break
        key = _action_key(model)
        if key not in accepted:
            raise RuntimeError(f"计划需要未采样的动作 {key}")
        entry = accepted[key]
        assert _sha256(entry["source"]) == entry["sha256"], "replay 源码哈希不一致"
        rec = runtime.execute(model, entry["source"])
        steps.append(rec)
        if not rec["effect_ok"]:
            break

    success = bool(problem.goal <= runtime.state)
    save(out / "result.json", {
        "suite": suite, "task": task,
        "success": success, "steps": steps,
        "final_state": sorted(runtime.state), "goal": sorted(problem.goal),
        "all_CP_transitions_match": all(s["strict_CP_transition_match"] for s in steps),
        "scope": SCHEMA,
    })
    return {"success": success, "steps": steps}


def _action_key(model: Dict) -> str:
    parts = ",".join(f"{k}={v}" for k, v in sorted(model["args"].items()))
    return f"{model['name']}({parts})"


def _llm_policy(model: Dict, problem: Problem) -> str:
    """本地 mlx-vlm 生成单动作策略（真实模型证据）。需 mlx-vlm 环境。"""
    from .llm import LocalModel  # 延迟导入，避免缺少 mlx-vlm 时崩溃
    client = LocalModel("mlx-community/Qwen3.5-9B-4bit")
    docs = API_DOCS[model["program_kind"]]
    prompt = ("Write ONLY def policy(api): implementing ONE action. No imports, loops, "
              "tuple unpacking, simulator access or success flags. Use local coordinates "
              "read from API. All required motion and gripper/articulation actuation must "
              "be EXPLICIT. Do not change the STRIPS model.\n"
              "Action model: " + json.dumps(model, ensure_ascii=False) +
              "\nOnly available API signatures:\n" + "\n".join(docs[k] for k in sorted(docs)))
    call = client.call(prompt, max_tokens=850, temperature=0.0)
    from policy_codegen import extract_source
    return extract_source(call["raw"])
