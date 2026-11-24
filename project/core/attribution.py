"""CABTO 归因器 V1（单臂 cover / blocks）。

把三探针对照实验封装成循环内可复用的组件，用于在策略采样/视觉闭环执行
失败时判断根因：动作模型错 / 策略弱 / 感知错 / 证据不足(abstain)。

三路探针（每路跑 k 次不同 seed，得到成功率 + 二项置信区间）：

    probe1  oracle 上限策略（真值位姿）  -> 物理可达性（模型 pre/效应对不对）
    probe2  VLM 策略 + oracle 定位       -> 策略代码能力
    probe3  VLM 策略 + qwen 定位         -> 感知误差（= 全视觉失败现场）

归因路由（阈值与置信区间可调）：

    p1 显著低                    -> model      （模型 pre/效应物理不可达）
    p1 高 & p2 低               -> policy     （策略代码能力不足）
    p2 高 & p3 低               -> perception （感知误差）
    三路都高                     -> ok
    任何一路置信区间跨阈值/重叠   -> abstain    （保持 unresolved，不修补）

设计原则：归因信号来自「换变量看结果变不变」的干预，不是历史统计；
置信区间重叠时宁可 abstain，也不把不确定的修补写进动作库。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # project/
PACKAGE = ROOT.parent                                # edps_cabto_exp/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PACKAGE / "CABTO/exp4_bt_tasks/stage3_cabto"))

from core.single_arm_rebuilt import RebuiltSingleArm   # noqa: E402
from core.cover_cabto_api import FineCover, API_DOCS, transition_check  # noqa: E402
from core.remaining_blocks_api import FineBlocks        # noqa: E402
from policy_codegen import PolicyProgram                # noqa: E402
from formal_bt import action_id                         # noqa: E402

TASKS = {
    "cover": ROOT / "scenes/cover_kitchen_sort_v2.json",
    "blocks": ROOT / "scenes/blocks_rebuilt_v1.json",
}

# 只把「与动作模型声称直接相关」的违规当作不一致，忽略碰撞噪声等无关谓词漂移。
KEY_VIOLATIONS = ("missing_pre", "missing_add", "remaining_del", "missing_preserved_pre")

# 归因结论标签（与下游路由一一对应）。
VERDICTS = ("model", "policy", "perception", "ok", "abstain")


def wilson(ok: int, n: int, z: float = 1.2815515655):
    """Wilson score interval 下/上界。z=1.2816 对应 80% 置信（小样本更稳健）。"""
    if n <= 0:
        return 0.0, 1.0
    p = ok / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def z_from_confidence(confidence: float) -> float:
    """把 0-1 置信度映射到正态 z 值（Wilson 区间用）。"""
    if confidence >= 0.99:
        return 2.5758
    if confidence >= 0.95:
        return 1.9600
    if confidence >= 0.90:
        return 1.6449
    return 1.2816


def key_violations(model, before, after, exception=None):
    v = transition_check(model, set(before), set(after), exception)["violations"]
    return {k: v[k] for k in KEY_VIOLATIONS}


def effect_ok(model, before, after, exception=None):
    kv = key_violations(model, before, after, exception)
    return (exception is None and not any(kv.values())), kv


# ---------------------------------------------------------------------------
# 三路探针
# ---------------------------------------------------------------------------

def oracle_probe(config, model, pairs, seed=0):
    """探针 1：oracle 上限策略可达性（真值位姿，零模型零训练零 VLM）。"""
    name = model["name"]
    args = model.get("args", {})
    env = RebuiltSingleArm(config, seed=seed)
    try:
        # 构造满足 pre 的前置状态（全部用 oracle 上限动作）。
        if name == "place":
            env.pick(args["object"])
        elif name == "home":
            for obj, sup in pairs:
                env.pick(obj)
                env.place(obj, sup)

        before = sorted(env.state())
        pre_holds = set(model["pre"]) <= set(before)
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
        ok, kv = effect_ok(model, before, after, exception)
        return {
            "ok": ok, "violations": kv, "exception": exception,
            "pre_holds": pre_holds, "before": before, "after": after,
        }
    finally:
        env.close()


def vlm_probe(task, config, model, source, pairs, seed=0, perception="oracle",
              out=None, client=None):
    """探针 2/3：VLM 生成的策略代码 + 指定定位后端执行。

    perception="oracle" -> 探针 2（隔离策略代码能力）
    perception="qwen"   -> 探针 3（全视觉链路，隔离感知误差）
    """
    name = model["name"]
    args = model.get("args", {})
    render = perception == "qwen"          # qwen 需要离屏渲染出真实图
    out = Path(out) if out is not None else Path(".")
    if task == "blocks":
        env = FineBlocks(config, seed=seed, render=render, perception=perception,
                         out=out, client=client)
    else:
        env = FineCover(config, seed=seed, render=render, perception=perception,
                        out=out, client=client)
    try:
        # blocks 的 qwen 感知需要在任何物体被移动之前缓存支撑块位置，
        # 否则 place 时被举起的对象会遮挡支撑块（与 replay 的 prepare_supports 语义一致）。
        if task == "blocks" and perception == "qwen" and hasattr(env, "perception"):
            env.perception.prepare_supports()

        # 前置状态用 oracle 上限构造（隔离目标动作本身的感知误差）。
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
        ok, kv = effect_ok(model, before, after, exception)
        return {
            "ok": ok, "violations": kv, "exception": exception,
            "before": before, "after": after,
        }
    finally:
        env.close()


# ---------------------------------------------------------------------------
# 统计化判定
# ---------------------------------------------------------------------------

def _status(ok, n, threshold, z):
    """把一路成功率映射到 low / high / ambiguous。"""
    lo, hi = wilson(ok, n, z)
    if hi < threshold:
        return "low", (lo, hi)
    if lo > threshold:
        return "high", (lo, hi)
    return "ambiguous", (lo, hi)


def classify(p1, p2, p3, threshold=0.5, z=1.2815515655):
    """根据三路 (ok, n) 判定根因。

    p1/p2/p3 为 (ok_count, total) 元组；p3 可为 None（未跑感知探针）。

    返回 (verdict, reason)。
    """
    s1, c1 = _status(p1[0], p1[1], threshold, z)
    s2, c2 = _status(p2[0], p2[1], threshold, z)

    if s1 == "low":
        return "model", "oracle 上限策略成功率显著低于阈值：动作模型 pre/效应物理不可达"
    if s1 == "high" and s2 == "low":
        return "policy", "oracle 可达但 VLM 策略(完美定位)失败：策略代码能力不足"

    if p3 is None:
        if s1 == "high" and s2 == "high":
            return "ok", "模型可达且策略(完美定位)通过；感知层未验证"
        return "abstain", "模型/策略层证据不足（置信区间重叠或 ambiguous）"

    s3, c3 = _status(p3[0], p3[1], threshold, z)
    if s2 == "high" and s3 == "low":
        return "perception", "策略(完美定位)成功但换真实感知失败：感知误差"
    if s1 == "high" and s2 == "high" and s3 == "high":
        return "ok", "三路全通"
    return "abstain", "证据不足（置信区间重叠或某路 ambiguous），保持 unresolved"


def _repeat(run, seeds):
    """跑多 seed，返回 (ok_count, total, [{seed, run}...])。"""
    ok_count = 0
    runs = []
    for s in seeds:
        r = run(s)
        runs.append({"seed": s, **r})
        ok_count += int(bool(r["ok"]))
    return ok_count, len(seeds), runs


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def attribute(task, model, source, client=None, k=3, seeds=None, out=None,
              threshold=0.5, confidence=0.8, run_perception=True,
              config=None, pairs=None):
    """对单个动作模型做统计化归因。

    参数
    ----
    task : "cover" | "blocks"
    model : 动作模型 dict（含 name / args / pre / add / del）
    source : 策略代码字符串（def policy(api): ...）
    client : qwen 感知 VLM（run_perception=True 时必需）
    k / seeds : 采样次数 / 具体 seed 列表（默认 seeds=[0..k-1]）
    threshold : 判 low/high 的成功率分界（默认 0.5）
    confidence : Wilson 区间置信度（默认 0.8）
    run_perception : 是否跑探针 3（感知层）

    返回结构化归因记录，verdict 为 VERDICTS 之一。
    """
    config = config or json.loads(TASKS[task].read_text())
    pairs = pairs or config["pairs"]
    seeds = list(seeds) if seeds is not None else list(range(k))
    z = z_from_confidence(confidence)
    out = Path(out) if out is not None else ROOT / "attribution" / "out" / task / "attributor"

    p1 = _repeat(lambda s: oracle_probe(config, model, pairs, s), seeds)
    p2 = _repeat(
        lambda s: vlm_probe(task, config, model, source, pairs, s, "oracle", out / "oracle"),
        seeds)
    p3 = None
    if run_perception:
        if client is None:
            raise ValueError("run_perception=True 需要提供 client（qwen 感知 VLM）")
        p3 = _repeat(
            lambda s: vlm_probe(task, config, model, source, pairs, s, "qwen",
                                out / "qwen", client),
            seeds)

    verdict, reason = classify((p1[0], p1[1]), (p2[0], p2[1]),
                               (p3[0], p3[1]) if p3 else None, threshold, z)

    def pack(ok, total, runs):
        return {
            "ok_count": ok, "total": total, "rate": round(ok / total, 4),
            "ci": [round(wilson(ok, total, z)[0], 4), round(wilson(ok, total, z)[1], 4)],
            "runs": runs,
        }

    return {
        "task": task,
        "action_id": action_id(model),
        "model": model,
        "seeds": seeds, "threshold": threshold, "confidence": confidence,
        "verdict": verdict, "reason": reason,
        "probes": {
            "oracle": pack(p1[0], p1[1], p1[2]),
            "policy": pack(p2[0], p2[1], p2[2]),
            "perception": pack(p3[0], p3[1], p3[2]) if p3 else None,
        },
    }


def attribute_accepted(task, client=None, k=3, seeds=None, out=None,
                       threshold=0.5, confidence=0.8, run_perception=True,
                       policies_path=None, action_id=None):
    """对已采样 accepted 的动作库逐个归因。

    policies_path 缺省时用单臂任务的已知采样产物路径。
    action_id 非空时只归因那一个动作。
    """
    if policies_path is None:
        known = {
            "cover": PACKAGE / "outputs/cover_cabto_fix/sampling_9b_complete/policies.json",
            "blocks": PACKAGE / "outputs/remaining_cabto/blocks/sampling/policies.json",
        }
        policies_path = known[task]
    accepted = json.loads(Path(policies_path).read_text())["accepted"]
    config = json.loads(TASKS[task].read_text())
    pairs = config["pairs"]

    rows = []
    for aid, rec in accepted.items():
        if action_id is not None and aid != action_id:
            continue
        row = attribute(task, rec["model"], rec["source"], client=client, k=k,
                        seeds=seeds, out=out, threshold=threshold,
                        confidence=confidence, run_perception=run_perception,
                        config=config, pairs=pairs)
        rows.append(row)
    return rows
