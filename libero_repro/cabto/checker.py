"""效果校验：STRIPS ⟨pre,add,del⟩ 转移一致性检查（与 cover_cabto_api.transition_check 同构）。

调用方（而非生成程序）拥有独立符号观测。`transition_check` 比较动作模型声明的
⟨pre,add,del⟩ 与真实观测到的 before→after 转移，输出四类违约与严格帧一致判定。

`symbolic_apply` 是纯符号的转移函数：`(before | add) - del`，用于符号层自检与代码生成
阶段的"参考转移"对照（不代表物理执行证据）。
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional


def symbolic_apply(model: Dict, before: Iterable[str]):
    pre, add, delete = set(model["pre"]), set(model["add"]), set(model["del"])
    if not pre <= set(before):
        raise ValueError("Symbolic precondition does not hold")
    return frozenset((set(before) | add) - delete)


def transition_check(model: Dict, before, after, exception=None) -> Dict:
    """比较声明效应与观测转移。before/after 为原子集合（list/set）。"""
    pre, add, delete = (set(model[k]) for k in ("pre", "add", "del"))
    before = set(before)
    after = set(after)
    predicted = (before | add) - delete
    violations = {
        "missing_pre": sorted(pre - before),
        "missing_add": sorted(add - after),
        "remaining_del": sorted(delete & after),
        "missing_preserved_pre": sorted((pre - delete) - after),
        "unexpected_added": sorted(after - predicted),
        "unexpected_removed": sorted(predicted - after),
    }
    return {
        "valid_trial": pre <= before,
        "paper_line20": (pre | add) - delete <= after,
        "strict_CP_transition_match": after == predicted,
        "goal_ok": add <= after,
        "effect_ok": exception is None and not any(violations.values()),
        "violations": violations,
    }


def check_transition(model: Dict, before, after, exception: Optional[str] = None) -> Dict:
    """便捷别名：保持与 pipeline 的调用命名一致。"""
    return transition_check(model, before, after, exception)
