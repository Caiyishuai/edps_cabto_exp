"""参考 STRIPS 动作库：把 7 类技能模板机械落地到每个 LIBERO 任务的具体对象/区域。

与 `outputs/remaining_cabto` 单臂技术路线一致：动作签名由工程指定，⟨pre,add,del⟩
为参考模型；LLM 可在其上做参考初始化修订（见 pipeline 的 propose），不冒充从零发现。

7 类技能（按 program_kind 路由到代码生成契约）：

| program_kind | 含义 | 原子效应 |
|--------------|------|----------|
| pick         | 抓起物体 | add holding, del 其所有位置原子 |
| place_on     | 放到表面/叠到物体上 | add on_region/on_object, add empty |
| place_in     | 放进容器 | add in_region, add empty（抽屉/微波炉需先 open） |
| open         | 打开抽屉/门 | add open, del closed |
| close        | 关闭抽屉/门 | add closed, del open |
| turn_on      | 打开电器 | add turned_on, del turned_off |
| turn_off     | 关闭电器 | add turned_off, del turned_on |
"""
from __future__ import annotations

from typing import Dict, List

from .atoms import Problem, container_open_entity


def _model(name, args, pre, add, delete, program_kind, description):
    return {
        "name": name,
        "args": args,
        "pre": sorted(set(pre)),
        "add": sorted(set(add)),
        "del": sorted(set(delete)),
        "program_kind": program_kind,
        "description": description,
    }


def reference_models(problem: Problem) -> List[Dict]:
    models: List[Dict] = []

    # 1. pick：抓起需要移动的物体。
    for obj in problem.pick_objects:
        models.append(_model(
            "pick", {"object": obj},
            pre=["empty()"],
            add=[f"holding({obj})"],
            delete=["empty()", *problem.location_atoms(obj)],
            program_kind="pick",
            description=f"Pick up {obj} and leave it held by the gripper.",
        ))

    # 2. place_on：放到表面区域或叠到另一物体上。
    for obj, target, kind in problem.place_on_targets:
        loc = f"on_region({obj},{target})" if kind == "region" else f"on_object({obj},{target})"
        models.append(_model(
            "place_on", {"object": obj, "target": target},
            pre=[f"holding({obj})"],
            add=[loc, "empty()"],
            delete=[f"holding({obj})"],
            program_kind="place_on",
            description=f"Place held {obj} on target {target} and release it.",
        ))

    # 3. place_in：放进容器（抽屉/微波炉需先打开对应实体）。
    for obj, region in problem.place_in_targets:
        open_entity = container_open_entity(problem.task, region)
        pre = [f"holding({obj})"]
        if open_entity is not None:
            pre.append(f"open({open_entity})")
        models.append(_model(
            "place_in", {"object": obj, "container": region},
            pre=pre,
            add=[f"in_region({obj},{region})", "empty()"],
            delete=[f"holding({obj})"],
            program_kind="place_in",
            description=f"Place held {obj} inside container region {region} and release it.",
        ))

    # 4/5. open / close 关节实体（抽屉区域或微波炉门）。
    for e in problem.open_close_entities:
        models.append(_model(
            "open", {"entity": e},
            pre=[f"closed({e})", "empty()"],
            add=[f"open({e})"],
            delete=[f"closed({e})"],
            program_kind="open",
            description=f"Open the articulated entity {e}.",
        ))
        models.append(_model(
            "close", {"entity": e},
            pre=[f"open({e})", "empty()"],
            add=[f"closed({e})"],
            delete=[f"open({e})"],
            program_kind="close",
            description=f"Close the articulated entity {e}.",
        ))

    # 6/7. turn_on / turn_off 电器（灶具）。
    for e in problem.turnable_entities:
        models.append(_model(
            "turn_on", {"entity": e},
            pre=[f"turned_off({e})", "empty()"],
            add=[f"turned_on({e})"],
            delete=[f"turned_off({e})"],
            program_kind="turn_on",
            description=f"Turn on the appliance {e}.",
        ))
        models.append(_model(
            "turn_off", {"entity": e},
            pre=[f"turned_on({e})", "empty()"],
            add=[f"turned_off({e})"],
            delete=[f"turned_on({e})"],
            program_kind="turn_off",
            description=f"Turn off the appliance {e}.",
        ))

    return models
