"""LIBERO 谓词 → CABTO STRIPS 原子规范化 + 每任务的问题构造。

LIBERO 的 6 类谓词与 CABTO 原子的对应关系（`On` 按第二参数区分区域/物体）：

| LIBERO 谓词        | 含义             | CABTO 原子                  |
|--------------------|------------------|------------------------------|
| `On(o, region)`    | 物体在表面区域   | `on_region(o, region)`       |
| `On(o1, o2)`       | 物体叠在物体上   | `on_object(o1, o2)`          |
| `In(o, region)`    | 物体在容器内     | `in_region(o, region)`       |
| `Open(x)`          | 抽屉/门打开      | `open(x)`                    |
| `Close(x)`         | 抽屉/门关闭      | `closed(x)`                  |
| `Turnon(x)`        | 电器开启         | `turned_on(x)`               |
| `Turnoff(x)`       | 电器关闭         | `turned_off(x)`              |
| （机器人自身）     | 手爪状态         | `empty()` / `holding(o)`     |

关节实体的类别判定依赖 fixture 类别：
- `*_cabinet` + 区域名 `top/middle/bottom_region` → 抽屉区域（可 open/close）
- `microwave` → 门（可 open/close）
- `flat_stove` → 可 turnon/turnoff
- 其余（basket / tray / desk_caddy / shelf 等）→ 静态容器或表面，无关节。

LIBERO 的 init 只列物体摆放，不列关节的默认闭态；本模块在构造 init 时为每个
`open/close` 实体补 `closed(e)`、为每个 turnable 实体补 `turned_off(e)`（若 init 未显式给出）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .bddl import Task

# 抽屉区域名（target 为 cabinet fixture 时）
_DRAWER_NAMES = {"top_region", "middle_region", "bottom_region"}
_CABINET_CATEGORIES = {"wooden_cabinet", "white_cabinet"}
_MICROWAVE_CATEGORIES = {"microwave"}
_STOVE_CATEGORIES = {"flat_stove"}


def _region_short_name(task: Task, region: str) -> str:
    target = task.regions[region]["target"]
    if region.startswith(target + "_"):
        return region[len(target) + 1:]
    return region


def surface_regions(task: Task) -> List[str]:
    """可 `On` 放置的表面区域（含 top_side / cook_region / *_region 等）。"""
    return list(task.regions.keys())


def drawer_regions(task: Task) -> List[str]:
    """target 为 cabinet 且区域名为抽屉名的区域。"""
    out = []
    for r in task.regions:
        t = task.regions[r]["target"]
        cat = task.category_of(t)
        if cat in _CABINET_CATEGORIES and _region_short_name(task, r) in _DRAWER_NAMES:
            out.append(r)
    return out


def _region_is_microwave(task: Task, r: str) -> bool:
    t = task.regions[r]["target"]
    return task.category_of(t) in _MICROWAVE_CATEGORIES


def open_close_entities(task: Task) -> List[str]:
    """所有可 open/close 的实体：open/close 事实中的，以及由 `In` 目标推断的抽屉/微波炉门。

    只推断本任务实际用到的关节实体，避免给纯 pick/place 任务塞入无关抽屉。
    """
    entities = set()
    for fact in task.initial_state + task.goal_state:
        if fact[0] in ("open", "close") and len(fact) >= 2:
            entities.add(fact[1])
    drawer = set(drawer_regions(task))
    for fact in task.goal_state:
        if fact[0] == "in" and len(fact) == 3:
            r = fact[2]
            if r in drawer:
                entities.add(r)
            elif _region_is_microwave(task, r):
                entities.add(task.regions[r]["target"])
    return sorted(entities)


def container_open_entity(task: Task, region: str):
    """把物体放进容器 region 前需要打开哪个实体；无需打开则返回 None。"""
    if region in set(drawer_regions(task)):
        return region
    if _region_is_microwave(task, region):
        return task.regions[region]["target"]
    return None


def turnable_entities(task: Task) -> List[str]:
    entities = set()
    for fact in task.initial_state + task.goal_state:
        if fact[0] in ("turnon", "turnoff") and len(fact) >= 2:
            entities.add(fact[1])
    return sorted(entities)


def _translate_fact(task: Task, fact: List[str]):
    """返回 (atom, is_static)；static 表示 fixture 背景事实（不参与动作）。"""
    pred = fact[0]
    if pred == "on" and len(fact) == 3:
        obj, y = fact[1], fact[2]
        # fixture 放置在区域上是静态背景，直接忽略（不进入动态原子）。
        if task.category_of(obj) in task.fixtures.values() or any(
            obj in vals for vals in task.fixtures.values()
        ):
            return None, True
        if y in task.regions:
            return f"on_region({obj},{y})", False
        return f"on_object({obj},{y})", False  # 物体叠物体
    if pred == "in" and len(fact) == 3:
        return f"in_region({fact[1]},{fact[2]})", False
    if pred == "open" and len(fact) == 2:
        return f"open({fact[1]})", False
    if pred == "close" and len(fact) == 2:
        return f"closed({fact[1]})", False
    if pred == "turnon" and len(fact) == 2:
        return f"turned_on({fact[1]})", False
    if pred == "turnoff" and len(fact) == 2:
        return f"turned_off({fact[1]})", False
    raise ValueError(f"未识别的谓词: {fact}")


def _is_fixture(task: Task, instance: str) -> bool:
    return any(instance in vals for vals in task.fixtures.values())


@dataclass
class Problem:
    """一个 LIBERO 任务映射到 CABTO 的符号规划问题。"""

    task: Task
    init: frozenset = field(default_factory=frozenset)
    goal: frozenset = field(default_factory=frozenset)
    atoms: frozenset = field(default_factory=frozenset)
    surface_regions: List[str] = field(default_factory=list)
    drawer_regions: List[str] = field(default_factory=list)
    open_close_entities: List[str] = field(default_factory=list)
    turnable_entities: List[str] = field(default_factory=list)
    pick_objects: List[str] = field(default_factory=list)
    # (object, target, kind) kind ∈ {'region','object'}
    place_on_targets: List[Tuple[str, str, str]] = field(default_factory=list)
    # (object, container_region)
    place_in_targets: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.task.suite}/{self.task.name}"

    def location_atoms(self, obj: str) -> frozenset:
        """obj 可能出现的所有位置原子（作为 pick 的 del）。"""
        out = set()
        for a in self.atoms:
            if a.startswith(f"on_region({obj},") or a.startswith(f"on_object({obj},") \
                    or a.startswith(f"in_region({obj},"):
                out.add(a)
        return frozenset(out)


def build_problem(task: Task) -> Problem:
    init = set()
    goal = set()
    for fact in task.initial_state:
        atom, static = _translate_fact(task, fact)
        if atom is not None and not static:
            init.add(atom)
    for fact in task.goal_state:
        atom, static = _translate_fact(task, fact)
        if atom is not None and not static:
            goal.add(atom)

    # 关节默认状态（LIBERO init 不显式列闭态/关态）。
    for e in open_close_entities(task):
        if f"open({e})" not in init and f"closed({e})" not in init:
            init.add(f"closed({e})")
    for e in turnable_entities(task):
        if f"turned_on({e})" not in init and f"turned_off({e})" not in init:
            init.add(f"turned_off({e})")

    # 手爪初始为空。
    init.add("empty()")

    # 待移动物体：goal 位置事实中的第一参数。
    pick_objects = []
    for fact in task.goal_state:
        if fact[0] in ("on", "in") and len(fact) >= 2 and not _is_fixture(task, fact[1]):
            if fact[1] not in pick_objects:
                pick_objects.append(fact[1])

    place_on_targets: List[Tuple[str, str, str]] = []
    place_in_targets: List[Tuple[str, str]] = []
    for fact in task.goal_state:
        if fact[0] == "on" and len(fact) == 3:
            obj, y = fact[1], fact[2]
            if _is_fixture(task, obj):
                continue
            kind = "region" if y in task.regions else "object"
            place_on_targets.append((obj, y, kind))
        elif fact[0] == "in" and len(fact) == 3:
            place_in_targets.append((fact[1], fact[2]))

    # 原子词汇：init ∪ goal ∪ 手爪原子。
    atoms = set(init) | set(goal)
    atoms.add("empty()")
    for o in task.object_instances:
        atoms.add(f"holding({o})")

    return Problem(
        task=task,
        init=frozenset(init),
        goal=frozenset(goal),
        atoms=frozenset(atoms),
        surface_regions=surface_regions(task),
        drawer_regions=drawer_regions(task),
        open_close_entities=open_close_entities(task),
        turnable_entities=turnable_entities(task),
        pick_objects=pick_objects,
        place_on_targets=place_on_targets,
        place_in_targets=place_in_targets,
    )
