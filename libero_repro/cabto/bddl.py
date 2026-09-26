"""纯 Python 的 LIBERO BDDL 解析与任务枚举。

LIBERO 官方环境（robosuite 1.4 + MuJoCo 2.3）在 macOS 26 上无法 import（见 libero_repro/README.md），
但 BDDL 任务规格的解析只依赖 `bddl` 包（纯 Python），与 MuJoCo 无关。本模块通过
`importlib` 单独加载 `libero/envs/bddl_utils.py`，避开 `libero.envs.__init__` 对 robosuite/mujoco 的重导入，
从而在本机拿到完整的 init/goal/objects/regions 规格。

解析产物与 LIBERO 官方 `robosuite_parse_problem` 完全一致：
- objects / fixtures：`{类别: [实例名, ...]}`
- obj_of_interest：目标物体实例名列表
- initial_state / goal_state：`[谓词, arg1, arg2?]` 三元组列表（谓词已小写）
- regions：`{<target>_<区域名>: {target, ranges, ...}}`
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from . import BDDL_DIR, LIBERO_ROOT, SUITES, VENDOR_DIR, resolve_bddl_dir

# 官方 suite -> 任务文件名（不含 .bddl 后缀）映射，优先于目录遍历以保证顺序稳定。
# 优先用已提交的 vendor 副本，缺失时回退到 LIBERO 完整克隆。
_TASK_MAP_FILE = VENDOR_DIR / "libero_suite_task_map.py"
_LIBERO_TASK_MAP_FILE = LIBERO_ROOT / "libero" / "libero" / "benchmark" / "libero_suite_task_map.py"
_task_map: Optional[Dict[str, List[str]]] = None

_bddl_utils = None


def load_bddl_utils():
    """单独加载 bddl_utils.py（vendor 副本优先，回退 LIBERO 克隆），避免导入 robosuite/mujoco。"""
    global _bddl_utils
    if _bddl_utils is not None:
        return _bddl_utils
    vendor = VENDOR_DIR / "bddl_utils.py"
    path = vendor if vendor.is_file() else (LIBERO_ROOT / "libero" / "libero" / "envs" / "bddl_utils.py")
    spec = importlib.util.spec_from_file_location("libero_bddl_utils", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["libero_bddl_utils"] = module
    spec.loader.exec_module(module)
    _bddl_utils = module
    return module


def _read_task_map() -> Dict[str, List[str]]:
    """读取官方 libero_suite_task_map.py，得到每个 suite 的稳定任务顺序。"""
    global _task_map
    if _task_map is not None:
        return _task_map
    mapping: Dict[str, List[str]] = {}
    namespace: Dict = {}
    task_map_file = _TASK_MAP_FILE if _TASK_MAP_FILE.is_file() else _LIBERO_TASK_MAP_FILE
    exec(task_map_file.read_text(), namespace)
    for suite in SUITES:
        if suite in namespace.get("libero_task_map", {}):
            mapping[suite] = list(namespace["libero_task_map"][suite])
    _task_map = mapping
    return mapping


def list_tasks(suite: Optional[str] = None) -> Dict[str, List[str]]:
    """返回 {suite: [任务名, ...]}；task 名不含 .bddl 后缀。"""
    mapping = _read_task_map()
    if suite is not None:
        if suite not in SUITES:
            raise ValueError(f"未知 suite: {suite}；可选 {SUITES}")
        return {suite: mapping.get(suite, [])}
    return {s: mapping.get(s, []) for s in SUITES}


def task_path(suite: str, task: str) -> Path:
    return resolve_bddl_dir() / suite / f"{task}.bddl"


@dataclass
class Task:
    """归一化后的单个 LIBERO 任务规格。"""

    suite: str
    name: str
    problem_name: str
    language: str
    objects: Dict[str, List[str]] = field(default_factory=dict)
    fixtures: Dict[str, List[str]] = field(default_factory=dict)
    obj_of_interest: List[str] = field(default_factory=list)
    initial_state: List[List[str]] = field(default_factory=list)
    goal_state: List[List[str]] = field(default_factory=list)
    regions: Dict[str, dict] = field(default_factory=dict)

    @property
    def object_instances(self) -> List[str]:
        return [i for vals in self.objects.values() for i in vals]

    @property
    def fixture_instances(self) -> List[str]:
        return [i for vals in self.fixtures.values() for i in vals]

    @property
    def all_instances(self) -> List[str]:
        return self.object_instances + self.fixture_instances

    def category_of(self, instance: str) -> Optional[str]:
        for cat, vals in {**self.fixtures, **self.objects}.items():
            if instance in vals:
                return cat
        return None

    @property
    def path(self) -> Path:
        return task_path(self.suite, self.name)


def parse_task(suite: str, task: str) -> Task:
    """解析一个 BDDL 文件，返回归一化 Task。"""
    path = task_path(suite, task)
    if not path.is_file():
        raise FileNotFoundError(f"BDDL 不存在: {path}")
    parsed = load_bddl_utils().robosuite_parse_problem(str(path))
    return Task(
        suite=suite,
        name=task,
        problem_name=parsed.get("problem_name", ""),
        language=" ".join(parsed.get("language_instruction", [])),
        objects=parsed.get("objects", {}),
        fixtures=parsed.get("fixtures", {}),
        obj_of_interest=parsed.get("obj_of_interest", []),
        initial_state=parsed.get("initial_state", []),
        goal_state=parsed.get("goal_state", []),
        regions=parsed.get("regions", {}),
    )


def load_suite(suite: str) -> List[Task]:
    return [parse_task(suite, task) for task in list_tasks(suite)[suite]]


def load_all() -> List[Task]:
    return [parse_task(s, t) for s in SUITES for t in list_tasks(s)[s]]
