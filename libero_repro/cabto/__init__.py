"""CABTO on LIBERO single-arm benchmark.

把 CABTO 的单臂技术路线（`outputs/remaining_cabto` 的 cover/blocks）接到 LIBERO
的单臂任务上。三层：

1. 符号层（本机可完整运行）：BDDL 解析 → 谓词规范化 → STRIPS 动作库 → 形式化 BT 规划
   → 符号可达性验证（复用 CABTO 的 `formal_bt` 内核）。
2. 代码生成层（本机可运行）：细粒度运动 API 契约 + `def policy(api)` 安全解释器
   （复用 `policy_codegen`）。
3. 执行/接地层（需 LIBERO 官方 Linux/GPU 栈，或 CABTO 的 MuJoCo 3.10 单臂场景 +
   LIBERO 资产）：提供统一 Runtime 接口与效果校验器，文档化物理执行边界。

见 `libero_repro/README_cabto.md`。
"""
from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
LIBERO_REPRO_DIR = PACKAGE_DIR.parent

# 已提交的自包含 LIBERO 产物：cabto 运行只依赖这些小型文件（130 个 bddl + 解析器 + task map），
# 不需要 650M 的 LIBERO 完整克隆（该克隆不入库）。见 vendor_libero/ 与 bddl_files/。
VENDOR_DIR = LIBERO_REPRO_DIR / "vendor_libero"
BDDL_DIR = LIBERO_REPRO_DIR / "bddl_files"

# 本地若存在 LIBERO 完整克隆，则作为 fallback（用于需要 robosuite/mujoco 的完整环境）。
LIBERO_ROOT = LIBERO_REPRO_DIR / "LIBERO"
_LIBERO_BDDL_DIR = LIBERO_ROOT / "libero" / "libero" / "bddl_files"

# CABTO 上游工作区（本仓库的上级目录），复用其形式化 BT 内核与代码生成解释器。
CABTO_WORKSPACE = PACKAGE_DIR.parents[2]  # /Users/yishuaicai/mywork/edps_cabto_exp
STAGE3_CABTO = CABTO_WORKSPACE / "CABTO" / "exp4_bt_tasks" / "stage3_cabto"
BT_ROOT = CABTO_WORKSPACE / "BTExpansion-demo"

SUITES = ["libero_object", "libero_spatial", "libero_goal", "libero_10", "libero_90"]


def _resolve(*candidates: Path) -> Path:
    """返回第一个存在的候选路径，都不存在则返回第一个。"""
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]


def resolve_bddl_dir() -> Path:
    """优先用已提交的 bddl_files，缺失时回退到 LIBERO 完整克隆。"""
    if BDDL_DIR.is_dir():
        return BDDL_DIR
    return _LIBERO_BDDL_DIR


__all__ = [
    "PACKAGE_DIR",
    "LIBERO_REPRO_DIR",
    "LIBERO_ROOT",
    "BDDL_DIR",
    "VENDOR_DIR",
    "CABTO_WORKSPACE",
    "STAGE3_CABTO",
    "BT_ROOT",
    "SUITES",
    "resolve_bddl_dir",
]
