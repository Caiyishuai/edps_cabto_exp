"""代码生成契约：每个技能的细粒度运动 API + `def policy(api)` 安全解释器。

复用 CABTO 的 `policy_codegen.PolicyProgram`（一个受限 AST 解释器，禁止 import/循环/
tuple 解包/模拟器访问/成功标志）。生成的策略是"一个动作"的直线式 API 调用序列，绝不含
整任务包装。

本模块提供：
- `API_DOCS`：每个 program_kind 的细粒度运动 API 文档（与 cover_cabto_api 同风格）。
- `REFERENCE_POLICIES`：工程指定的参考策略源码（离线确定性基线，明确非模型证据）。
- `validate_policy` / `build_program`：对候选源码做契约校验。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

from . import STAGE3_CABTO

if str(STAGE3_CABTO) not in sys.path:
    sys.path.insert(0, str(STAGE3_CABTO))

from policy_codegen import PolicyProgram  # noqa: E402

# 每个技能可调用的运动 API 契约（key=program_kind，value={方法名: 说明}）。
API_DOCS: Dict[str, Dict[str, str]] = {
    "pick": {
        "grasp_point": "api.grasp_point() -> (x,y,z). Read-only grasp-center estimate in world metres. Backend selected externally (oracle or VLM). No movement.",
        "open_gripper": "api.open_gripper() -> None. Command gripper open and wait; does NOT move arm.",
        "approach": "api.approach(point) -> None. Move OPEN end effector above the point at safe transport height. One xyz tuple. Does NOT descend or close.",
        "descend": "api.descend(point) -> None. Move OPEN end effector vertically to this grasp xyz. Requires already above it. Does NOT close or lift.",
        "close_gripper": "api.close_gripper() -> None. Close at CURRENT location and wait. Does NOT approach, descend or lift.",
        "lift": "api.lift() -> None. Raise CLOSED end effector vertically to transport height. Does NOT close.",
    },
    "place_on": {
        "destination_point": "api.destination_point() -> (x,y,z). Read-only desired PAYLOAD ORIGIN at its target surface/object. No movement.",
        "carry": "api.carry(point) -> None. Move CLOSED gripper carrying payload horizontally above target; does NOT lower, open or retreat. One xyz tuple.",
        "lower": "api.lower(point) -> None. Lower CLOSED gripper so payload origin reaches target xyz. Does NOT release.",
        "open_gripper": "api.open_gripper() -> None. Open fingers and settle at CURRENT location. Does NOT retreat.",
        "withdraw": "api.withdraw() -> None. Move OPEN gripper vertically away to safe height and settle.",
    },
    "place_in": {
        "destination_point": "api.destination_point() -> (x,y,z). Read-only payload placement point inside the container. No movement.",
        "carry": "api.carry(point) -> None. Move CLOSED gripper above the container opening; does NOT lower, open or retreat.",
        "lower": "api.lower(point) -> None. Lower CLOSED gripper so payload reaches the container interior. Does NOT release.",
        "open_gripper": "api.open_gripper() -> None. Open fingers and settle at CURRENT location. Does NOT retreat.",
        "withdraw": "api.withdraw() -> None. Move OPEN gripper vertically away to safe height and settle.",
    },
    "open": {
        "handle_point": "api.handle_point() -> (x,y,z). Read-only grasp point on the drawer/door handle. No movement.",
        "approach": "api.approach(point) -> None. Move OPEN end effector to the handle at a safe offset. One xyz tuple.",
        "grasp_handle": "api.grasp_handle() -> None. Close fingers on the handle and confirm contact. Does NOT move arm.",
        "pull_open": "api.pull_open() -> None. Pull the handle along the opening direction until the articulation is open.",
        "release": "api.release() -> None. Open fingers to let go of the handle. Does NOT retreat.",
        "retreat": "api.retreat() -> None. Move OPEN end effector away to a safe posture and settle.",
    },
    "close": {
        "handle_point": "api.handle_point() -> (x,y,z). Read-only contact point on the drawer/door face. No movement.",
        "approach": "api.approach(point) -> None. Move OPEN end effector to the contact point at a safe offset.",
        "push_close": "api.push_close() -> None. Push the face along the closing direction until the articulation is closed.",
        "release": "api.release() -> None. Withdraw finger contact pressure. Does NOT retreat.",
        "retreat": "api.retreat() -> None. Move OPEN end effector away to a safe posture and settle.",
    },
    "turn_on": {
        "knob_point": "api.knob_point() -> (x,y,z). Read-only grasp point on the knob/switch. No movement.",
        "approach": "api.approach(point) -> None. Move OPEN end effector to the knob at a safe offset.",
        "grasp_knob": "api.grasp_knob() -> None. Close fingers on the knob and confirm contact. Does NOT move arm.",
        "rotate_on": "api.rotate_on() -> None. Rotate the knob in the ON direction until the appliance turns on.",
        "release": "api.release() -> None. Open fingers to let go of the knob. Does NOT retreat.",
        "retreat": "api.retreat() -> None. Move OPEN end effector away to a safe posture and settle.",
    },
    "turn_off": {
        "knob_point": "api.knob_point() -> (x,y,z). Read-only grasp point on the knob/switch. No movement.",
        "approach": "api.approach(point) -> None. Move OPEN end effector to the knob at a safe offset.",
        "grasp_knob": "api.grasp_knob() -> None. Close fingers on the knob and confirm contact. Does NOT move arm.",
        "rotate_off": "api.rotate_off() -> None. Rotate the knob in the OFF direction until the appliance turns off.",
        "release": "api.release() -> None. Open fingers to let go of the knob. Does NOT retreat.",
        "retreat": "api.retreat() -> None. Move OPEN end effector away to a safe posture and settle.",
    },
}

# 工程参考策略源码（离线确定性基线）。真实模型生成的源码见 pipeline 的 sample 阶段。
REFERENCE_POLICIES: Dict[str, str] = {
    "pick": "def policy(api):\n"
            "    p = api.grasp_point()\n"
            "    api.open_gripper()\n"
            "    api.approach(p)\n"
            "    api.descend(p)\n"
            "    api.close_gripper()\n"
            "    api.lift()\n",
    "place_on": "def policy(api):\n"
                "    p = api.destination_point()\n"
                "    api.carry(p)\n"
                "    api.lower(p)\n"
                "    api.open_gripper()\n"
                "    api.withdraw()\n",
    "place_in": "def policy(api):\n"
                "    p = api.destination_point()\n"
                "    api.carry(p)\n"
                "    api.lower(p)\n"
                "    api.open_gripper()\n"
                "    api.withdraw()\n",
    "open": "def policy(api):\n"
            "    p = api.handle_point()\n"
            "    api.approach(p)\n"
            "    api.grasp_handle()\n"
            "    api.pull_open()\n"
            "    api.release()\n"
            "    api.retreat()\n",
    "close": "def policy(api):\n"
             "    p = api.handle_point()\n"
             "    api.approach(p)\n"
             "    api.push_close()\n"
             "    api.release()\n"
             "    api.retreat()\n",
    "turn_on": "def policy(api):\n"
               "    p = api.knob_point()\n"
               "    api.approach(p)\n"
               "    api.grasp_knob()\n"
               "    api.rotate_on()\n"
               "    api.release()\n"
               "    api.retreat()\n",
    "turn_off": "def policy(api):\n"
                "    p = api.knob_point()\n"
                "    api.approach(p)\n"
                "    api.grasp_knob()\n"
                "    api.rotate_off()\n"
                "    api.release()\n"
                "    api.retreat()\n",
}


def api_names(program_kind: str):
    return frozenset(API_DOCS[program_kind].keys())


def build_program(source: str, program_kind: str) -> PolicyProgram:
    return PolicyProgram(source, api_names(program_kind))


def validate_policy(source: str, program_kind: str) -> Dict:
    """对候选源码做纯契约校验（不执行运动）。返回结构化结果。"""
    try:
        program = build_program(source, program_kind)
        return {"valid": True, "program_kind": program_kind, "source": source}
    except Exception as exc:
        return {"valid": False, "program_kind": program_kind,
                "error": f"{type(exc).__name__}: {exc}", "source": source}


def reference_policy(program_kind: str) -> str:
    return REFERENCE_POLICIES[program_kind]
