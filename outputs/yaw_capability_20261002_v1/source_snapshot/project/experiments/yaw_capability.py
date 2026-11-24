"""Minimal yaw capability: keep tool down, rotate gripper closing direction about world Z.

Open-scene compatibility screen only. No channel walls, no side grasp, no model calls.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import functools
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import mujoco
import numpy as np

from blocks_p0 import ROOT, BlocksExperiment, digest, execute_trial, make_config, save
from run_blocks_p0 import assert_sources, snapshot_sources, source_files

PROTOCOL = Path(__file__).with_name("yaw_capability_protocol.json")


def yaw_down(degrees):
    """Tool +z stays down, closing direction rotates by `degrees` about world Z (wxyz)."""
    half = np.deg2rad(float(degrees)) / 2
    return np.array([0.0, np.cos(half), np.sin(half), 0.0])


def quat_rotation_error(target_wxyz, actual_wxyz):
    err = np.zeros(3)
    mujoco.mju_subQuat(err, np.asarray(target_wxyz, float), np.asarray(actual_wxyz, float))
    return float(np.linalg.norm(err))


class YawExperiment(BlocksExperiment):
    """BlocksExperiment with an optional absolute end-effector target orientation."""

    def __init__(self, config, task, protocol, render=False, yaw_deg=None):
        super().__init__(config, task, protocol, render)
        self.yaw_deg = None if yaw_deg is None else float(yaw_deg)
        if self.yaw_deg is not None:
            self.r.target_quat = yaw_down(self.yaw_deg)

    def snapshot(self):
        snap = super().snapshot()
        _, quat = self.r.arm.get_ee_pose()
        snap["tcp_quat_wxyz"] = [float(v) for v in quat]
        snap["target_quat_wxyz"] = [float(v) for v in
                                    (self.r.target_quat if self.r.target_quat is not None else self.r.arm.grasp_quat)]
        return snap


def validate(p):
    if p["scope"] != "open_scene_compatibility_screen_not_channel_execution":
        raise ValueError("Only the open-scene compatibility screen is supported")
    if p["require_clear"] is not True or p["grasp_offset_z_m"] != 0.0:
        raise ValueError("Compatibility screen uses the fixed J candidate (clear(support), offset=0)")
    ids = [v["id"] for v in p["variants"]]
    if ids != ["legacy_0", "default", "yaw_0", "yaw_90"] or len(ids) != len(set(ids)):
        raise ValueError("Four registered variants required in fixed order")
    yaws = [v["yaw_deg"] for v in p["variants"]]
    if yaws != [None, None, 0.0, 90.0]:
        raise ValueError("Variant yaw list must be [null, null, 0, 90]")
    if p["max_actions"] != 16 or p["stable_seconds"] != 1.0:
        raise ValueError("Frozen execution budget")


def run(protocol_path, output):
    p = json.loads(Path(protocol_path).read_text())
    validate(p)
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Output outside workspace")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    paths = source_files(protocol_path, p)
    hashes = snapshot_sources(paths, output)
    save(output / "protocol.json", p)
    save(output / "manifest.json", {"created_at": datetime.now(timezone.utc).isoformat(),
                                    "protocol_sha256": digest(p), "source_hashes": hashes,
                                    "python": sys.version, "argv": sys.argv})
    layout = p["layout"]
    config = make_config(p, layout, occupied=False)
    task = {key: deepcopy(layout[key]) for key in ("payload", "support", "distractor", "buffer_xy")}
    variants, summary_rows = [], []
    for variant in p["variants"]:
        assert_sources(hashes)
        yaw = variant["yaw_deg"]
        experiment_type = functools.partial(YawExperiment, yaw_deg=yaw)
        result = execute_trial(p, config, task, p["require_clear"], p["grasp_offset_z_m"],
                               output / variant["id"], experiment_type=experiment_type)
        final_quat = (result.get("final") or {}).get("tcp_quat_wxyz")
        target_quat = (result.get("initial") or {}).get("target_quat_wxyz")
        grasp_errors = []
        for step in result.get("steps", []):
            if step.get("model", {}).get("name") in ("pick_table", "unstack"):
                after = step.get("after") or {}
                if after.get("tcp_quat_wxyz") and target_quat:
                    grasp_errors.append(quat_rotation_error(target_quat, after["tcp_quat_wxyz"]))
        row = {"id": variant["id"], "yaw_deg": yaw, "success": result["success"],
               "trial_valid": result["trial_valid"], "exception": result["exception"],
               "target_quat_wxyz": target_quat,
               "grasp_phase_orientation_error_rad": max(grasp_errors) if grasp_errors else None,
               "final_tcp_quat_wxyz": final_quat,
               "home_orientation_vs_target_rad": quat_rotation_error(target_quat, final_quat) if (final_quat and target_quat) else None,
               "simulation_seconds": result.get("simulation_seconds"),
               "wall_seconds": result.get("wall_seconds")}
        variants.append(row)
        save(output / "progress.json", {"variants": variants, "status": "running"})
        print(f"{variant['id']}: success={result['success']} exception={result['exception']}", flush=True)
    assert_sources(hashes)
    by_id = {r["id"]: r for r in variants}
    compatibility = {
        "legacy_baseline_ok": by_id["legacy_0"]["success"] and by_id["legacy_0"]["trial_valid"],
        "new_default_preserves_legacy": by_id["default"]["success"] == by_id["legacy_0"]["success"]
                                        and by_id["default"]["trial_valid"] == by_id["legacy_0"]["trial_valid"],
        "explicit_yaw0_equivalent": by_id["yaw_0"]["success"] == by_id["legacy_0"]["success"],
        "explicit_yaw90_ok": by_id["yaw_90"]["success"] and by_id["yaw_90"]["trial_valid"],
        "all_trial_valid": all(r["trial_valid"] for r in variants),
        "yaw90_grasp_orientation_error_rad": by_id["yaw_90"].get("grasp_phase_orientation_error_rad"),
        "note": "grasp_phase orientation is enforced by RebuiltSingleArm.move (<0.03 rad) when target_quat is set; "
                "home() does not consume target_quat, so the final TCP returns to the default yaw=0 orientation.",
    }
    report = {"status": "complete", "compatibility": compatibility, "variants": variants,
              "decision": "minimal_yaw_compatible_in_open_scene" if (compatibility["legacy_baseline_ok"]
                          and compatibility["new_default_preserves_legacy"] and compatibility["explicit_yaw0_equivalent"]
                          and compatibility["explicit_yaw90_ok"] and compatibility["all_trial_valid"])
                          else "compatibility_gap",
              "total_wall_seconds": time.monotonic() - started,
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024),
              "limits": ["Open scene only; no channel walls, no side grasp, no path sweep.",
                         "Fixed J candidate (clear(support), offset=0) and one layout.",
                         "Four correlated trials, not independent generalization samples.",
                         "Positive screen does not prove channel success or a stable yaw-aware grasp."]}
    save(output / "results.json", report)
    print(json.dumps(compatibility, indent=2), flush=True)
    return 0 if report["decision"] == "minimal_yaw_compatible_in_open_scene" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(args.protocol.resolve(), args.output)


if __name__ == "__main__":
    raise SystemExit(main())
