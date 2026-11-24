"""Static yaw/clearance screening on scratch MuJoCo data; no robot rollout."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import inspect
import itertools
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from blocks_p0 import ROOT, digest, save
from contract_tradeoff import plain
from core.cover_cabto_api import API_DOCS, FineCover
from core.single_arm_rebuilt import RebuiltSingleArm
from exp4_env import ArmInterface
from run_blocks_p0 import assert_sources, snapshot_sources, source_files

PROTOCOL = Path(__file__).with_name("approach_interface_protocol.json")


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(p):
    if p["scenes"] != ["open", "rails_x", "rails_y"] or p["yaw_degrees"] != [0., 90.]:
        raise ValueError("Only the six registered endpoint queries are supported")
    if p["max_pose_queries"] != 6 or p["dynamic_steps"] != 0 or p["model_calls"] != 0 or p["render"]:
        raise ValueError("No stepping, rendering or models in this audit")
    if p["ik"] != {"local_only": True, "iters": 60, "w_rot": 1.}:
        raise ValueError("IK settings must be frozen")
    if p["channel"] != {"inner_width_m": .08, "wall_thickness_m": .01, "wall_length_m": .12,
                        "wall_height_m": .06, "grasp_offset_z_m": .004}:
        raise ValueError("No adaptive channel tuning")


def yaw_down(degrees):
    half = np.deg2rad(degrees) / 2
    return np.array([0., np.cos(half), np.sin(half), 0.])


def add_channel(xml, name, p, xy, table_z):
    if name not in p["scenes"]:
        raise ValueError("Unregistered scene")
    root = ET.fromstring(xml)
    if name != "open":
        c = p["channel"]
        axis = 0 if name == "rails_x" else 1
        for sign in (-1, 1):
            center = [float(xy[0]), float(xy[1]), table_z + c["wall_height_m"] / 2]
            center[axis] += sign * (c["inner_width_m"] + c["wall_thickness_m"]) / 2
            half = [c["wall_length_m"] / 2, c["wall_length_m"] / 2, c["wall_height_m"] / 2]
            half[axis] = c["wall_thickness_m"] / 2
            body = ET.SubElement(root.find("worldbody"), "body", name=f"audit_rail_{sign:+d}", pos=" ".join(map(str, center)))
            ET.SubElement(body, "geom", name=f"audit_rail_geom_{sign:+d}", type="box", size=" ".join(map(str, half)),
                          rgba="0.3 0.4 0.5 1", contype="1", conaffinity="1", friction="1 .02 .001")
    xml_new = ET.tostring(root, encoding="unicode")
    stripped = ET.fromstring(xml_new)
    for body in list(stripped.find("worldbody")):
        if body.get("name", "").startswith("audit_rail_"):
            stripped.find("worldbody").remove(body)
    if ET.tostring(stripped) != ET.tostring(ET.fromstring(xml)):
        raise ValueError("Channel builder changed the original scene")
    return xml_new


def state_fingerprint(data):
    return digest({k: plain(getattr(data, k)) for k in
                   ("time", "qpos", "qvel", "act", "ctrl", "eq_active", "qfrc_applied", "xfrc_applied")})


def archived_array(value, target):
    values = np.asarray(value, dtype=target.dtype)
    if values.size == 0 and target.size == 0:
        return values.reshape(target.shape)
    return values


def restore_recorded_state(model, state):
    data = mujoco.MjData(model)
    for name, value in state.items():
        if name == "time":
            data.time = float(value)
        else:
            target = getattr(data, name)
            values = archived_array(value, target)
            if name == "xfrc_applied" and values.shape != target.shape:
                added = [body_name(model, i) for i in range(len(values), model.nbody)]
                if values.ndim != 2 or values.shape[1:] != (6,) or not added or not all(n.startswith("audit_rail_") for n in added):
                    raise ValueError("Unexpected body change in force-array restoration")
                extended = np.zeros_like(target)
                extended[:len(values)] = values
                values = extended
            if target.shape != values.shape:
                raise ValueError(f"Archived state shape mismatch: {name}")
            target[:] = values
    mujoco.mj_forward(model, data)
    return data


def body_name(model, bid):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid)) or "world"


def geom_name(model, gid):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f"geom_{gid}"


def geometry_check(model, data, p):
    active = [i for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]]
    robot_bodies = {i for i in range(model.nbody) if body_name(model, i).startswith("link")
                    or body_name(model, i) in ("hand", "left_finger", "right_finger")}
    rails = [i for i in active if body_name(model, model.geom_bodyid[i]).startswith("audit_rail_")]
    robot = [i for i in active if model.geom_bodyid[i] in robot_bodies]
    objects = [i for i in active if body_name(model, model.geom_bodyid[i]) in ("red", "green", "yellow", "blue")]
    environment = [i for i in active if i not in robot and i not in rails]
    cap, margin = p["screen"]["distance_cap_m"], p["screen"]["obstacle_clearance_m"]

    def distances(left, right, allow_base=False):
        found = []
        for a, b in itertools.product(left, right):
            ba, bb = body_name(model, model.geom_bodyid[a]), body_name(model, model.geom_bodyid[b])
            if allow_base and ba == "link0" and bb in ("table", "pedestal"):
                continue
            if not ((model.geom_contype[a] & model.geom_conaffinity[b]) or (model.geom_contype[b] & model.geom_conaffinity[a])):
                continue
            distance = float(mujoco.mj_geomDistance(model, data, a, b, cap, None))
            found.append({"geoms": [geom_name(model, a), geom_name(model, b)], "bodies": [ba, bb], "distance_m": distance})
        return {"pair_count": len(found), "min_distance_m": min((x["distance_m"] for x in found), default=None),
                "pairs_below_clearance": [x for x in found if x["distance_m"] < margin]}

    rr = distances(robot, rails)
    ob = distances(objects, rails)
    re = distances(robot, environment, allow_base=True)
    self_contacts = []
    for contact in data.contact:
        a, b = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
        if a not in robot_bodies or b not in robot_bodies or a == b:
            continue
        if model.body_parentid[a] == b or model.body_parentid[b] == a:
            continue
        names = {body_name(model, a), body_name(model, b)}
        if names in ({"link5", "link7"}, {"link5", "hand"}):
            continue
        if contact.dist < p["screen"]["deep_contact_m"]:
            self_contacts.append({"bodies": sorted(names), "distance_m": float(contact.dist)})
    return {"robot_to_rails": rr, "objects_to_rails": ob, "robot_to_environment": re,
            "deep_self_contacts": self_contacts,
            "obstacle_clearance_ok": not rr["pairs_below_clearance"] and not ob["pairs_below_clearance"],
            "other_endpoint_contact_ok": not self_contacts and not any(x["distance_m"] < p["screen"]["deep_contact_m"] for x in re["pairs_below_clearance"]),
            "scope": "collision-enabled robot-environment/obstacle geoms plus eligible MuJoCo self contacts; not swept-path clearance"}


def inspect_interfaces():
    symbols = (FineCover.approach, FineCover.descend, FineCover.lift, RebuiltSingleArm.move,
               RebuiltSingleArm.move_object, ArmInterface.solve_ik)
    records = []
    for symbol in symbols:
        records.append({"symbol": symbol.__qualname__, "signature": str(inspect.signature(symbol)),
                        "source": str(Path(inspect.getsourcefile(symbol)).resolve().relative_to(ROOT)),
                        "start_line": inspect.getsourcelines(symbol)[1]})
    return {"symbols": records, "pick_api": API_DOCS["pick"],
            "pose_argument_in_current_move": "target_quat" in inspect.signature(RebuiltSingleArm.move).parameters,
            "yaw_argument_in_public_pick_methods": any("yaw" in inspect.signature(getattr(FineCover, name)).parameters for name in API_DOCS["pick"]),
            "ik_accepts_quaternion": "target_quat" in inspect.signature(ArmInterface.solve_ik).parameters,
            "interpretation": "A pose-aware IK backend is not a pose-aware executable task API."}


def resume_open_checks(path, p, input_hashes, seed):
    previous, manifest = load(path / "results.json"), load(path / "manifest.json")
    if load(path / "protocol.json") != p or manifest["archived_input_hashes"] != input_hashes or manifest["seed"] != seed:
        raise ValueError("Resume protocol, archived inputs or seed changed")
    if previous["error"] != "ValueError: Archived state shape mismatch: xfrc_applied":
        raise ValueError("Only the documented force-array infrastructure failure may resume")
    rows, scenes = previous["pose_checks"], previous["scene_checks"]
    if [(r["scene"], r["yaw_deg"]) for r in rows] != [("open", 0.), ("open", 90.)] or len(scenes) != 1:
        raise ValueError("Expected two completed open-scene endpoints")
    if previous["forbidden_calls"] or not all(r["reference_state_unchanged"] for r in rows):
        raise ValueError("Cannot reuse invalid static queries")
    if sha(path / "open.xml") != scenes[0]["scene_sha256"]:
        raise ValueError("Prior open-scene geometry changed")
    for name, expected in manifest["source_hashes"].items():
        if sha(path / "source_snapshot" / name) != expected:
            raise ValueError("Prior source snapshot changed")
    return rows, scenes, {"path": str(path), "reused_queries": 2, "results_sha256": sha(path / "results.json"),
                         "manifest_sha256": sha(path / "manifest.json"), "prior_wall_seconds": previous["total_wall_seconds"]}


def run(protocol_path, output, resume=None):
    p = load(protocol_path)
    validate(p)
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Output outside workspace")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source = ROOT / p["source_trial"]
    archived = load(source / "result.json")
    if not archived["success"] or not archived["trial_valid"]:
        raise ValueError("Registered seed trial must be successful")
    trajectory = load(source / "trajectory.json")
    seed_row = next(r for r in trajectory if r["phase"] == p["seed_phase"])
    physics = archived["initial"]["execution_state"]["physics"]
    original_xml = (source / "scene.xml").read_text()
    source_paths = [source / name for name in ("scene.xml", "result.json", "trajectory.json")]
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    snapshots = snapshot_sources(source_files(protocol_path, p), output)
    save(output / "protocol.json", p)
    save(output / "manifest.json", {"created_at": datetime.now(timezone.utc).isoformat(), "protocol_sha256": digest(p),
                                   "source_hashes": snapshots, "archived_input_hashes": source_hashes,
                                   "seed": seed_row, "seed_is_initialization_not_new_execution": True,
                                   "python": sys.version, "mujoco": mujoco.__version__})
    save(output / "interfaces.json", inspect_interfaces())
    rows, scenes, reused = [], [], None
    if resume is not None:
        rows, scenes, reused = resume_open_checks(resume.resolve(), p, source_hashes, seed_row)
        (output / "open.xml").write_bytes((resume / "open.xml").read_bytes())
        save(output / "resume.json", reused)
        save(output / "pose_checks.json", rows)
        save(output / "scene_checks.json", scenes)
    status, error = "complete", None
    forbidden = []

    def no_dynamics(*args, **kwargs):
        forbidden.append("dynamic_or_render_call")
        raise RuntimeError("Static screening forbids dynamics and renderer")

    try:
        with ExitStack() as stack:
            for name in ("mj_step", "mj_step1", "mj_step2", "Renderer"):
                stack.enter_context(patch.object(mujoco, name, side_effect=no_dynamics))
            for scene in p["scenes"]:
                if reused is not None and scene == "open":
                    continue
                assert_sources(snapshots)
                if any(sha(ROOT / name) != value for name, value in source_hashes.items()):
                    raise ValueError("Archived input changed")
                obj = archived["initial"]["objects"][p["payload"]]["position"]
                table_z = load(ROOT / p["base_config"])["table_top"]
                xml = add_channel(original_xml, scene, p, obj[:2], table_z)
                (output / f"{scene}.xml").write_text(xml)
                model = mujoco.MjModel.from_xml_string(xml)
                data = restore_recorded_state(model, physics)
                baseline = state_fingerprint(data)
                environment = SimpleNamespace(m=model, d=data, ik_data=mujoco.MjData(model))
                arm = ArmInterface(environment, "", 0., [])
                arm._grasp_enabled = False
                initial_geometry = geometry_check(model, data, p)
                valid_initial = initial_geometry["obstacle_clearance_ok"]
                scenes.append({"scene": scene, "scene_sha256": hashlib.sha256(xml.encode()).hexdigest(),
                               "initial_state_sha256": baseline, "valid_initial_obstacles": valid_initial,
                               "initial_geometry": initial_geometry, "object_state_source": "archived settled initial state"})
                save(output / "scene_checks.json", scenes)
                if not valid_initial:
                    status = "invalid_scene_stop"
                    break
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, p["payload"])
                target = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ np.array([0., 0., p["channel"]["grasp_offset_z_m"]])
                for yaw in p["yaw_degrees"]:
                    if len(rows) >= p["max_pose_queries"]:
                        raise ValueError("Pose-query budget exhausted")
                    arm.arm_target = np.asarray(seed_row["q"], float).copy()
                    quat = yaw_down(yaw)
                    q = arm.solve_ik(target, quat, **p["ik"])
                    if state_fingerprint(data) != baseline:
                        raise ValueError("IK changed reference state")
                    fk = restore_recorded_state(model, physics)
                    fk.qpos[arm.arm_qadr] = q
                    mujoco.mj_forward(model, fk)
                    actual = np.zeros(4)
                    mujoco.mju_mat2Quat(actual, fk.site_xmat[arm.tcp_sid])
                    rot_error = np.zeros(3)
                    mujoco.mju_subQuat(rot_error, quat, actual)
                    pe = float(np.linalg.norm(fk.site_xpos[arm.tcp_sid] - target))
                    re = float(np.linalg.norm(rot_error))
                    limits = bool(np.isfinite(q).all() and np.all(q >= arm._jnt_lo - p["screen"]["joint_limit_tolerance_rad"])
                                  and np.all(q <= arm._jnt_hi + p["screen"]["joint_limit_tolerance_rad"]))
                    ik_ok = limits and pe <= p["screen"]["position_tolerance_m"] and re <= p["screen"]["rotation_tolerance_rad"]
                    geometry = geometry_check(model, fk, p)
                    row = {"scene": scene, "yaw_deg": yaw, "target_xyz": target.tolist(), "target_quat_wxyz": quat.tolist(),
                           "joint_seed": list(seed_row["q"]), "joint_solution": q.tolist(), "position_error_m": pe,
                           "rotation_error_rad": re, "joint_limits_ok": limits, "ik_endpoint_ok": bool(ik_ok),
                           "actual_tool_z_world": fk.site_xmat[arm.tcp_sid].reshape(3, 3)[:, 2].tolist(),
                           "geometry": geometry, "static_endpoint_pass": bool(ik_ok and geometry["obstacle_clearance_ok"] and geometry["other_endpoint_contact_ok"]),
                           "scratch_time": float(fk.time), "reference_state_unchanged": state_fingerprint(data) == baseline,
                           "grasp_executed": False}
                    if fk.time != data.time or np.any(fk.eq_active != np.asarray(physics["eq_active"])):
                        raise ValueError("Time or equality state changed")
                    rows.append(row)
                    save(output / "pose_checks.json", rows)
                    print(f"{scene}/yaw{yaw:g}: IK={ik_ok} rail_clear={geometry['obstacle_clearance_ok']} endpoint={row['static_endpoint_pass']} pos_error={pe:.6f}m rot_error={re:.6f}rad", flush=True)
                del arm, environment, fk, data, model
        assert_sources(snapshots)
    except Exception as exc:
        status, error = "invalid_audit", f"{type(exc).__name__}: {exc}"
    report = {"status": status, "error": error, "interfaces": inspect_interfaces(), "pose_checks": rows,
              "scene_checks": scenes, "completed_pose_queries": len(rows), "dynamic_steps": 0,
              "reused_queries": 2 if reused else 0, "new_queries": len(rows) - (2 if reused else 0),
              "resume_provenance": reused,
              "forbidden_calls": forbidden, "model_calls": 0, "executed_grasps": 0,
              "full_pose_side_grasp_supported": False, "total_wall_seconds": time.monotonic() - started,
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024),
              "decision": "endpoint_evidence_only_pose_api_and_continuous_execution_still_unimplemented",
              "limits": ["Six fixed grasp endpoints, not swept paths or pregrasp trajectories.",
                         "Local IK failure is not physical impossibility; success is not grasp success.",
                         "Designed channels and archived same-object seed are development conditions.",
                         "No training, controller change, force-closure or dynamic safety claim."]}
    save(output / "results.json", report)
    print(json.dumps({k: report[k] for k in ("status", "error", "completed_pose_queries", "dynamic_steps", "executed_grasps", "total_wall_seconds", "peak_rss_mib")}, indent=2), flush=True)
    return 1 if status == "invalid_audit" else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-open", type=Path)
    args = parser.parse_args()
    return run(args.protocol.resolve(), args.output, args.resume_open)


if __name__ == "__main__":
    raise SystemExit(main())
