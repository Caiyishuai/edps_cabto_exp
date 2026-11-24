"""Collect real, phase-synchronized single-pick intervention groups without training."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
from functools import wraps
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import uuid

import mujoco
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(ROOT / "CABTO/exp4_bt_tasks/stage3_cabto"))
from core.cover_cabto_api import API_DOCS, FineCover, transition_check
from core.remaining_blocks_api import FineBlocks
from policy_codegen import PolicyProgram
from formal_bt import validate_model
from blocks_p0 import digest, save


def file_hash(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def load_reference(protocol):
    path = ROOT / protocol["historical_policy_file"]
    entry = json.loads(path.read_text())["accepted"][protocol["historical_policy_action"]]
    actual = hashlib.sha256(entry["source"].encode()).hexdigest()
    if actual != entry["sha256"] or actual != protocol["historical_policy_sha256"]:
        raise ValueError("Historical generated source digest mismatch")
    PolicyProgram(entry["source"], API_DOCS["pick"])
    return entry


def derive_source(source, offset):
    if offset == 0:
        return source
    tree = ast.parse(source)
    body = tree.body[0].body
    assignments = [node for node in body if isinstance(node, ast.Assign)
                   and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                   and node.value.func.attr == "grasp_point"]
    if len(assignments) != 1 or not isinstance(assignments[0].targets[0], ast.Name):
        raise ValueError("Expected one bound grasp-point assignment")
    statement = assignments[0]
    name = statement.targets[0].id
    transformed = ast.parse(f"{name} = ({name}[0], {name}[1], {name}[2] + {float(offset)!r})").body[0]
    body.insert(body.index(statement) + 1, transformed)
    source = ast.unparse(ast.fix_missing_locations(tree))
    PolicyProgram(source, API_DOCS["pick"])
    return source


def model_for(obj, support, omission):
    model = {"name": "pick", "args": {"object": obj, "support": support},
             "pre": ["empty()", f"at_source({obj})"], "add": [f"holding({obj})"],
             "del": ["empty()", f"at_source({obj})", "home_completed()"]}
    if omission:
        model["del"].remove("empty()")
    validate_model(model)
    return model


def snapshot(fine):
    r, env = fine.r, fine.r.env
    names = list(env.obj_names) + [s for _, s in r.config["pairs"] if s not in env.obj_names]
    return {"sim_time": float(env.d.time), "state": sorted(fine.state()),
            "objects": {obj: {"position": r.pos(obj).tolist(), "quaternion": r.pose(obj)[1].tolist(),
                               "velocity": env.d.qvel[env.obj_dofadr[obj]:env.obj_dofadr[obj] + 6].tolist() if obj in env.obj_dofadr else [0.] * 6}
                        for obj in dict.fromkeys(names)},
            "tcp": r.ee().tolist(), "finger_forces_N": dict(r.force_now),
            "gripper_command": float(r.arm._grip_cmd), "gripper_width_m": float(r.arm.get_gripper_width()),
            "active_welds": r.active_welds()}


class PhaseRecorder:
    def __init__(self, fine, output, protocol):
        self.fine, self.output, self.protocol = fine, Path(output), protocol
        self.records = []
        self.renderer = None

    def capture(self, stage, status):
        fine, env = self.fine, self.fine.r.env
        record = {"index": len(self.records), "stage": stage, "status": status, "observation": snapshot(fine), "images": []}
        before_q = env.d.qpos.copy()
        before_control = env.d.ctrl.copy()
        before_time = float(env.d.time)
        if self.renderer is None:
            self.renderer = mujoco.Renderer(env.m, self.protocol["image_size"], self.protocol["image_size"])
        for camera in self.protocol["cameras"]:
            self.renderer.update_scene(env.d, camera=camera)
            image = np.asarray(self.renderer.render()).copy()
            if image.max() == image.min():
                raise RuntimeError("Constant/invalid rendered frame")
            ci = mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            path = self.output / f"frame{record['index']:02d}_{camera}.png"
            Image.fromarray(image).save(path)
            record["images"].append({"path": path.name, "sha256": file_hash(path), "camera": camera,
                                     "sim_time": before_time, "width": image.shape[1], "height": image.shape[0],
                                     "camera_position": env.d.cam_xpos[ci].tolist(),
                                     "camera_rotation": env.d.cam_xmat[ci].tolist(), "fovy_deg": float(env.m.cam_fovy[ci]),
                                     "optical_convention": "MuJoCo/OpenGL forward -Z, right +X, up +Y"})
        if env.d.time != before_time or not np.array_equal(before_q, env.d.qpos) or not np.array_equal(before_control, env.d.ctrl):
            raise RuntimeError("Observation changed physical state or controls")
        self.records.append(record)
        save(self.output / "frames.json", self.records)
        return record

    def wrap(self, mapping):
        result = {}
        for name, method in mapping.items():
            def bind_callback(name, method):
                @wraps(method)
                def callback(*args, **kwargs):
                    try:
                        value = method(*args, **kwargs)
                    except Exception:
                        self.capture(name, "failed")
                        raise
                    if name in self.protocol["capture_after_apis"]:
                        self.capture(name, "returned")
                    return value
                return callback
            result[name] = bind_callback(name, method)
        return result

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


def physical_pick(fine, obj, initial):
    r = fine.r
    metrics = r.assess()
    displacement = float(r.pos(obj)[2] - initial["objects"][obj]["position"][2])
    contacts = metrics["forbidden_contacts"]
    non_targets = {name: float(np.linalg.norm(r.pos(name) - value["position"]))
                   for name, value in initial["objects"].items() if name != obj}
    bilateral = min(r.force_now.values()) >= .05
    success = bool(r.held(obj) and displacement > .075 and bilateral and metrics["grasp_verified"]
                   and metrics["motion_verified"] and not contacts and not r.active_welds()
                   and max(non_targets.values(), default=0.) < .004)
    return {"success": success, "lift_delta_m": displacement, "bilateral_contact": bool(bilateral),
            "max_non_target_displacement_m": max(non_targets.values(), default=0.),
            "forbidden_contacts": contacts, "max_grasp_slip_m": r.max_grasp_slip,
            "max_contact_gap_seconds": r.max_contact_gap, "motion_verified": metrics["motion_verified"],
            "grasp_verified": metrics["grasp_verified"], "weld_violation": r.strict_weld_violation,
            "scope": "single_pick_not_place_release_or_home; fixed_existing_contact_tracking_thresholds"}


def collect_trial(scene_name, config, seed, model, source, output, protocol):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {"trial_id": uuid.uuid4().hex, "valid": False, "artifact_complete": True,
              "exception": None, "infrastructure_errors": [], "api_trace": [],
              "model": model, "source": source, "model_sha256": digest(model),
              "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "physical_pick_success": False,
              "contract_consistent": False, "joint_success": False, "scope": "single_pick_data_acceptance"}
    save(output / "result.json", {**result, "status": "started"})
    started = time.monotonic()
    fine, recorder = None, None
    try:
        cls = FineCover if scene_name == "cover" else FineBlocks
        fine = cls(deepcopy(config), seed=seed, render=False, perception="oracle", out=output)
        api = fine.bind(model)
        recorder = PhaseRecorder(fine, output, protocol)
        initial = recorder.capture("initial", "observed")["observation"]
        result["before"] = initial
        result["scene_sha256"] = hashlib.sha256(fine.r.xml.encode()).hexdigest()
        result["initial_qpos_sha256"] = hashlib.sha256(fine.r.env.d.qpos.tobytes()).hexdigest()
        (output / "scene.xml").write_text(fine.r.xml)
        (output / "policy.py").write_text(source)
        save(output / "model.json", model)
        if not set(model["pre"]) <= set(initial["state"]):
            raise ValueError("Initial physical precondition not satisfied")
        predicted = (set(initial["state"]) | set(model["add"])) - set(model["del"])
        result["static_mutex_warning"] = "empty()" in predicted and any(x.startswith("holding(") for x in predicted)
        program = PolicyProgram(source, API_DOCS["pick"])
        program.validate_api(api)
        result["valid"] = True
        try:
            result["api_trace"] = program.run(recorder.wrap(api), call_limit=protocol["max_api_calls"])["trace"]
        except RuntimeError as exc:
            result["exception"] = f"{type(exc).__name__}: {exc}"
            result["api_trace"] = getattr(exc, "policy_trace", [])
            if not result["api_trace"] or not recorder.records or recorder.records[-1]["status"] != "failed":
                raise
        terminal = recorder.capture("action_end", "observed")["observation"]
        result["action_end"] = terminal
        result["check_at_end"] = transition_check(model, initial["state"], terminal["state"], result["exception"])
        before_hold = physical_pick(fine, model["args"]["object"], initial)
        stable = before_hold["success"]
        if result["exception"] is None and stable:
            count = round(protocol["holding_window_seconds"] / (fine.r.dt * fine.r.steps))
            for _ in range(count):
                fine.r.wait(fine.r.dt * fine.r.steps, 0.)
                stable = stable and physical_pick(fine, model["args"]["object"], initial)["success"]
            recorder.capture("holding_window", "observed")
        result["after"] = snapshot(fine)
        result["physical_metrics"] = physical_pick(fine, model["args"]["object"], initial)
        result["physical_pick_success"] = bool(stable and result["exception"] is None and result["physical_metrics"]["success"])
        result["check"] = transition_check(model, initial["state"], result["after"]["state"], result["exception"])
        result["contract_consistent"] = bool(result["check"]["effect_ok"])
        result["joint_success"] = result["physical_pick_success"] and result["contract_consistent"]
        result["simulation_seconds"] = result["after"]["sim_time"] - initial["sim_time"]
    except Exception as exc:
        result["valid"] = False
        result["infrastructure_errors"].append(f"{type(exc).__name__}: {exc}")
        result["exception"] = result["exception"] or result["infrastructure_errors"][-1]
    finally:
        if recorder is not None:
            result["frames"] = recorder.records
            try:
                recorder.close()
            except Exception as exc:
                result["artifact_complete"] = False
                result["infrastructure_errors"].append(f"renderer_close: {exc}")
        if fine is not None:
            try:
                save(output / "events.json", fine.r.events)
                save(output / "tcp_joint_trajectory.json", fine.r.trace)
                save(output / "contact_summary.json", list(fine.r.contacts.values()))
                fine.close()
            except Exception as exc:
                result["artifact_complete"] = False
                result["infrastructure_errors"].append(f"finalize: {exc}")
        result["wall_seconds"] = time.monotonic() - started
        result["status"] = "complete" if result["valid"] else "invalid"
        save(output / "result.json", result)
    return result


def source_snapshot(output, protocol, protocol_path):
    paths = {Path(__file__).resolve(), Path(protocol_path).resolve(), ROOT / protocol["historical_policy_file"]}
    paths.update(ROOT / row["config"] for row in protocol["scenes"])
    paths.update(Path(__file__).parent.glob("*pick_groups*.py"))
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name and Path(name).resolve().is_relative_to(ROOT) and Path(name).suffix == ".py":
            paths.add(Path(name).resolve())
    hashes = {}
    for path in sorted(paths):
        relative = path.relative_to(ROOT)
        hashes[str(relative)] = file_hash(path)
        dst = output / "source_snapshot" / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(path.read_bytes())
        if file_hash(dst) != hashes[str(relative)]:
            raise ValueError("Source changed while snapshotting")
    assets = ROOT / "project/mujoco_workspace/franka_articulated/assets/panda_assets"
    asset_hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in assets.rglob("*") if p.is_file()}
    return hashes, asset_hashes


def collect(args):
    protocol = json.loads(args.protocol.read_text())
    reference = load_reference(protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    hashes, assets = source_snapshot(args.output, protocol, args.protocol)
    def unchanged():
        for name, sha in {**hashes, **assets}.items():
            if file_hash(ROOT / name) != sha:
                raise RuntimeError(f"Frozen source or asset changed: {name}")
    groups = []
    for scene in protocol["scenes"]:
        for context in protocol["contexts"]:
            context_id = uuid.uuid4().hex
            for obj, support in scene["objects"]:
                for condition in protocol["conditions"]:
                    groups.append({"id": uuid.uuid4().hex, "context_id": context_id, "scene": scene["name"],
                                   "config_path": scene["config"], "split": context["split"], "seed": context["seed"],
                                   "object": obj, "support": support, "condition": condition})
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
                "groups": groups, "source_hashes": hashes, "asset_hashes": assets,
                "reference_program": {"artifact": protocol["historical_policy_file"], "source_sha256": reference["sha256"],
                                      "origin": "historical_actual_LM_program; transfer_and_AST_mutation_not_new_generation"},
                "versions": {m: importlib.metadata.version(m) for m in ("mujoco", "numpy", "Pillow")},
                "scope": "32 groups share 4 scene-seed contexts and one source family; pipeline acceptance only"}
    save(args.output / "manifest.json", manifest)
    index = []
    for group in groups:
        unchanged()
        group_dir = args.output / "groups" / group["id"]
        group_dir.mkdir(parents=True)
        config = json.loads((ROOT / group["config_path"]).read_text())
        save(group_dir / "config.json", config)
        trials = {}
        for intervention in protocol["interventions"]:
            omission = group["condition"] in ("effect_omission", "combined") and intervention not in ("model_only", "cross_level")
            offset = protocol["policy_offset_z_m"] if group["condition"] in ("parameter", "combined") and intervention not in ("policy_only", "cross_level") else 0.
            model = model_for(group["object"], group["support"], omission)
            source = derive_source(reference["source"], offset)
            trial_dir = group_dir / uuid.uuid4().hex
            result = collect_trial(group["scene"], config, group["seed"], model, source, trial_dir, protocol)
            trials[intervention] = {"path": str((trial_dir / "result.json").relative_to(args.output)),
                                    "sha256": file_hash(trial_dir / "result.json"), "valid": result["valid"],
                                    "physical_pick_success": result["physical_pick_success"],
                                    "contract_consistent": result["contract_consistent"], "joint_success": result["joint_success"],
                                    "model_sha256": result["model_sha256"], "source_sha256": result["source_sha256"],
                                    "scene_sha256": result.get("scene_sha256"), "initial_qpos_sha256": result.get("initial_qpos_sha256"),
                                    "exception": result["exception"]}
            print(group["scene"], group["object"], group["seed"], group["condition"], intervention,
                  "physical", result["physical_pick_success"], "contract", result["contract_consistent"], "valid", result["valid"], flush=True)
        paired = len({r["scene_sha256"] for r in trials.values()}) == 1 and len({r["initial_qpos_sha256"] for r in trials.values()}) == 1
        for left, right, field in (("none", "model_only", "source_sha256"), ("none", "policy_only", "model_sha256"),
                                   ("model_only", "cross_level", "model_sha256"), ("policy_only", "cross_level", "source_sha256")):
            paired = paired and trials[left][field] == trials[right][field]
        entry = {**group, "trials": trials, "paired_layer_invariance": paired}
        save(group_dir / "group.json", entry)
        index.append({"path": str((group_dir / "group.json").relative_to(args.output)), "sha256": file_hash(group_dir / "group.json")})
        save(args.output / "progress.json", {"groups_finished": len(index), "groups_total": len(groups)})
    unchanged()
    save(args.output / "collection_frozen.json", {"manifest_sha256": file_hash(args.output / "manifest.json"),
                                                  "groups": index, "source_unchanged": True,
                                                  "training_started": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("pick_groups_protocol.json"))
    collect(parser.parse_args())
