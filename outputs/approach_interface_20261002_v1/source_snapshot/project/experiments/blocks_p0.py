"""Controlled Blocks grounding experiments using actuator-only MuJoCo execution."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(ROOT / "CABTO/exp4_bt_tasks/stage3_cabto"))

from core.single_arm_rebuilt import RebuiltSingleArm, scene, vec
from core.cover_cabto_api import API_DOCS, FineCover, transition_check
from formal_bt import ModelLibrary, PlanningError, validate_model
from policy_codegen import PolicyProgram


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def load_protocol(path=None):
    path = Path(path) if path else Path(__file__).with_name("blocks_p0_protocol.json")
    protocol = json.loads(path.read_text())
    ids = [row["id"] for row in protocol["layouts"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Layout IDs must be unique across splits")
    for row in protocol["layouts"]:
        if len({row[k] for k in ("payload", "support", "blocker", "distractor")}) != 4:
            raise ValueError("Each physical object must have one role")
    return protocol


def make_config(protocol, layout, occupied):
    config = json.loads((ROOT / protocol["base_config"]).read_text())
    config["jitter_m"] = 0.0
    config["pairs"] = [[layout["payload"], layout["support"]]]
    config["blocks"] = {
        layout["payload"]: list(layout["source_xy"]),
        layout["support"]: list(layout["target_xy"]),
        layout["blocker"]: list(layout["target_xy"] if occupied else layout["buffer_xy"]),
        layout["distractor"]: list(layout["distractor_xy"]),
    }
    config["initial_on"] = [[layout["blocker"], layout["support"]]] if occupied else []
    return config


def build_scene(config, seed=0):
    xml, names = scene(config, seed)
    root = ET.fromstring(xml)
    bodies = {body.attrib["name"]: body for body in root.findall("./worldbody/body")}
    for obj, support in config.get("initial_on", []):
        base = np.fromstring(bodies[support].attrib["pos"], sep=" ")
        base[2] += 2 * config["cube_half_size"] + 0.001
        bodies[obj].set("pos", vec(base))
    return ET.tostring(root, encoding="unicode"), names


def action_models(objects, support, payload, require_clear):
    models = []
    for obj in objects:
        if obj == support:
            continue
        holding, table, clear = f"holding({obj})", f"ontable({obj})", f"clear({obj})"
        models.append({"name": "pick_table", "args": {"object": obj},
                       "pre": ["empty()", table, clear], "add": [holding],
                       "del": ["empty()", table, "home_completed()"]})
        models.append({"name": "unstack", "args": {"object": obj, "support": support},
                       "pre": ["empty()", f"on({obj},{support})", clear],
                       "add": [holding, f"clear({support})"],
                       "del": ["empty()", f"on({obj},{support})", "home_completed()"]})
        models.append({"name": "put_table", "args": {"object": obj, "support": "table"},
                       "pre": [holding], "add": ["empty()", table],
                       "del": [holding, "home_completed()"]})
    models.append({"name": "stack", "args": {"object": payload, "support": support},
                   "pre": [f"holding({payload})"] + ([f"clear({support})"] if require_clear else []),
                   "add": ["empty()", f"on({payload},{support})"],
                   "del": [f"holding({payload})", f"clear({support})", "home_completed()"]})
    models.append({"name": "home", "args": {},
                   "pre": ["empty()", f"on({payload},{support})"],
                   "add": ["home_completed()"], "del": []})
    for model in models:
        validate_model(model)
    return models


def policy_source(kind, offset_z=0.0):
    if kind in ("pick_table", "unstack"):
        return ("def policy(api):\n"
                "    p = api.grasp_point()\n"
                f"    target = (p[0], p[1], p[2] + {float(offset_z)!r})\n"
                "    api.open_gripper()\n"
                "    api.approach(target)\n"
                "    api.descend(target)\n"
                "    api.close_gripper()\n"
                "    api.lift()\n")
    if kind in ("stack", "put_table"):
        return ("def policy(api):\n"
                "    p = api.destination_point()\n"
                "    api.carry(p)\n"
                "    api.lower(p)\n"
                "    api.open_gripper()\n"
                "    api.withdraw()\n")
    if kind == "home":
        return "def policy(api):\n    api.return_home()\n"
    raise ValueError(kind)


class BlocksExperiment(FineCover):
    def __init__(self, config, task, protocol, render=False):
        self.r = RebuiltSingleArm(deepcopy(config), render=False, scene_builder=build_scene)
        self.env = self.r.env
        self.task = deepcopy(task)
        self.protocol = protocol
        self.binding = None
        self.reads = []
        self.source_supports = {}
        self.render_requested = render
        self.goal = {f"on({task['payload']},{task['support']})", "empty()", "home_completed()"}

    def on(self, upper, lower):
        r = self.r
        p, base = r.pos(upper), r.pos(lower)
        half = r.config["cube_half_size"]
        return bool(np.linalg.norm(p[:2] - base[:2]) < 2 * half - .002
                    and abs(p[2] - base[2] - 2 * half) < .008
                    and r.supporting_contact(upper, lower))

    def state(self):
        r = self.r
        atoms = set()
        if r.released():
            atoms.add("empty()")
        names = r.env.obj_names
        for obj in names:
            held = (r.held(obj) and r.pos(obj)[2] > r.initial[obj][2] + .075
                    and min(r.force_now.values()) >= .05)
            if held:
                atoms.add(f"holding({obj})")
            elif (r.supporting_contact(obj, "table")
                  and abs(r.pos(obj)[2] - r.config["table_top"] - r.config["cube_half_size"]) < .008):
                atoms.add(f"ontable({obj})")
            above = [other for other in names if other != obj and self.on(other, obj)]
            if not above:
                atoms.add(f"clear({obj})")
            for other in names:
                if other != obj and self.on(obj, other):
                    atoms.add(f"on({obj},{other})")
        if r.assess()["final_home"]:
            atoms.add("home_completed()")
        return atoms

    def snapshot(self):
        r = self.r
        return {"time": float(self.env.d.time), "state": sorted(self.state()),
                "objects": {obj: {"position": r.pos(obj).tolist(), "quaternion": r.pose(obj)[1].tolist(),
                                   "velocity": self.env.d.qvel[self.env.obj_dofadr[obj]:self.env.obj_dofadr[obj]+6].tolist()}
                            for obj in self.env.obj_names},
                "finger_forces_N": dict(r.force_now), "tcp": r.ee().tolist(),
                "active_welds": r.active_welds()}

    def bind(self, model):
        kind = model["name"]
        api_kind = "pick" if kind in ("pick_table", "unstack") else "home" if kind == "home" else "place"
        mapped = {**deepcopy(model), "name": api_kind}
        result = super().bind(mapped)
        if api_kind == "pick":
            self.source_supports[self.obj] = {name for name in self.env.obj_names if name != self.obj and self.on(self.obj, name)}
        return api_kind, result

    def grasp_point(self):
        point = tuple(map(float, self.r.grip_point(self.obj)))
        self.reads.append({"kind": "oracle_grasp", "object": self.obj, "xyz": point})
        return point

    def destination_point(self):
        if self.support == "table":
            point = [*self.task["buffer_xy"], self.r.config["table_top"] + self.r.config["cube_half_size"] + .003]
        else:
            point = self.r.pos(self.support) + [0, 0, 2 * self.r.config["cube_half_size"] + .003]
        point = tuple(map(float, point))
        self.reads.append({"kind": "oracle_destination", "object": self.support, "xyz": point})
        return point

    def withdraw(self):
        r = self.r
        if not r.released():
            raise RuntimeError("Withdraw requires physical release")
        r.mark("withdraw_" + self.obj)
        r.move([*r.ee()[:2], r.c["transport_z"]], 1., r.c["lift_speed_m_s"])
        r.target = None
        r.wait(.10, 1.)

    def contact_audit(self):
        allowed, forbidden = [], []
        for row in self.r.contacts.values():
            if row["min_distance_m"] >= -.0003:
                continue
            source_contact = False
            for obj, supports in self.source_supports.items():
                if row["phase"] not in {prefix + obj for prefix in ("approach_", "descend_", "grasp_", "lift_")}:
                    continue
                source_contact |= (row["kind"] == "object_obstacle" and obj in row["bodies"]
                                   and bool(supports.intersection(row["bodies"]))
                                   and row["min_distance_m"] >= -.002)
            (allowed if source_contact else forbidden).append(deepcopy(row))
        return {"allowed_initial_support_contacts": allowed, "forbidden_contacts": forbidden,
                "allowance": "only observed source support during unstack; maximum penetration 2mm"}

    def final_check(self):
        r = self.r
        legacy = r.assess()
        goal_ok = self.goal <= self.state()
        placement = r.assess_one((self.task["payload"], self.task["support"]))
        velocities = {obj: self.env.d.qvel[self.env.obj_dofadr[obj]:self.env.obj_dofadr[obj]+6].copy()
                      for obj in self.env.obj_names}
        velocity_ok = all(np.linalg.norm(v[:3]) < self.protocol["max_object_linear_speed_m_s"]
                          and np.linalg.norm(v[3:]) < self.protocol["max_object_angular_speed_rad_s"] for v in velocities.values())
        static_error = {obj: float(np.linalg.norm(r.pos(obj) - r.initial[obj]))
                        for obj in (self.task["support"], self.task["distractor"])}
        audit = self.contact_audit()
        mechanical_ok = bool(placement["ok"] and legacy["released"] and legacy["final_home"]
                             and legacy["grasp_verified"] and legacy["motion_verified"]
                             and max(static_error.values()) < .004 and velocity_ok
                             and not audit["forbidden_contacts"])
        return {"success": bool(goal_ok and mechanical_ok), "goal_reached": goal_ok,
                "mechanical_ok": mechanical_ok, "placement": placement, "static_displacement_m": static_error,
                "velocity_ok": bool(velocity_ok), "contact_audit": audit, "legacy_metrics": legacy}

    def stable_final(self):
        r = self.r
        r.mark("final_stability")
        stable = True
        for _ in range(math.ceil(self.protocol["stable_seconds"] / (r.dt * r.steps))):
            r.wait(r.dt * r.steps, 1.)
            stable = stable and self.final_check()["success"]
        result = self.final_check()
        result["stable_window_pass"] = bool(stable)
        result["success"] = bool(result["success"] and stable)
        return result

    def save_image(self, path):
        if not self.render_requested:
            return
        import mujoco
        from PIL import Image
        if self.env.renderer is None:
            self.env.renderer = mujoco.Renderer(self.env.m, 720, 720)
        self.env.renderer.update_scene(self.env.d, camera="overview")
        image = np.asarray(self.env.renderer.render()).copy()
        if image.max() == image.min():
            raise RuntimeError("Renderer produced a constant image")
        Image.fromarray(image).save(path)


class PhysicalTrialFailure(RuntimeError):
    """An observed control, transition or safety failure, not an infrastructure error."""


def execute_trial(protocol, config, task, require_clear, offset_z, output, render=False, *,
                  models_override=None, experiment_type=BlocksExperiment):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"success": False, "trial_valid": False, "artifact_complete": True,
              "steps": [], "exception": None, "infrastructure_errors": [],
              "scope": "oracle_actuator_only_simulation_finite_candidate_pilot"}
    save(output / "result.json", {**result, "status": "started"})
    runtime = None

    def collect(label, operation, *, essential=False):
        try:
            return operation()
        except Exception as exc:
            result["artifact_complete"] = False
            result["infrastructure_errors"].append({"stage": label, "error": f"{type(exc).__name__}: {exc}"})
            if essential:
                result["trial_valid"] = False
                result["success"] = False
            return None

    try:
        runtime = experiment_type(config, task, protocol, render)
        result["initial"] = runtime.snapshot()
        result["scene_sha256"] = hashlib.sha256(runtime.r.xml.encode()).hexdigest()
        (output / "scene.xml").write_text(runtime.r.xml)
        collect("before_render", lambda: runtime.save_image(output / "before.png"))
        models = (deepcopy(models_override) if models_override is not None else
                  action_models(runtime.env.obj_names, task["support"], task["payload"], require_clear))
        for model in models:
            validate_model(model)
        sources = {m["name"]: policy_source(m["name"], offset_z) for m in models}
        save(output / "models.json", models)
        save(output / "policies.json", sources)
        result["models_sha256"], result["policies_sha256"] = digest(models), digest(sources)
        tree = ModelLibrary(models).build(runtime.state(), runtime.goal)
        (output / "tree.json").write_text(tree.export())
        result["trial_valid"] = True
        for _ in range(protocol["max_actions"]):
            before = runtime.snapshot()
            status, model = tree.tick(set(before["state"]))
            result["bt_status"] = status
            if status != "running":
                break
            row = {"model": model, "before": before, "exception": None, "api_trace": [], "status": "started"}
            result["steps"].append(row)
            step_path = output / f"step{len(result['steps']):02d}.json"
            save(step_path, row)
            if not set(model["pre"]) <= set(before["state"]):
                raise ValueError("Planner selected an inapplicable action")
            try:
                api_kind, api = runtime.bind(model)
                program = PolicyProgram(sources[model["name"]], API_DOCS[api_kind])
                row["api_trace"] = program.run(api, call_limit=20)["trace"]
            except Exception as exc:
                row["exception"] = f"{type(exc).__name__}: {exc}"
                row["api_trace"] = getattr(exc, "policy_trace", [])
                save(step_path, row)
                if not isinstance(exc, RuntimeError) or not row["api_trace"]:
                    raise
            row["after"] = runtime.snapshot()
            row["check"] = transition_check(model, before["state"], row["after"]["state"], row["exception"])
            row["contact_audit"] = runtime.contact_audit()
            row["status"] = "complete"
            save(step_path, row)
            if not row["check"]["effect_ok"] or row["contact_audit"]["forbidden_contacts"]:
                raise PhysicalTrialFailure(row["exception"] or "Observed action contract or contact safety failed")
        status, _ = tree.tick(runtime.state())
        result["bt_status"] = status
        result["final_check"] = runtime.stable_final()
        result["success"] = bool(status == "success" and result["final_check"]["success"])
    except PhysicalTrialFailure as exc:
        result["exception"] = str(exc)
    except PlanningError as exc:
        result["trial_valid"] = True
        result["candidate_rejected"] = True
        result["exception"] = f"PlanningError: {exc}"
    except Exception as exc:
        result["trial_valid"] = False
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["infrastructure_errors"].append({"stage": "execution", "error": result["exception"]})
    finally:
        if runtime is not None:
            result["final"] = collect("final_observation", runtime.snapshot, essential=True)
            if "final_check" not in result:
                result["final_check"] = collect("final_check", runtime.final_check, essential=True)
            result["simulation_seconds"] = collect("clock", lambda: float(runtime.env.d.time - result.get("initial", {"time": .16})["time"]), essential=True)
            result["oracle_reads"] = collect("oracle_reads", lambda: runtime.reads)
            collect("events", lambda: save(output / "events.json", runtime.r.events))
            collect("trajectory", lambda: save(output / "trajectory.json", runtime.r.trace))
            collect("after_render", lambda: runtime.save_image(output / "after.png"))
            collect("close", runtime.close)
        result["wall_seconds"] = time.monotonic() - started
        result["status"] = "complete" if result["trial_valid"] else "invalid"
        save(output / "result.json", result)
    return result


def diagnose(outcomes):
    """Identify intervention effects, never read injection labels or layout metadata."""
    keys = ("none", "model_only", "policy_only", "cross_level")
    if set(outcomes) != set(keys) or any(type(outcomes[key]) is not bool for key in keys):
        raise ValueError("Four boolean physical outcomes required")
    bits = tuple(outcomes[key] for key in keys)
    return {(True, True, True, True): "ok", (False, True, False, True): "model",
            (False, False, True, True): "policy", (False, False, False, True): "mixed"}.get(bits, "abstain")
