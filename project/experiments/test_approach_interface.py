"""No simulator construction or model inference in these interface tests."""
import ast
from copy import deepcopy
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from approach_interface_audit import (PROTOCOL, add_channel, archived_array, inspect_interfaces, load,
                                      restore_recorded_state, state_fingerprint, validate, yaw_down)
from core.cover_cabto_api import FineCover
from core.single_arm_rebuilt import RebuiltSingleArm


class ApproachInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.p = load(PROTOCOL)
        self.xml = '<mujoco><worldbody><body name="yellow"><freejoint name="yellow_free"/><geom type="box" size=".022 .022 .022" mass=".05" friction="1 .02 .001"/></body></worldbody></mujoco>'

    def test_six_fixed_queries_no_dynamics(self):
        validate(self.p)
        for key, value in (("max_pose_queries", 7), ("dynamic_steps", 1), ("render", True), ("model_calls", 1)):
            bad = deepcopy(self.p)
            bad[key] = value
            with self.assertRaises(ValueError):
                validate(bad)

    def test_empty_json_array_shape_is_restored_without_relaxing_nonempty(self):
        empty = archived_array([], np.zeros((0, 3)))
        self.assertEqual(empty.shape, (0, 3))
        nonempty = archived_array([1, 2, 3], np.zeros((1, 3)))
        self.assertEqual(nonempty.shape, (3,))

    def test_appended_fixed_walls_get_zero_force_and_existing_forces_preserved(self):
        data = SimpleNamespace(time=0., xfrc_applied=np.zeros((4, 6)))
        original = np.arange(12).reshape(2, 6).tolist()
        with patch("approach_interface_audit.mujoco.MjData", return_value=data), patch("approach_interface_audit.mujoco.mj_forward"), patch("approach_interface_audit.body_name", return_value="audit_rail_new"):
            result = restore_recorded_state(SimpleNamespace(nbody=4), {"time": .16, "xfrc_applied": original})
        np.testing.assert_array_equal(result.xfrc_applied[:2], original)
        np.testing.assert_array_equal(result.xfrc_applied[2:], 0.)
        self.assertEqual(result.time, .16)

    def test_unregistered_body_addition_rejected(self):
        data = SimpleNamespace(xfrc_applied=np.zeros((4, 6)))
        with patch("approach_interface_audit.mujoco.MjData", return_value=data), patch("approach_interface_audit.body_name", return_value="unknown_body"):
            with self.assertRaises(ValueError):
                restore_recorded_state(SimpleNamespace(nbody=4), {"xfrc_applied": np.zeros((2, 6)).tolist()})

    def test_yaw_changes_closing_direction_not_down_axis(self):
        rotations = []
        for yaw in (0, 90):
            q = yaw_down(yaw)
            self.assertAlmostEqual(np.linalg.norm(q), 1.)
            matrix = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            np.testing.assert_allclose(matrix[:, 2], [0., 0., -1.], atol=1e-12)
            rotations.append(matrix)
        self.assertAlmostEqual(float(rotations[0][:, 1] @ rotations[1][:, 1]), 0., places=10)

    def test_walls_preserve_original_geometry_and_have_no_joints(self):
        original = ET.fromstring(self.xml).find("worldbody/body")
        for scene in self.p["scenes"]:
            root = ET.fromstring(add_channel(self.xml, scene, self.p, [.34, .15], .4))
            self.assertEqual(ET.tostring(original), ET.tostring(root.find("worldbody/body[@name='yellow']")))
            rails = [b for b in root.find("worldbody") if b.get("name", "").startswith("audit_rail_")]
            self.assertEqual(len(rails), 0 if scene == "open" else 2)
            self.assertFalse(any(b.findall("joint") or b.findall("freejoint") for b in rails))
            for body in rails:
                geom = body.find("geom")
                self.assertEqual(geom.get("contype"), "1")
                self.assertEqual(geom.get("conaffinity"), "1")

    def test_channel_dimensions_follow_protocol(self):
        root = ET.fromstring(add_channel(self.xml, "rails_x", self.p, [.34, .15], .4))
        rails = [b for b in root.find("worldbody") if b.get("name", "").startswith("audit_rail_")]
        x = sorted(float(b.get("pos").split()[0]) for b in rails)
        thickness = 2 * float(rails[0].find("geom").get("size").split()[0])
        self.assertAlmostEqual(x[1] - x[0] - thickness, .08)
        self.assertGreater(.08, 2 * .022)

    def test_public_move_is_xyz_only_despite_pose_ik(self):
        audit = inspect_interfaces()
        self.assertFalse(audit["pose_argument_in_current_move"])
        self.assertFalse(audit["yaw_argument_in_public_pick_methods"])
        self.assertTrue(audit["ik_accepts_quaternion"])
        for symbol in (FineCover.approach, FineCover.descend):
            self.assertEqual(list(inspect.signature(symbol).parameters), ["self", "point"])

    def test_current_move_does_not_consume_abs_quat(self):
        point = np.array([.34, .15, .62])
        calls = []

        def solve(xyz, quat, **kwargs):
            calls.append(np.array(quat))
            return np.zeros(7)

        arm = SimpleNamespace(grasp_quat=yaw_down(0), abs_quat=yaw_down(90), arm_target=np.zeros(7), solve_ik=solve)
        runtime = SimpleNamespace(arm=arm, c={"cartesian_speed_m_s": .2, "joint_speed_rad_s": .85, "settle_seconds": .3},
                                  dt=.002, steps=10, ee=lambda: point.copy(), tick=lambda *args: None,
                                  wait=lambda *args: None, max_tracking=0., phase="test")
        RebuiltSingleArm.move(runtime, point, 1.)
        self.assertTrue(calls)
        for quat in calls:
            np.testing.assert_allclose(quat, yaw_down(0))
        self.assertFalse(np.allclose(calls[-1], arm.abs_quat))

    def test_control_state_hash_detects_changes(self):
        data = SimpleNamespace(time=.16, qpos=np.zeros(2), qvel=np.zeros(2), act=np.zeros(0),
                               ctrl=np.zeros(2), eq_active=np.zeros(0), qfrc_applied=np.zeros(2), xfrc_applied=np.zeros((1, 6)))
        old = state_fingerprint(data)
        data.ctrl[0] = 1
        self.assertNotEqual(old, state_fingerprint(data))

    def test_runner_has_no_step_or_runtime_reset_calls(self):
        tree = ast.parse(Path(__file__).with_name("approach_interface_audit.py").read_text())
        forbidden = {"mj_step", "mj_step1", "mj_step2", "reset", "reset_arm", "step_joint", "Renderer", "Exp4Env", "BlocksExperiment", "RebuiltSingleArm"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else None
                self.assertNotIn(name, forbidden)


if __name__ == "__main__":
    unittest.main()
