"""Pure-logic tests for yaw_capability; no simulator construction or model inference."""
import json
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from blocks_p0 import BlocksExperiment
from yaw_capability import PROTOCOL, YawExperiment, quat_rotation_error, validate, yaw_down


class YawCapabilityTests(unittest.TestCase):
    def setUp(self):
        self.p = json.loads(PROTOCOL.read_text())

    def test_yaw_down_zero_is_default_down(self):
        np.testing.assert_allclose(yaw_down(0.0), [0.0, 1.0, 0.0, 0.0], atol=1e-12)

    def test_yaw_down_keeps_tool_plus_z_down(self):
        for yaw in (0.0, 30.0, 90.0, 180.0, 270.0):
            q = yaw_down(yaw)
            self.assertAlmostEqual(np.linalg.norm(q), 1.0)
            matrix = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            np.testing.assert_allclose(matrix[:, 2], [0.0, 0.0, -1.0], atol=1e-12)

    def test_yaw_down_closing_directions_are_orthogonal(self):
        m0 = Rotation.from_quat(yaw_down(0.0)[[1, 2, 3, 0]]).as_matrix()
        m90 = Rotation.from_quat(yaw_down(90.0)[[1, 2, 3, 0]]).as_matrix()
        self.assertAlmostEqual(float(m0[:, 1] @ m90[:, 1]), 0.0, places=10)

    def test_quat_rotation_error_identity_is_zero(self):
        q = yaw_down(90.0)
        self.assertAlmostEqual(quat_rotation_error(q, q), 0.0, places=12)

    def test_validate_accepts_registered_protocol(self):
        validate(self.p)

    def test_validate_rejects_bad_variants(self):
        bad = deepcopy(self.p)
        bad["variants"][0]["id"] = "other"
        with self.assertRaises(ValueError):
            validate(bad)
        bad2 = deepcopy(self.p)
        bad2["variants"][-1]["yaw_deg"] = 45.0
        with self.assertRaises(ValueError):
            validate(bad2)
        bad3 = deepcopy(self.p)
        bad3["grasp_offset_z_m"] = 0.045
        with self.assertRaises(ValueError):
            validate(bad3)

    def test_yaw_experiment_default_keeps_grasp_quat(self):
        def fake_init(self, config, task, protocol, render=False):
            self.r = SimpleNamespace(target_quat=None,
                                     arm=SimpleNamespace(grasp_quat=np.array([0.0, 1.0, 0.0, 0.0])))

        with patch.object(BlocksExperiment, "__init__", fake_init):
            exp = YawExperiment({}, {}, {}, render=False, yaw_deg=None)
        self.assertIsNone(exp.r.target_quat)

    def test_yaw_experiment_sets_target_quat_for_90(self):
        def fake_init(self, config, task, protocol, render=False):
            self.r = SimpleNamespace(target_quat=None,
                                     arm=SimpleNamespace(grasp_quat=np.array([0.0, 1.0, 0.0, 0.0])))

        with patch.object(BlocksExperiment, "__init__", fake_init):
            exp = YawExperiment({}, {}, {}, render=False, yaw_deg=90.0)
        np.testing.assert_allclose(exp.r.target_quat, yaw_down(90.0), atol=1e-12)
        self.assertAlmostEqual(exp.yaw_deg, 90.0)


if __name__ == "__main__":
    unittest.main()
