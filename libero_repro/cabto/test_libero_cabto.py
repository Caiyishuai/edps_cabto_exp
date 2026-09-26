"""CABTO-on-LIBERO 单元测试。

运行（在 libero_repro 目录）：
    .venv/bin/python -m unittest cabto.test_libero_cabto -v
或
    .venv/bin/python cabto/test_libero_cabto.py

覆盖：BDDL 解析（130 任务）、谓词→原子转换、动作库合法性、形式化 BT 符号规划可达性、
效果校验器、代码生成契约、以及 prepare/sample/replay 端到端。
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cabto.bddl import list_tasks, load_all, load_suite, parse_task
from cabto.atoms import build_problem
from cabto.library import reference_models
from cabto.planner import plancheck
from cabto.checker import transition_check
from cabto.codegen import REFERENCE_POLICIES, API_DOCS, build_program
from cabto import pipeline


class TestBDDLParsing(unittest.TestCase):
    def test_all_130_tasks_parse(self):
        tasks = load_all()
        self.assertEqual(len(tasks), 130)
        for t in tasks:
            self.assertTrue(t.language, f"{t.suite}/{t.name} 缺语言指令")
            self.assertIsInstance(t.initial_state, list)
            self.assertIsInstance(t.goal_state, list)

    def test_suite_task_counts(self):
        expected = {"libero_object": 10, "libero_spatial": 10,
                    "libero_goal": 10, "libero_10": 10, "libero_90": 90}
        for suite, n in expected.items():
            self.assertEqual(len(list_tasks(suite)[suite]), n, suite)

    def test_milk_basket_goal(self):
        t = parse_task("libero_object", "pick_up_the_milk_and_place_it_in_the_basket")
        p = build_problem(t)
        self.assertIn("in_region(milk_1,basket_1_contain_region)", p.goal)

    def test_drawer_articulation(self):
        t = parse_task("libero_goal", "open_the_top_drawer_and_put_the_bowl_inside")
        p = build_problem(t)
        self.assertIn("closed(wooden_cabinet_1_top_region)", p.init)
        self.assertIn("in_region(akita_black_bowl_1,wooden_cabinet_1_top_region)", p.goal)

    def test_stove_turnon(self):
        t = parse_task("libero_goal", "turn_on_the_stove")
        p = build_problem(t)
        self.assertIn("turned_off(flat_stove_1)", p.init)
        self.assertIn("turned_on(flat_stove_1)", p.goal)

    def test_microwave_open_init(self):
        t = parse_task("libero_10", "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it")
        p = build_problem(t)
        self.assertIn("open(microwave_1)", p.init)
        self.assertIn("closed(microwave_1)", p.goal)


class TestActionLibrary(unittest.TestCase):
    def test_all_libraries_valid(self):
        for t in load_all():
            p = build_problem(t)
            models = reference_models(p)
            from formal_bt import validate_model, action_id
            ids = []
            for m in models:
                ids.append(validate_model(m))  # pre∩add=∅, add∩del=∅
            self.assertEqual(len(ids), len(set(ids)), f"{t.suite}/{t.name} 重复 action_id")
            # 每个动作必须非空（pick 至少一个，place 至少一个）
            self.assertTrue(models, f"{t.suite}/{t.name} 空动作库")


class TestSymbolicPlanning(unittest.TestCase):
    def test_all_130_reachable(self):
        for t in load_all():
            p = build_problem(t)
            r = plancheck(p, reference_models(p))
            self.assertTrue(r["reached_goal"], f"{t.suite}/{t.name}: {r.get('reason')}")

    def test_multi_step_plan_order(self):
        t = parse_task("libero_goal", "open_the_top_drawer_and_put_the_bowl_inside")
        p = build_problem(t)
        r = plancheck(p, reference_models(p))
        self.assertEqual(r["action_ids"][0], "open(entity=wooden_cabinet_1_top_region)")
        self.assertEqual(r["action_ids"][-1], "place_in(container=wooden_cabinet_1_top_region,object=akita_black_bowl_1)")


class TestEffectChecker(unittest.TestCase):
    def test_valid_transition(self):
        m = {"name": "pick", "args": {"object": "a"},
             "pre": ["empty()", "on_region(a,r)"], "add": ["holding(a)"],
             "del": ["empty()", "on_region(a,r)"]}
        before = {"empty()", "on_region(a,r)"}
        after = {"holding(a)"}
        c = transition_check(m, before, after)
        self.assertTrue(c["effect_ok"])
        self.assertTrue(c["strict_CP_transition_match"])

    def test_missing_add(self):
        m = {"name": "pick", "args": {"object": "a"},
             "pre": ["empty()", "on_region(a,r)"], "add": ["holding(a)"],
             "del": ["empty()", "on_region(a,r)"]}
        c = transition_check(m, {"empty()", "on_region(a,r)"}, {"empty()"})
        self.assertFalse(c["effect_ok"])
        self.assertEqual(c["violations"]["missing_add"], ["holding(a)"])


class TestCodegen(unittest.TestCase):
    def test_reference_policies_valid(self):
        for kind in API_DOCS:
            prog = build_program(REFERENCE_POLICIES[kind], kind)
            self.assertIsNotNone(prog)

    def test_malformed_policy_rejected(self):
        bad = "def policy(api):\n    import os\n    os.system('rm -rf /')\n"
        with self.assertRaises(Exception):
            build_program(bad, "pick")

    def test_unknown_api_rejected(self):
        bad = "def policy(api):\n    api.fly()\n"
        with self.assertRaises(Exception):
            build_program(bad, "pick")


class TestPipeline(unittest.TestCase):
    def test_prepare_sample_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            r = pipeline.prepare("libero_goal", "open_the_top_drawer_and_put_the_bowl_inside",
                                 tmp / "prepare")
            self.assertTrue(r["check"]["reached_goal"])
            models_path = tmp / "prepare" / "models.json"
            s = pipeline.sample("libero_goal", "open_the_top_drawer_and_put_the_bowl_inside",
                                models_path, tmp / "sample")
            self.assertTrue(s["final_reached_goal"])
            rp = pipeline.replay("libero_goal", "open_the_top_drawer_and_put_the_bowl_inside",
                                 models_path, tmp / "sample" / "policies.json", tmp / "replay")
            self.assertTrue(rp["success"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
