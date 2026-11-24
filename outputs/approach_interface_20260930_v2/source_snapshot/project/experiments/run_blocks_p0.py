"""Run and freeze a pre-registered, paired Blocks intervention pilot."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

from blocks_p0 import (ROOT, ModelLibrary, diagnose, digest, execute_trial,
                       load_protocol, make_config, save)


INTERVENTIONS = ("none", "model_only", "policy_only", "cross_level")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files(protocol_path, protocol):
    ModelLibrary([]).build(set(), set())
    paths = {Path(__file__).resolve(), Path(protocol_path).resolve(), (ROOT / protocol["base_config"]).resolve()}
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            path = Path(name).resolve()
            if path.is_relative_to(ROOT) and path.suffix == ".py":
                paths.add(path)
    assets = ROOT / "project/mujoco_workspace/franka_articulated/assets/panda_assets"
    paths.update(p.resolve() for p in assets.rglob("*") if p.is_file())
    return sorted(paths)


def snapshot_sources(paths, output):
    hashes = {}
    for path in paths:
        relative = path.relative_to(ROOT)
        hashes[str(relative)] = file_hash(path)
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        if file_hash(target) != hashes[str(relative)]:
            raise RuntimeError(f"Source changed during snapshot: {relative}")
    return hashes


def assert_sources(hashes):
    changed = [name for name, previous in hashes.items() if file_hash(ROOT / name) != previous]
    if changed:
        raise RuntimeError(f"Source changed after experiment freeze: {changed}")


def load_frozen(path, phase, protocol):
    path = Path(path)
    parent = json.loads(path.read_text())
    if not isinstance(parent, dict) or parent.get("approved") is not True:
        raise ValueError("Frozen candidate must be an explicitly approved object")
    required_phase = {"validation": "development", "heldout": "validation"}.get(phase)
    if required_phase is None or parent.get("phase") != required_phase:
        raise ValueError("Frozen candidate has the wrong phase")
    if parent.get("protocol_sha256") != digest(protocol):
        raise ValueError("Frozen protocol differs from current protocol")
    if (parent.get("schema_patch") != "clear(support)"
            or parent.get("schema_patch") != protocol["candidate_precondition"]
            or type(parent.get("policy_offset_z_m")) not in (int, float)
            or parent["policy_offset_z_m"] != protocol["candidate_policy_offset_z_m"]):
        raise ValueError("Frozen candidate differs from executable candidate")
    hashes = parent.get("source_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Frozen source hashes are missing")
    evidence_path = path.parent / "results.json"
    if not evidence_path.is_file() or file_hash(evidence_path) != parent.get("evidence_results_sha256"):
        raise ValueError("Frozen evidence digest does not match")
    evidence = json.loads(evidence_path.read_text())
    if (evidence.get("phase") != required_phase or evidence.get("protocol_sha256") != digest(protocol)
            or evidence.get("source_unchanged") is not True):
        raise ValueError("Frozen evidence has incompatible provenance")
    summary = summarize(evidence["cases"])
    expected = freeze_candidates(protocol, hashes, evidence["cases"], summary)
    for field in ("approved", "model_evidence", "policy_evidence", "regressions", "summary"):
        if parent.get(field) != expected[field]:
            raise ValueError(f"Frozen {field} does not match physical evidence")
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if manifest.get("source_hashes") != hashes or manifest.get("protocol_sha256") != digest(protocol):
        raise ValueError("Frozen manifest differs from candidate")
    return parent


def summarize(cases):
    counts = {}
    for arm in INTERVENTIONS:
        valid = [case for case in cases if case.get("results", {}).get(arm, {}).get("trial_valid", True)]
        counts[arm] = {"success": sum(case["outcomes"][arm] for case in valid), "total": len(valid),
                       "invalid": len(cases) - len(valid)}
    confusion = Counter((case["evaluation_only_truth"], case["diagnosis"]) for case in cases)
    return {"groups": counts, "case_count": len(cases),
            "attribution_correct": sum(case["evaluation_only_truth"] == case["diagnosis"] for case in cases),
            "abstentions": sum(case["diagnosis"] == "abstain" for case in cases),
            "confusion": [{"truth": truth, "prediction": prediction, "count": count}
                          for (truth, prediction), count in sorted(confusion.items())],
            "cross_level_regressions": [case["id"] for case in cases
                                         if case["outcomes"]["none"] and not case["outcomes"]["cross_level"]],
            "interpretation": "paired finite interventions; cases within a layout are correlated; no universal or LM capability claim"}


def freeze_candidates(protocol, hashes, cases, summary, parent=None):
    model_evidence = [c["id"] for c in cases if c["diagnosis"] == "model"]
    policy_evidence = [c["id"] for c in cases if c["diagnosis"] == "policy"]
    regressions = [c["id"] for c in cases if c["outcomes"]["none"] and
                   not all(c["outcomes"].values())]
    approved = bool(model_evidence and policy_evidence and not regressions
                    and all(c["outcomes"]["cross_level"] for c in cases))
    return {"schema_patch": protocol["candidate_precondition"],
            "policy_offset_z_m": protocol["candidate_policy_offset_z_m"],
            "approved": approved, "protocol_sha256": digest(protocol), "source_hashes": hashes,
            "model_evidence": model_evidence, "policy_evidence": policy_evidence,
            "regressions": regressions, "summary": summary,
            "parent_freeze_sha256": digest(parent) if parent else None,
            "provenance": "engineer-specified finite candidates selected by paired physical evidence; no LLM calls"}


def write_summary(output, report):
    rows = ["# Blocks P0 实验结果", "", f"阶段：`{report['phase']}`", "",
            "这是固定 Oracle、固定控制器、预先给定候选修订的 MuJoCo 机制实验；不是大模型自主发现或真机实验。", "",
            "| 干预 | 严格任务成功 |", "|---|---:|"]
    for arm, count in report["summary"]["groups"].items():
        rows.append(f"| `{arm}` | {count['success']}/{count['total']} |")
    rows += ["", f"成对干预归因匹配：{report['summary']['attribution_correct']}/{report['summary']['case_count']}；"
             f"弃权：{report['summary']['abstentions']}。", "",
             "归因发生在四个成对物理干预完成之后，只读取四个任务成功布尔值；不是单次失败后的零成本诊断。",
             "`clear(support)`、归零抓取偏移和移除技能均为预设候选，不宣称已学会未知谓词或未知控制器。",
             "偏移只作用于策略读取 Oracle 坐标之后的目标变换；遮挡来自场景 XML 初始化。",
             "所有动作经执行器积分；未用运行时物体瞬移或 weld 完成抓放。",
             "每次干预均独立初始化同一场景，恢复不是从已经受损的失败现场继续；成本含四组执行，不代表在线恢复。", "",
             "| 案例 | 评价器标签 | 归因 | 不修订 | 只修模型 | 只修策略 | 跨层 |", "|---|---|---|---|---|---|---|"]
    for case in report["cases"]:
        values = " | ".join("通过" if case["outcomes"][arm] else "失败" for arm in INTERVENTIONS)
        rows.append(f"| `{case['id']}` | {case['evaluation_only_truth']} | {case['diagnosis']} | {values} |")
    rows += ["", "## 范围与可追溯性", "",
             f"本阶段实际执行 {len(report['cases']) * len(INTERVENTIONS)} 个 episode。",
             f"墙钟总耗时：{report['wall_seconds']:.2f} 秒。",
             "同一布局的四种故障条件相关；不把这些 episode 当作独立同分布的大样本。",
             "`manifest.json` 固定协议、源文件与资产哈希；`source_snapshot/` 保存本次代码和资产。",
             "`cases/` 保存匿名案例的完整动作日志、模型、策略、场景、接触和轨迹。",
             "`evaluation_only.json` 单独保存注入标签；不传给归因函数。",
             "失败与异常全部保留。新版本必须新建输出目录，不覆盖此结果。", ""]
    (output / "summary.md").write_text("\n".join(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("development", "validation", "heldout"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("blocks_p0_protocol.json"))
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    if args.output.exists():
        parser.error("Use a new output directory; historical results are immutable")
    if args.phase != "development" and args.frozen is None:
        parser.error("Validation/heldout require a frozen candidate artifact")
    if protocol["candidate_precondition"] != "clear(support)":
        parser.error("This pilot supports only the declared clear(support) candidate")
    if args.phase == "development" and args.frozen is not None:
        parser.error("Development must not consume a later freeze")
    parent = load_frozen(args.frozen, args.phase, protocol) if args.frozen is not None else None
    if parent is not None:
        assert_sources(parent["source_hashes"])
    candidate_offset = parent["policy_offset_z_m"] if parent is not None else protocol["candidate_policy_offset_z_m"]
    paths = source_files(args.protocol, protocol)
    args.output.mkdir(parents=True)
    hashes = snapshot_sources(paths, args.output)
    if parent and hashes != parent["source_hashes"]:
        raise RuntimeError("Imported source set differs from candidate freeze")
    manifest = {"phase": args.phase, "created_at": datetime.now(timezone.utc).isoformat(),
                "protocol": protocol, "protocol_sha256": digest(protocol), "source_hashes": hashes,
                "versions": {name: importlib.metadata.version(name) for name in ("mujoco", "numpy", "scipy", "Pillow")},
                "python": sys.version, "argv": sys.argv,
                "parent_freeze": str(args.frozen) if args.frozen else None,
                "parent_freeze_sha256": file_hash(args.frozen) if args.frozen else None}
    save(args.output / "manifest.json", manifest)
    cases, private = [], []
    start = time.monotonic()
    for layout in protocol["layouts"]:
        if layout["split"] != args.phase:
            continue
        for occupied in protocol["factors"]["target_occupied"]:
            for offset in protocol["factors"]["grasp_offset_z_m"]:
                case_id = f"case{len(cases):04d}"
                truth = "mixed" if occupied and offset else "model" if occupied else "policy" if offset else "ok"
                private.append({"case": case_id, "layout": layout["id"], "target_occupied": occupied,
                                "grasp_offset_z_m": offset, "truth": truth})
                save(args.output / "evaluation_only.json", private)
                config = make_config(protocol, layout, occupied)
                task = {key: deepcopy(layout[key]) for key in ("payload", "support", "distractor", "buffer_xy")}
                case = {"id": case_id, "outcomes": {}, "results": {}, "evaluation_only_truth": truth}
                for arm in INTERVENTIONS:
                    assert_sources(hashes)
                    clear = arm in ("model_only", "cross_level")
                    policy_offset = candidate_offset if arm in ("policy_only", "cross_level") else offset
                    result = execute_trial(protocol, config, task, clear, policy_offset,
                                           args.output / "cases" / case_id / arm, args.render)
                    case["outcomes"][arm] = result["success"]
                    case["results"][arm] = {key: result.get(key) for key in
                                            ("exception", "trial_valid", "artifact_complete", "wall_seconds", "simulation_seconds", "models_sha256", "policies_sha256", "scene_sha256")}
                    save(args.output / "progress.json", {"cases": cases, "active_case": case, "status": "running"})
                    print(case_id, arm, result["success"], result["exception"], flush=True)
                records = case["results"]
                if not all(r["trial_valid"] for r in records.values()):
                    case["diagnosis"] = "abstain"
                    case["invalid_reason"] = "Infrastructure failure excluded from intervention attribution"
                    cases.append(case)
                    save(args.output / "progress.json", {"cases": cases, "summary": summarize(cases)})
                    continue
                if len({r["scene_sha256"] for r in records.values()}) != 1:
                    raise RuntimeError("Physical scene differs across paired interventions")
                for left, right, field in (("none", "model_only", "policies_sha256"),
                                           ("none", "policy_only", "models_sha256"),
                                           ("model_only", "cross_level", "models_sha256"),
                                           ("policy_only", "cross_level", "policies_sha256")):
                    if records[left][field] != records[right][field]:
                        raise RuntimeError("Non-target layer changed in paired intervention")
                evidence = deepcopy(case["outcomes"])
                save(args.output / "cases" / case_id / "diagnostic_input.json", evidence)
                case["diagnosis"] = diagnose(evidence)
                cases.append(case)
                save(args.output / "progress.json", {"cases": cases, "summary": summarize(cases)})
    if not cases:
        raise ValueError("Protocol has no layouts for requested split")
    assert_sources(hashes)
    summary = summarize(cases)
    report = {"phase": args.phase, "summary": summary, "cases": cases,
              "wall_seconds": time.monotonic() - start, "source_unchanged": True,
              "protocol_sha256": digest(protocol)}
    save(args.output / "results.json", report)
    if args.phase != "heldout":
        frozen = freeze_candidates(protocol, hashes, cases, summary, parent)
        frozen["phase"] = args.phase
        frozen["evidence_results_sha256"] = file_hash(args.output / "results.json")
        save(args.output / "frozen_candidates.json", frozen)
    write_summary(args.output, report)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
