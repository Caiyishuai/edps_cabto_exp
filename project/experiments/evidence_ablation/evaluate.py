"""Run the evidence-ablation comparison across methods A / B / C.

Usage:
    python evaluate.py [--output result.json]

Prints a per-method table and writes the full per-episode breakdown as JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract_sim import summarize  # noqa: E402
from evidence import build_archive, load_protocol  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("result.json"))
    args = parser.parse_args(argv)

    protocol = load_protocol(args.protocol)
    archive = build_archive(protocol)
    methods = protocol["methods"]

    results = {}
    for key in ("A", "B", "C"):
        results[key] = summarize(archive, methods[key])

    lines = []
    header = (
        f"{'method':<6} {'label':<26} {'corr':>5} {'wrongT':>7} {'wrongA':>7} "
        f"{'abst':>5} {'frames':>7} {'insp':>5} {'cov':>6}"
    )
    lines.append(header)
    for key in ("A", "B", "C"):
        r = results[key]
        o = r["outcomes"]
        cov = "-" if r["task_coverage"] is None else f"{r['task_coverage']:.2f}"
        row = (
            f"{key:<6} {methods[key]['label']:<26} {o['correct']:>5} "
            f"{o['wrong_tighten']:>7} {o['wrong_accept']:>7} {o['abstain']:>5} "
            f"{r['evidence_frames_total']:>7} {r['inspect_calls_total']:>5} {cov:>6}"
        )
        lines.append(row)
    table = "\n".join(lines)
    print(table)

    payload = {
        "protocol_version": protocol["version"],
        "experiment_question": protocol["experiment_question"],
        "methods": {key: methods[key]["label"] for key in methods},
        "results": {
            key: {
                "outcomes": results[key]["outcomes"],
                "evidence_frames_total": results[key]["evidence_frames_total"],
                "inspect_frames_total": results[key]["inspect_frames_total"],
                "inspect_calls_total": results[key]["inspect_calls_total"],
                "task_coverage": results[key]["task_coverage"],
                "executable_tasks": results[key]["executable_tasks"],
                "retained_tasks": results[key]["retained_tasks"],
                "per_episode": results[key]["per_episode"],
            }
            for key in ("A", "B", "C")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
