#!/usr/bin/env python3
"""归因器 CLI：对单臂任务（cover / blocks）的动作库做统计化三路归因。

把三探针对照实验（oracle 上限 / VLM 策略+oracle 定位 / VLM 策略+qwen 定位）
升级为「多 seed 采样 + 二项置信区间 + abstain」的循环内组件。

用法
----
    python3 probe_attribution.py blocks                       # 全动作库三路归因
    python3 probe_attribution.py cover --no-perception        # 只判 model/policy 两层
    python3 probe_attribution.py blocks --action-id "pick(object=yellow,support=green)"
    python3 probe_attribution.py blocks --k 3 --threshold 0.5 --confidence 0.8

依赖：已采样的 policies.json（blocks/cover 各有一份历史产物），
以及 qwen 感知 VLM（run_perception 时需能本地加载）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # project/
PACKAGE = ROOT.parent                                # edps_cabto_exp/
sys.path.insert(0, str(ROOT))

from core.attribution import attribute_accepted, TASKS  # noqa: E402
from core.cover_cabto_grounding import LocalModel       # noqa: E402

DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"


def main():
    ap = argparse.ArgumentParser(description="CABTO 单臂归因器")
    ap.add_argument("task", choices=["cover", "blocks"])
    ap.add_argument("--model-id", default=DEFAULT_MODEL, help="qwen 感知 VLM")
    ap.add_argument("--k", type=int, default=3, help="每路探针采样次数")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="显式 seed 列表（覆盖 --k）")
    ap.add_argument("--threshold", type=float, default=0.5, help="判 low/high 的成功率分界")
    ap.add_argument("--confidence", type=float, default=0.8, help="Wilson 区间置信度")
    ap.add_argument("--no-perception", action="store_true", help="跳过探针3（感知层）")
    ap.add_argument("--action-id", default=None, help="只归因指定动作")
    ap.add_argument("--policies", type=Path, default=None, help="指定 policies.json 路径")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    client = None
    if not args.no_perception:
        print(f"加载 qwen 感知 VLM: {args.model_id} ...")
        client = LocalModel(args.model_id)

    out = args.output or ROOT / "attribution" / "out" / args.task / "attributor"
    rows = attribute_accepted(
        args.task, client=client, k=args.k, seeds=args.seeds, out=out,
        threshold=args.threshold, confidence=args.confidence,
        run_perception=not args.no_perception,
        policies_path=args.policies, action_id=args.action_id)

    outdir = out if isinstance(out, Path) else Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "attribution.json").write_text(
        json.dumps({"task": args.task, "rows": rows}, indent=2, ensure_ascii=False))

    n_seeds = len(args.seeds) if args.seeds else args.k
    header = f"{'action_id':<34}{'p1(oracle)':<18}{'p2(vlm+oracle)':<20}{'p3(vlm+qwen)':<20}{'verdict'}"
    print(f"\n=== 归因器 | task={args.task} | seeds={n_seeds} | thr={args.threshold} | conf={args.confidence} ===")
    print(header)
    print("-" * len(header))
    for r in rows:
        def fmt(p):
            if p is None:
                return "N/A"
            return f"{p['ok_count']}/{p['total']} [{p['ci'][0]},{p['ci'][1]}]"
        print(f"{r['action_id']:<34}{fmt(r['probes']['oracle']):<18}"
              f"{fmt(r['probes']['policy']):<20}{fmt(r['probes']['perception']):<20}"
              f"{r['verdict']}")
        if r["verdict"] != "ok":
            print(f"      -> {r['reason']}")
            for key, probe in r["probes"].items():
                if probe is None:
                    continue
                for run in probe["runs"]:
                    if not run["ok"]:
                        print(f"         {key} seed={run['seed']} exc={run.get('exception')} "
                              f"violations={ {k: v for k, v in run.get('violations', {}).items() if v} }")

    print(f"\n结果写入: {outdir / 'attribution.json'}")


if __name__ == "__main__":
    main()
