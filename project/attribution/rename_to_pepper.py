#!/usr/bin/env python3
"""机械重命名 cover 证据：potato → pepper。

换物体只改「物体名 + 颜色」（几何复用 potato 椭球体），动作模型结构不变，
策略代码的 API 调用物体无关（仅注释提到物体名）。所以机械替换 potato→pepper
是诚实的（明确标注 mechanical rename），随后用 replay 重新物理验证。

不覆盖旧证据：输出到 outputs/cover_cabto_fix/pepper/。
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # project/
PACKAGE = ROOT.parent

CONFIG = ROOT / "scenes/cover_kitchen_sort_v2.json"
OLD_MODELS = PACKAGE / "outputs/cover_cabto_fix/proposal_final_v2/models.json"
OLD_POLICIES = PACKAGE / "outputs/cover_cabto_fix/sampling_9b_complete/policies.json"
OUT = PACKAGE / "outputs/cover_cabto_fix/pepper"


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha256_str(s):
    return hashlib.sha256(s.encode()).hexdigest()


def main():
    new_config_sha = digest(CONFIG)
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 1) models.json：models / P / check 里所有 potato → pepper ----
    models = json.loads(OLD_MODELS.read_text())
    models = json.loads(json.dumps(models).replace("potato", "pepper"))
    models["config_sha256"] = new_config_sha
    models["rename_note"] = {
        "from": "potato", "to": "pepper", "mode": "mechanical_object_rename",
        "source": str(OLD_MODELS), "source_sha256": digest(OLD_MODELS),
        "reason": "换物体：potato(棕) 与 shrimp(橙) 颜色相近难辨，换成认知一致的青椒(绿)",
    }
    (OUT / "models.json").write_text(json.dumps(models, indent=2, ensure_ascii=False))

    # ---- 2) policies.json：accepted 里所有 potato → pepper，重算 source sha256 ----
    policies = json.loads(OLD_POLICIES.read_text())
    new_accepted = {}
    for key, entry in policies["accepted"].items():
        new_key = key.replace("potato", "pepper")
        new_entry = json.loads(json.dumps(entry).replace("potato", "pepper"))
        new_entry["sha256"] = sha256_str(new_entry["source"])   # source 注释变了，重算
        new_accepted[new_key] = new_entry
    policies["accepted"] = new_accepted
    policies["config_sha256"] = new_config_sha
    policies["rename_note"] = {
        "from": "potato", "to": "pepper", "mode": "mechanical_object_rename",
        "source": str(OLD_POLICIES), "source_sha256": digest(OLD_POLICIES),
    }
    (OUT / "policies.json").write_text(json.dumps(policies, indent=2, ensure_ascii=False))

    # ---- 验证 ----
    assert "potato" not in json.dumps(models["models"]), "models 里还有 potato"
    assert "potato" not in json.dumps(policies["accepted"]), "accepted 里还有 potato"
    for key, entry in policies["accepted"].items():
        assert entry["sha256"] == sha256_str(entry["source"]), f"sha256 不一致: {key}"
    print("config_sha256:", new_config_sha)
    print("models.json  ->", OUT / "models.json")
    print("policies.json ->", OUT / "policies.json")
    print("动作模型数:", len(models["models"]), "| 策略数:", len(policies["accepted"]))
    print("OK：potato 已全部替换为 pepper，source sha256 已重算")


if __name__ == "__main__":
    main()
