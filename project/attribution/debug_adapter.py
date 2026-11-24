"""诊断：adapter 是否真的加载 + 训练图推理"""
import json
from pathlib import Path

from mlx_vlm import load, generate

MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-bf16"
ADAPTER = "/Users/yishuaicai/mywork/edps_cabto_exp/outputs/epm_px_data/adapters"

print("=== 加载模型（无 adapter）===", flush=True)
model0, _ = load(MODEL, processor_config={"trust_remote_code": True})
q0 = model0.language_model.model.layers[0].self_attn.q_proj
print("无 adapter 时 q_proj 类型:", type(q0).__name__, flush=True)

print("=== 加载模型（有 adapter）===", flush=True)
model1, processor = load(MODEL, adapter_path=ADAPTER, processor_config={"trust_remote_code": True})
q1 = model1.language_model.model.layers[0].self_attn.q_proj
print("有 adapter 时 q_proj 类型:", type(q1).__name__, flush=True)
print("有 lora_a:", hasattr(q1, "lora_a"), "有 lora_b:", hasattr(q1, "lora_b"), flush=True)

# 统计 LoRA 层数量
lora_cnt = sum(1 for _, m in model1.named_modules() if type(m).__name__ in ("LoRALinear", "LoRaLayer"))
print("LoRA 层数量:", lora_cnt, flush=True)

# 用训练图推理
print("=== 用训练图推理 ===", flush=True)
rows = [json.loads(l) for l in Path("/Users/yishuaicai/mywork/edps_cabto_exp/outputs/epm_px_data/train.jsonl").read_text().splitlines()]
for i in [0, 1]:
    row = rows[i]
    img = row["images"][0]
    out = generate(model1, processor, row["question"], image=[img], max_tokens=128)
    t = out.text if hasattr(out, "text") else str(out)
    print(f"--- 训练样本 {i} ({Path(img).name}) ---", flush=True)
    print("真值:", repr(row["answer"]), flush=True)
    print("输出:", repr(t), flush=True)
