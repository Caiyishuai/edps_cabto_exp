"""手动自回归验证：用训练一致的 prompt 构造 + argmax 贪心，看是否输出正确坐标"""
import json
from pathlib import Path

import numpy as np

import mlx_vlm.models.rope_utils as rope_utils
rope_utils._HAS_METAL = False

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-bf16"
ADAPTER = "/Users/yishuaicai/mywork/edps_cabto_exp/outputs/epm_px_data/adapters"
TRAIN_JSONL = "/Users/yishuaicai/mywork/edps_cabto_exp/outputs/epm_px_data/train.jsonl"

model, processor = load(MODEL, adapter_path=ADAPTER, processor_config={"trust_remote_code": True})
config = model.config.__dict__
model_type = config.get("model_type")
image_token_index = config["image_token_id"]
eos = config["eos_token_id"]
for _, m in model.named_modules():
    if hasattr(m, "fused_apply"):
        m.fused_apply = False
        m._compiled_apply = {}

tok = processor.tokenizer
rows = [json.loads(l) for l in Path(TRAIN_JSONL).read_text().splitlines()]

def manual_generate(question, img_path, max_tokens=60):
    messages = [{"role": "user", "content": question}]
    prompt = apply_chat_template(processor, config, messages, add_generation_prompt=True, num_images=1)
    inputs = prepare_inputs(
        processor, images=[img_path], prompts=[prompt],
        image_token_index=image_token_index, resize_shape=None,
    )
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    attention_mask = inputs["attention_mask"]
    data_kwargs = {k: v for k, v in inputs.items() if k not in ["input_ids", "pixel_values", "attention_mask"]}

    generated = []
    for _ in range(max_tokens):
        outputs = model(input_ids, pixel_values, attention_mask, **data_kwargs)
        logits = outputs.logits[0, -1]
        next_token = int(mx.argmax(logits))
        if next_token == eos:
            break
        generated.append(next_token)
        input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)
        attention_mask = mx.concatenate([attention_mask, mx.ones((1, 1))], axis=1)
    return tok.decode(generated, skip_special_tokens=False)

for i in [0, 1]:
    row = rows[i]
    img = row["images"][0]
    out = manual_generate(row["question"], img)
    print(f"=== train 样本 {i} ({Path(img).name}) ===")
    print("真值:", repr(row["answer"]))
    print("输出:", repr(out))
