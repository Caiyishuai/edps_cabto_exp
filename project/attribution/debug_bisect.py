"""二分定位：vision tower vs language model 的反向传播"""
import traceback
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm import load
from mlx_vlm.lora import setup_model_for_training, transform_dataset_to_messages
from mlx_vlm.trainer.datasets import VisionDataset
from datasets import load_dataset

MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-bf16"
DATA = "/Users/yishuaicai/mywork/edps_cabto_exp/outputs/epm_data/parquet/train"

print("loading model ...", flush=True)
model, processor = load(MODEL, processor_config={"trust_remote_code": True})
args = SimpleNamespace(
    full_finetune=False, lora_rank=8, lora_alpha=16, lora_dropout=0.0, train_vision=False
)
model = setup_model_for_training(model, args, None)


def disable_fused_rope(m):
    """禁用 RoPE 的 fused metal kernel（其 vjp 未实现，训练反向传播会报 CustomKernel）"""
    cnt = 0
    for _, module in m.named_modules():
        if hasattr(module, "fused_apply"):
            module.fused_apply = False
            module._compiled_apply = {}
            cnt += 1
    print(f"disabled fused_apply on {cnt} rotary modules", flush=True)


disable_fused_rope(model)

ds = load_dataset(DATA)["train"]
config = model.config.__dict__
ds = transform_dataset_to_messages(ds, config.get("model_type"))
vd = VisionDataset(ds, config, processor)
batch = vd[0]

pixel_values = batch["pixel_values"]
grid_thw = batch["image_grid_thw"]
input_ids = batch["input_ids"]

# 1. vision tower backward
print("=== vision tower backward ===", flush=True)
try:
    def vision_loss(m):
        return m.vision_tower(pixel_values, grid_thw).sum()
    nn.value_and_grad(model, vision_loss)(model)
    print("vision tower backward OK", flush=True)
except Exception:
    print("vision tower backward FAIL:", flush=True)
    traceback.print_exc()

# 2. language model backward (纯文本, 无图)
print("=== language model backward (text-only) ===", flush=True)
try:
    def lm_loss(m):
        out = m.language_model(input_ids, None, mask=None)
        return out.logits.sum()
    nn.value_and_grad(model, lm_loss)(model)
    print("language model backward OK", flush=True)
except Exception:
    print("language model backward FAIL:", flush=True)
    traceback.print_exc()
