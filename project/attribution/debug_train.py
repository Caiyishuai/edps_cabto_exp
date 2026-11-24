"""精确定位训练前向/反向报错的位置"""
import traceback
from types import SimpleNamespace

import mlx.core as mx

from mlx_vlm import load
from mlx_vlm.lora import setup_model_for_training, transform_dataset_to_messages
from mlx_vlm.trainer.datasets import VisionDataset
from mlx_vlm.trainer.sft_trainer import vision_language_loss_fn
from datasets import load_dataset

MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-bf16"
DATA = "/Users/yishuaicai/mywork/edps_cabto_exp/outputs/epm_data/parquet/train"

print("loading model ...", flush=True)
model, processor = load(MODEL, processor_config={"trust_remote_code": True})

print("setting up LoRA ...", flush=True)
args = SimpleNamespace(
    full_finetune=False, lora_rank=8, lora_alpha=16, lora_dropout=0.0, train_vision=False
)
model = setup_model_for_training(model, args, None)

print("loading dataset ...", flush=True)
ds = load_dataset(DATA)["train"]
config = model.config.__dict__
model_type = config.get("model_type")
ds = transform_dataset_to_messages(ds, model_type)
vd = VisionDataset(ds, config, processor)
batch = vd[0]
print("batch keys:", list(batch.keys()), flush=True)
print("pixel_values shape:", batch["pixel_values"].shape, flush=True)
print("input_ids shape:", batch["input_ids"].shape, flush=True)

print("=== forward ===", flush=True)
try:
    loss = vision_language_loss_fn(model, batch)
    mx.eval(loss)
    print("forward OK, loss =", loss, flush=True)
except Exception:
    traceback.print_exc()

print("=== backward ===", flush=True)
try:
    import mlx.nn as nn
    loss_fn = lambda m, b: vision_language_loss_fn(m, b)
    grad_fn = nn.value_and_grad(model, loss_fn)
    lvalue, grad = grad_fn(model, batch)
    mx.eval(lvalue, grad)
    print("backward OK, loss =", lvalue, flush=True)
except Exception:
    traceback.print_exc()
