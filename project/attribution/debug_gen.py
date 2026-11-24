"""定位 generation 崩溃点：teacher-forcing 下第一个 answer token 的 logits top-k"""
import numpy as np

import mlx_vlm.models.rope_utils as rope_utils
rope_utils._HAS_METAL = False

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.lora import transform_dataset_to_messages
from mlx_vlm.trainer.datasets import VisionDataset
from datasets import load_dataset

MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-bf16"
ADAPTER = "/Users/yishuaicai/mywork/edps_cabto_exp/outputs/epm_px_data/adapters"
DATA = "/Users/yishuaicai/mywork/edps_cabto_exp/outputs/epm_px_data/parquet/train"

model, processor = load(MODEL, adapter_path=ADAPTER, processor_config={"trust_remote_code": True})
config = model.config.__dict__
model_type = config.get("model_type")
for _, m in model.named_modules():
    if hasattr(m, "fused_apply"):
        m.fused_apply = False
        m._compiled_apply = {}

ds = load_dataset(DATA)["train"]
ds = transform_dataset_to_messages(ds, model_type)
vd = VisionDataset(ds, config, processor)
batch = vd[0]

input_ids = batch["input_ids"]
pixel_values = batch["pixel_values"]
attention_mask = batch["attention_mask"]
kwargs = {k: v for k, v in batch.items() if k not in ["input_ids", "pixel_values", "attention_mask"]}

outputs = model(input_ids, pixel_values, attention_mask, **kwargs)
logits = np.array(outputs.logits[0].astype(mx.float32))

tok = processor.tokenizer
ids = np.array(input_ids[0])
assistant_id = 77091
pos = np.where(ids == assistant_id)[0][0]

# 第一个 answer token 由 logits[pos] 预测（预测 pos+1 位置）
print("=== 第一个 answer token 的 top-10 预测（logits[pos] 预测位置 pos+1）===", flush=True)
first_logits = logits[pos]
topk = np.argsort(first_logits)[::-1][:10]
for t in topk:
    print(f"  {t}: {tok.decode([t])!r}  logit={first_logits[t]:.2f}", flush=True)
print("真实第一个 answer token:", repr(tok.decode([ids[pos+1]])), flush=True)

# 第二个 answer token（logits[pos+1] 预测 pos+2）
print("\n=== 第二个 answer token 的 top-5（logits[pos+1]）===", flush=True)
second_logits = logits[pos+1]
topk2 = np.argsort(second_logits)[::-1][:5]
for t in topk2:
    print(f"  {t}: {tok.decode([t])!r}  logit={second_logits[t]:.2f}", flush=True)
print("真实第二个 answer token:", repr(tok.decode([ids[pos+2]])), flush=True)
