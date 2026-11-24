"""决定性诊断：训练好的模型对训练 batch 的 argmax 预测 vs labels"""
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

print("loading model + adapter ...", flush=True)
model, processor = load(MODEL, adapter_path=ADAPTER, processor_config={"trust_remote_code": True})
config = model.config.__dict__
model_type = config.get("model_type")

ds = load_dataset(DATA)["train"]
ds = transform_dataset_to_messages(ds, model_type)
vd = VisionDataset(ds, config, processor)

for i in [0]:
    batch = vd[i]
    input_ids = batch["input_ids"]
    pixel_values = batch["pixel_values"]
    attention_mask = batch["attention_mask"]

    # forward（传递 image_grid_thw 等额外参数）
    kwargs = {k: v for k, v in batch.items() if k not in ["input_ids", "pixel_values", "attention_mask"]}
    outputs = model(input_ids, pixel_values, attention_mask, **kwargs)
    logits = outputs.logits.astype(mx.float32)
    logits_np = np.array(logits[0])

    # labels = shift
    labels = np.array(input_ids[0][1:])

    # 找 assistant 位置，只看 answer 部分
    assistant_id = 77091
    pos = np.where(np.array(input_ids[0]) == assistant_id)[0][0]
    print("assistant 位置:", pos, flush=True)

    # 对 answer 部分（pos+1 之后）的每个位置，看 argmax 是否等于 label
    answer_range = range(pos, pos + 20)
    pred_tokens = []
    label_tokens = []
    for t in answer_range:
        if t >= len(labels):
            break
        pred = int(np.argmax(logits_np[t]))
        label = int(labels[t])
        pred_tokens.append(pred)
        label_tokens.append(label)

    print("模型 argmax 预测 answer:", processor.tokenizer.decode(pred_tokens), flush=True)
    print("真实 answer labels   :", processor.tokenizer.decode(label_tokens), flush=True)
