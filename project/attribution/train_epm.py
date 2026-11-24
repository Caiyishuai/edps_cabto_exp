"""CABTO-EPM 最小切片训练：LoRA 微调 Qwen2.5-VL，从 RGB 图直接输出物体 3D 坐标。

数据：outputs/epm_data/parquet/{train,val}/data.parquet
输出：outputs/epm_data/adapters/adapters.safetensors + adapter_config.json

关键修复：Qwen2.5-VL 的 RoPE 在 Metal 下启用 fused metal kernel（mx.fast.metal_kernel），
其 vjp 未实现，训练反向传播会报 `[Primitive::vjp] Not implemented for CustomKernel`。
解决办法是禁用 fused_apply，让 RoPE 走纯 MLX 实现（前向等价，只是略慢）。
"""
import sys
from types import SimpleNamespace

import mlx.core as mx
import mlx.optimizers as optim
from datasets import load_dataset

# 修复 1：在加载模型前，关闭 RoPE fused metal kernel 的全局开关
import mlx_vlm.models.rope_utils as rope_utils  # noqa: E402

rope_utils._HAS_METAL = False

from mlx_vlm import load  # noqa: E402
from mlx_vlm.lora import setup_model_for_training, transform_dataset_to_messages  # noqa: E402
from mlx_vlm.trainer.datasets import VisionDataset  # noqa: E402
from mlx_vlm.trainer.sft_trainer import TrainingArgs, train  # noqa: E402

MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-bf16"
TRAIN_DATA = "/Users/yishuaicai/mywork/edps_cabto_exp/outputs/epm_px_data/parquet/train"
VAL_DATA = "/Users/yishuaicai/mywork/edps_cabto_exp/outputs/epm_px_data/parquet/val"
OUTPUT = "/Users/yishuaicai/mywork/edps_cabto_exp/outputs/epm_px_data/adapters/adapters.safetensors"

BATCH_SIZE = 1
EPOCHS = 8
LEARNING_RATE = 5e-5
LORA_RANK = 32
LORA_ALPHA = 64
MAX_SEQ_LENGTH = 2048


def disable_fused_rope(m):
    """修复 2：兜底，直接关掉所有 rotary 模块的 fused_apply。"""
    cnt = 0
    for _, module in m.named_modules():
        if hasattr(module, "fused_apply"):
            module.fused_apply = False
            module._compiled_apply = {}
            cnt += 1
    print(f"[fix] disabled fused_apply on {cnt} rotary modules", flush=True)


def main():
    print("loading model ...", flush=True)
    model, processor = load(MODEL, processor_config={"trust_remote_code": True})

    print("setting up LoRA ...", flush=True)
    lo_args = SimpleNamespace(
        full_finetune=False,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        train_vision=False,
    )
    model = setup_model_for_training(model, lo_args, None)
    disable_fused_rope(model)

    config = model.config.__dict__
    model_type = config.get("model_type")

    print("loading datasets ...", flush=True)
    train_ds = load_dataset(TRAIN_DATA)["train"]
    train_ds = transform_dataset_to_messages(train_ds, model_type)
    train_dataset = VisionDataset(train_ds, config, processor)

    val_ds = load_dataset(VAL_DATA)["train"]
    val_ds = transform_dataset_to_messages(val_ds, model_type)
    val_dataset = VisionDataset(val_ds, config, processor)

    n_train = len(train_ds)
    iters = (n_train // BATCH_SIZE) * EPOCHS
    print(f"train samples: {n_train}, epochs: {EPOCHS}, iters: {iters}", flush=True)

    optimizer = optim.Adam(learning_rate=LEARNING_RATE)

    training_args = TrainingArgs(
        batch_size=BATCH_SIZE,
        iters=iters,
        steps_per_report=10,
        steps_per_eval=100,
        steps_per_save=200,
        val_batches=5,
        max_seq_length=MAX_SEQ_LENGTH,
        adapter_file=OUTPUT,
        learning_rate=LEARNING_RATE,
    )

    train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        args=training_args,
        train_on_completions=True,   # loss 只算 assistant 回复（answer）部分
    )
    print("TRAINING_DONE", flush=True)


if __name__ == "__main__":
    main()
