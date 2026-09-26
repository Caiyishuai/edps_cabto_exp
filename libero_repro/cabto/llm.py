"""可选：本地 mlx-vlm 模型后端（真实模型证据，仅在装有 mlx-vlm 的环境可用）。

libero_repro/.venv 不含 mlx-vlm；如需真实 LLM/VLM 提案与打点，请在装有 mlx-vlm 的
环境（如 /Users/yishuaicai/.workbuddy/binaries/python/envs/cabto）运行。本模块延迟导入，
缺依赖时给出明确错误而非静默降级。
"""
from __future__ import annotations

import time


class LocalModel:
    """每次调用按精确 model_id 实例化；raw 提示/回复由调用方持久化。"""

    def __init__(self, model_id: str = "mlx-community/Qwen3.5-9B-4bit"):
        try:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
        except ImportError as exc:
            raise ImportError(
                "缺少 mlx-vlm：真实模型提案需在装有 mlx-vlm 的环境运行（如 cabto venv）；"
                "符号层请使用 --backend reference。"
            ) from exc
        self.model_id = model_id
        self.model, self.processor = load(model_id)
        self.config = load_config(model_id)

    def call(self, text, images=(), max_tokens=2400, temperature=0.0):
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        prompt = apply_chat_template(self.processor, self.config, text,
                                     num_images=len(images), enable_thinking=False)
        started = time.perf_counter()
        args = {"max_tokens": max_tokens, "temperature": temperature, "verbose": False}
        if images:
            args["image"] = [str(p) for p in images]
        output = generate(self.model, self.processor, prompt, **args)
        return {
            "model_id": self.model_id, "text": text,
            "image_files": [str(p) for p in images],
            "raw": output.text if hasattr(output, "text") else str(output),
            "wall_seconds": time.perf_counter() - started,
        }
