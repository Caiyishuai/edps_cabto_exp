"""清理卡住的下载进程 + 断点续传 bf16 模型（单文件，带进度输出）"""
import os
import signal
import subprocess
import time
import glob

# 1. 杀掉卡住的 snapshot_download 进程（排除自身）
try:
    out = subprocess.run(
        ["pgrep", "-f", "snapshot_download"],
        capture_output=True, text=True, timeout=10,
    ).stdout
    my_pid = os.getpid()
    for pid in out.split():
        pid = pid.strip()
        if not pid.isdigit():
            continue
        pid = int(pid)
        if pid != my_pid:
            try:
                os.kill(pid, signal.SIGKILL)
                print(f"[1] killed stuck pid {pid}", flush=True)
            except ProcessLookupError:
                print(f"[1] pid {pid} already gone", flush=True)
            except Exception as e:
                print(f"[1] skip pid {pid}: {e}", flush=True)
except Exception as e:
    print(f"[1] pgrep failed: {e}", flush=True)

time.sleep(2)

# 2. 清理锁文件
hub = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-VL-3B-Instruct-bf16"
)
for f in glob.glob(os.path.join(hub, "blobs", "*.lock")):
    try:
        os.remove(f)
        print(f"[2] removed lock {f}", flush=True)
    except Exception as e:
        print(f"[2] remove lock failed {f}: {e}", flush=True)

# 3. 断点续传下载缺失的分片
from huggingface_hub import hf_hub_download

print("[3] downloading model-00001-of-00002.safetensors ...", flush=True)
p = hf_hub_download(
    "mlx-community/Qwen2.5-VL-3B-Instruct-bf16",
    "model-00001-of-00002.safetensors",
)
print("DOWNLOADED_OK:", p, flush=True)
