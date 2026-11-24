"""手动补齐 bf16 模型的快照软链接（model-00001 已手动下载到 blobs）"""
import os
import signal
import subprocess
import glob

# 1. 杀卡住的 snapshot_download
try:
    out = subprocess.run(["pgrep", "-f", "snapshot_download"],
                         capture_output=True, text=True).stdout
    for p in out.split():
        p = p.strip()
        if not p.isdigit():
            continue
        p = int(p)
        if p == os.getpid():
            continue
        try:
            os.kill(p, signal.SIGKILL)
            print("killed snapshot_download", p, flush=True)
        except Exception:
            pass
except Exception as e:
    print("pgrep skip", e, flush=True)

# 2. 定位快照目录
hub = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-VL-3B-Instruct-bf16"
)
snaps = sorted(glob.glob(os.path.join(hub, "snapshots", "*")))
snap = snaps[-1]
print("snapshot dir:", snap, flush=True)

# 3. 创建 model-00001 软链接
link = os.path.join(snap, "model-00001-of-00002.safetensors")
blob = "1798b28d924248ef1cec6aa62fdfbd29db97d3c087213b0119a653d5c3e48f28"
target = "../../blobs/" + blob
if os.path.islink(link) or os.path.exists(link):
    print("symlink exists:", os.readlink(link), flush=True)
else:
    os.symlink(target, link)
    print("created symlink:", link, "->", target, flush=True)

# 4. 列出快照目录全部文件，确认齐全
print("=== snapshot 内容 ===", flush=True)
for f in sorted(os.listdir(snap)):
    p = os.path.join(snap, f)
    if os.path.islink(p):
        real = os.path.realpath(p)
        size = os.path.getsize(real) if os.path.exists(real) else -1
        print(f, "->", size, flush=True)
    else:
        print(f, "->", os.path.getsize(p), flush=True)
