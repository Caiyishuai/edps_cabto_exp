import os
import glob

# Must be set BEFORE importing libero (it reads this at import time)
os.environ["LIBERO_CONFIG_PATH"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".libero"
)

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

bddl_dir = get_libero_path("bddl_files")
bddl_files = sorted(glob.glob(os.path.join(bddl_dir, "libero_spatial", "*.bddl")))
print("example bddl:", bddl_files[0])

env = OffScreenRenderEnv(
    bddl_file_name=bddl_files[0],
    camera_heights=128,
    camera_widths=128,
)
env.seed(0)
obs = env.reset()
print("obs keys:", sorted(obs.keys()))
print("agentview_image:", obs["agentview_image"].shape, obs["agentview_image"].dtype)

for i in range(5):
    obs, reward, done, info = env.step([0.0] * 7)
print("5 steps done, reward=", reward, "done=", done)

env.close()
print("ENV_OK")
