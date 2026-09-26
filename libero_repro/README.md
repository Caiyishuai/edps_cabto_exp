# LIBERO 复现（macOS）—— 记录与结论

> 目标：在这台 Mac 上复现 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
> （终身机器人学习基准，基于 robosuite + MuJoCo）。

## 一句话结论

**LIBERO 官方技术栈在这台机器上无法运行。** 根因是三层连锁的「旧基准 × 新 macOS」不兼容，
不是装错依赖、也不是缺某个包能补上的。

> **进展（2026-09）**：虽然官方物理环境跑不起来，但 LIBERO 的 **BDDL 任务规格解析是纯 Python**，
> 因此已把 CABTO 的单臂技术路线（`outputs/remaining_cabto` 的 cover/blocks）落到 LIBERO 的
> **符号层**：130 个单臂任务全部符号可达。见 **[README_cabto.md](README_cabto.md)**，
> 代码在 `cabto/`，入口 `run_cabto.py`。

---

## 机器环境

| 项 | 值 |
|---|---|
| 芯片 | Apple Silicon（`arm64`） |
| 系统 | macOS 26.5.2（BuildVersion 25F84） |
| Python（uv 管理） | 3.11.15 |

> 关键背景：macOS 26 已经**移除了 OpenGL 二进制**（`/System/Library/Frameworks/OpenGL.framework`
> 目录还在，但里面的 dylib 只剩 stub）。这是后面所有问题的起点。

---

## 目录结构

```
libero_repro/
├── .venv/                 # 独立虚拟环境（Python 3.11.15，uv 创建）
├── .libero/config.yaml    # LIBERO 路径配置（通过 LIBERO_CONFIG_PATH 指定）
├── LIBERO/                # 官方仓库（git clone --depth 1）
├── datasets/              # 数据集目录（未下载）
├── test_env.py            # 最小环境创建 + 渲染诊断脚本
└── README.md              # 本文档
```

## 已完成的工作

1. 新建文件夹 `libero_repro/`，并在其内创建独立 venv：`.venv/`。
2. 用 `uv` 锁定 Python 3.11.15（本机缓存，无需联网下载）。
3. 克隆官方 LIBERO 仓库到 `LIBERO/`。
4. 安装依赖（numpy / mujoco / robosuite / bddl / gym / opencv 等）。
5. 绕过 LIBERO `setup.py` 的 namespace-package 缺陷（见下），使 `import libero` 可用。
6. 定位并确认了最终阻塞点（见下）。

### 安装时的两个坑（已解决）

- **LIBERO 的 `setup.py` 用 `find_packages()` 但包是 namespace 结构**（`libero/` 下没有
  `__init__.py`），导致 `pip/uv` 构建出的 wheel 是空的。改用 legacy editable 模式安装：
  ```bash
  uv pip install --python .venv/bin/python -e ./LIBERO --config-settings editable_mode=compat
  ```
- **LIBERO 首次 import 会交互式询问数据集路径**。通过 `LIBERO_CONFIG_PATH` 环境变量指向
  `.libero/` 并预置 `config.yaml` 规避（`test_env.py` 里已设置）。

---

## 阻塞点：三层不兼容链

### 第 1 层：MuJoCo 2.3.x 在 macOS 26 上无法 import

LIBERO 官方 `requirements.txt` 钉死 `robosuite==1.4.0`，而 robosuite 1.4.0 依赖
**MuJoCo 2.3.x**。MuJoCo 2.3.7 在 `import mujoco` 时直接崩溃：

```
OSError: dlopen(/System/Library/OpenGL.framework/OpenGL, 0x0006):
  '/System/Library/OpenGL.framework/OpenGL' (no such file)
```

MuJoCo 2.3.x 的 macOS 渲染依赖 OpenGL（CGL），macOS 26 已移除 OpenGL。

### 第 2 层：robosuite 1.4.0 与 MuJoCo 3.x 不兼容

换到 **MuJoCo 3.14**（在 macOS 26 上 `import mujoco` 正常，因为新版不再依赖 OpenGL）后，
robosuite 1.4.0 在创建环境时崩溃：

```
File ".../robosuite/utils/binding_utils.py", line 521, in get_joint_qpos_addr
    assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
AssertionError
```

根因：MuJoCo 3.x 把 `mjtJoint` 从 `enum.IntEnum`（int 子类）改成了**自定义枚举类**
（`mujoco._enums.mjtJoint`，非 int 子类）。于是：

```python
np.int32(3) == mujoco.mjtJoint.mjJNT_HINGE   # True
np.int32(3) in (mjJNT_HINGE, mjJNT_SLIDE)     # False  ← 这里炸了
```

robosuite 1.4.0 里大量这种比较，修一处还会有很多处。

### 第 3 层：LIBERO 与 robosuite 1.5.x 不兼容

robosuite **1.5.2** 官方已支持 MuJoCo 3.x（`mujoco>=3.3.0,<3.10`），但 LIBERO 是给
robosuite 1.4.0 写的，一 import 就崩：

```
ModuleNotFoundError: No module named 'robosuite.environments.manipulation.single_arm_env'
```

robosuite 1.5 重构了内部模块结构。

### 结论

三条路全部堵死：

| 组合 | 结果 |
|---|---|
| robosuite 1.4.0 + MuJoCo 2.3.x（官方钉死） | MuJoCo 无法 import（OpenGL 被移除） |
| robosuite 1.4.0 + MuJoCo 3.x | robosuite 枚举比较崩溃 |
| robosuite 1.5.2 + MuJoCo 3.x | LIBERO 模块导入崩溃 |

即：**没有任何「直接 pip 装出来就能跑」的版本组合**。

---

## 可行的路径

1. **Linux + NVIDIA GPU 机器（推荐，唯一官方支持路径）**：在 Ubuntu 上用官方流程
   `python=3.8` + `torch==1.11+cu113` + `mujoco 2.3.x` + `robosuite==1.4.0` 即可完整复现
   （含训练）。
2. **云 GPU / 远程 Linux 服务器**：同上。
3. **投入工程去 patch（高风险，未知工作量）**：
   - 方向 A：把 robosuite 1.4.0 适配到 MuJoCo 3.x（改枚举比较等，需大量排查）；或
   - 方向 B：把 LIBERO 适配到 robosuite 1.5.2（改模块导入 + 内部 API）。
4. **LeRobot**：HuggingFace LeRobot 官方也集成了 LIBERO，但其 LIBERO 集成同样
   **要求 Linux**（`sys_platform == 'linux'`），Mac 上不可用。

> 注：本工作区的 CABTO 项目能在 Mac 上跑，是因为它用的是 **MuJoCo 3.10**（新版已不依赖
> OpenGL），而 LIBERO 被钉死在 MuJoCo 2.3.x 这条老线上。

---

## 复现这些诊断

```bash
cd /Users/yishuaicai/mywork/edps_cabto_exp/libero_repro
.venv/bin/python test_env.py
```

`test_env.py` 会创建 `OffScreenRenderEnv` 并尝试 reset，当前会停在上述某一层错误。

### 如需重置环境

```bash
cd /Users/yishuaicai/mywork/edps_cabto_exp/libero_repro
rm -rf .venv
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r <依赖> ...
```

---

## 附录：当前 venv 关键包版本

| 包 | 版本 |
|---|---|
| Python | 3.11.15 |
| mujoco | 3.14.0 |
| robosuite | 1.5.2 |
| numpy | 1.26.4 |
| bddl | 1.0.1 |
| gym | 0.25.2 |
| libero | 0.1.0（editable） |
