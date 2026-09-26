# CABTO on LIBERO（单臂任务）

把 CABTO 的**单臂技术路线**（参考 `outputs/remaining_cabto` 的 cover / blocks）接到
LIBERO 的单臂基准上。本包位于 `libero_repro/cabto/`，入口 `libero_repro/run_cabto.py`。

## 一句话结论

LIBERO 官方技术栈（robosuite 1.4 + MuJoCo 2.3）在 macOS 26 上跑不起来（见 `libero_repro/README.md`
的三层不兼容链），但 **LIBERO 的 BDDL 任务规格解析是纯 Python**。因此本包把 CABTO 的
`prepare → sample → replay` 技术路线落到 LIBERO 的符号层，**130 个单臂任务全部符号可达**；
物理执行层保留统一接口并文档化边界。

## 技术路线（与 outputs/remaining_cabto 单臂任务对应）

| 阶段 | CABTO 单臂（cover/blocks） | 本包（LIBERO） |
|------|---------------------------|----------------|
| ① BDDL/任务规格 | `project/scenes/cover_kitchen_sort_v2.json` | 130 个 `.bddl` 文件（`cabto/bddl.py`） |
| ② 谓词体系 | `at_source/holding/at_target/empty/home` | `on_region/on_object/in_region/open/closed/turned_on/turned_off/holding/empty`（`cabto/atoms.py`） |
| ③ STRIPS 动作库 | `pick/place/home` 参考模型 | `pick/place_on/place_in/open/close/turn_on/turn_off`（`cabto/library.py`） |
| ④ 形式化 BT 规划 | `formal_bt.ModelLibrary + symbolic_dry_run` | 同（`cabto/planner.py`） |
| ⑤ 代码生成 | `policy_codegen.PolicyProgram` + `API_DOCS` | 同（`cabto/codegen.py`） |
| ⑥ 效果校验 | `transition_check` | 同（`cabto/checker.py`） |
| ⑦ 编排 | `run_remaining_cabto.py`（prepare/sample/replay） | `cabto/pipeline.py` + `run_cabto.py` |

> 设计与过程记录（规划器内核、动作库、谓词映射、130/130 判定依据、两级准确率、FD vs BT）见
> **[`cabto/DESIGN.md`](cabto/DESIGN.md)**。

## 目录结构

```
libero_repro/
├── cabto/
│   ├── __init__.py        # 路径 / suite 注册表
│   ├── bddl.py            # 纯 Python BDDL 解析 + 130 任务枚举
│   ├── atoms.py           # LIBERO 谓词 → CABTO STRIPS 原子 + 每任务问题构造
│   ├── library.py         # 参考 STRIPS 动作库（7 类技能机械落地）
│   ├── planner.py         # 形式化 BT 规划 + 符号可达性验证
│   ├── codegen.py         # 运动 API 契约 + def policy(api) 安全解释器
│   ├── checker.py         # 效果校验（transition_check）
│   ├── llm.py             # 可选本地 mlx-vlm 后端
│   ├── pipeline.py        # prepare / plan / sample / replay 编排
│   └── test_libero_cabto.py
├── run_cabto.py           # CLI 入口
└── README_cabto.md        # 本文档
```

## 谓词映射

LIBERO 的 6 类谓词映射到 CABTO 原子（`On` 按第二参数区分区域/物体）：

| LIBERO 谓词 | CABTO 原子 |
|-------------|-----------|
| `On(o, region)` | `on_region(o, region)` |
| `On(o1, o2)`（叠放） | `on_object(o1, o2)` |
| `In(o, region)` | `in_region(o, region)` |
| `Open(x)` / `Close(x)` | `open(x)` / `closed(x)` |
| `Turnon(x)` / `Turnoff(x)` | `turned_on(x)` / `turned_off(x)` |
| 机器人自身 | `empty()` / `holding(o)` |

关节实体的类别判定靠 fixture 类别：`*_cabinet` + 区域名 `top/middle/bottom_region` → 抽屉；
`microwave` → 门；`flat_stove` → 可开关电器；其余（basket/tray/shelf 等）→ 静态容器。
LIBERO 的 `:init` 不列关节默认闭态，本包为每个 open/close 实体补 `closed(e)`、turnable 补
`turned_off(e)`（若 init 未显式给出，如 `(Open microwave_1)` 则尊重显式开态）。

## 运行

环境：`libero_repro/.venv`（Python 3.11）。依赖（均已装入该 venv）：
`bddl`（BDDL 解析）、`numpy`、`py_trees`、`tabulate`（后两个供 CABTO `formal_bt` 内核）。
自包含数据：130 个 `.bddl` 任务规格已拷入 `libero_repro/bddl_files/`，解析器与 suite 任务表
拷入 `libero_repro/vendor_libero/`——**跑 cabto 不需要 650M 的 LIBERO 完整克隆**（本地若存在
`libero_repro/LIBERO/` 则作为 fallback）。

```bash
cd /Users/yishuaicai/mywork/edps_cabto_exp/libero_repro
PY=.venv/bin/python

# 1. 符号规划验证整个 suite（或 all = 全部 130 任务）
$PY run_cabto.py plan libero_goal --output runs/plan_goal
$PY run_cabto.py plan all --output runs/plan_all

# 2. 构造动作库 + 规划校验
$PY run_cabto.py prepare libero_goal --task open_the_top_drawer_and_put_the_bowl_inside --output runs/prepare

# 3. 代码生成（reference 离线基线；llm 需 mlx-vlm 环境）
$PY run_cabto.py sample libero_goal --task open_the_top_drawer_and_put_the_bowl_inside \
    --models runs/prepare/models.json --backend reference --output runs/sample

# 4. 独立复验
$PY run_cabto.py replay libero_goal --task open_the_top_drawer_and_put_the_bowl_inside \
    --models runs/prepare/models.json --policies runs/sample/policies.json --output runs/replay

# 测试
$PY -m unittest cabto.test_libero_cabto -v
```

**结果：130/130 任务符号可达**（libero_object/spatial/goal/10 各 10，libero_90 共 90）。

## 边界（诚实声明，与 CABTO 上游一致）

1. **符号层完整**：BDDL 解析、谓词规范化、STRIPS 动作库、形式化 BT 规划、符号可达性验证、
   代码生成契约、效果校验器，均在本机完整运行并通过测试。
2. **动作签名由工程指定**：`pick/place_on/place_in/open/close/turn_on/turn_off` 七类技能是
   参考模型；`--refine`/`--backend llm` 只在装有 mlx-vlm 的环境做参考初始化修订，不冒充从零
   发现技能。
3. **`sample` 的 reference 后端是工程参考策略**（离线确定性基线，明确非模型证据）；
   真实模型生成需在 mlx-vlm 环境跑 `--backend llm`。
4. **物理执行（接地）不在本包范围**：真实机器人运动、VLM 打点、碰撞审计需要 LIBERO 官方
   Linux/GPU 栈，或 CABTO 的 MuJoCo 3.10 单臂场景 + LIBERO 物体/关节资产。`pipeline.SymbolicRuntime`
   只做符号转移，不产生物理位移；效果校验是 STRIPS 帧一致，不冒充真机验证。
5. **"push"（推盘子）任务**在符号层建模为 pick+place_on（目标状态一致），物理上"推"与"抓放"
   的接地不同，属建模简化。
6. 所有可达性/一致性均为**有限符号验证**，不证明连续状态空间的普适一致性。
