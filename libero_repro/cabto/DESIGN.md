# 设计记录：CABTO 单臂技术路线在 LIBERO 上的落地

> 本文是 `libero_repro/cabto/` 实现的**设计与过程记录**，回答「做了什么、怎么做的、判定依据是什么」。
> 快速上手见同目录上层的 [`../README_cabto.md`](../README_cabto.md)。

## 1. 一句话定位

LIBERO 官方物理栈（robosuite 1.4 + MuJoCo 2.3）在 macOS 26 上无法运行，但 **BDDL 任务规格解析是纯 Python**。
因此把 CABTO 的 `prepare → sample → replay` 单臂技术路线（参考 `outputs/remaining_cabto` 的 cover/blocks）
落到 LIBERO 的**符号层**：BDDL 谓词 → STRIPS 动作库 → 形式化 BT 规划 → 符号重放判定。**130/130 任务符号可达。**

## 2. 架构与数据流

```
130 个 *.bddl
   │  bddl.py（纯 Python 解析，importlib 单独加载 libero/envs/bddl_utils.py）
   ▼
Task（problem_name / objects / fixtures / init / goal / regions / language）
   │  atoms.py（谓词规范化 + 关节实体判定 + 默认闭态补全）
   ▼
Problem（init / goal / atoms / 抽屉/微波炉/灶具实体 / pick 对象 / place 目标）
   │  library.py（7 类技能模板，机械落地到具体对象/区域名）
   ▼
STRIPS 动作库 [pick / place_on / place_in / open / close / turn_on / turn_off]
   │  planner.py → formal_bt.ModelLibrary.build(init, goal)
   ▼
BTExpansion 内核（run_algorithm_selTree）→ 形式化行为树
   │  formal_bt.symbolic_dry_run(bt, init, goal, 256)
   ▼
「goal ⊆ 最终状态」判定 → 130/130 可达
```

## 3. 规划器详解

**规划器不是新写的，是复用 CABTO 自带的 BTExpansion 内核。**

- 算法类：`BTExpAlgorithm`，位于 `BTExpansion-demo/src/bt_expansion/algorithm.py`，注释标注「本文所提出的完备规划算法」。
- 入口：`run_algorithm_selTree(start, goal, actions)`。
- 思想（CABTO 论文的 HOBTEA/OBTEA 那套）：**从 goal 条件叶子出发反向扩展**——初始行为树只有一个 goal 条件节点，
  反复把未满足的条件叶子，用「哪些动作的 add 能让它成立」反向展开成动作节点，递归直到所有条件被满足，
  最终**选一棵代价最小的树**（`selTree` = 选树变体）。

调用链（我写的适配层 → 内核）：

```
planner.plancheck()
  └─ formal_bt.ModelLibrary(models).build(init, goal)
       └─ FormalBTBuilder.build()
            └─ BTExpAlgorithm().clear()
            └─ BTExpAlgorithm().run_algorithm_selTree(start, goal, actions)
                 # actions = 内核的 Action(name, pre, add, del_set)，由动作库字典翻译而来
       └─ 返回 FormalBT
  └─ formal_bt.symbolic_dry_run(bt, init, goal, 256)   # STRIPS 重放，确认真到 goal
```

- `formal_bt` 是 CABTO 自己的封装层（`CABTO/exp4_bt_tasks/stage3_cabto/formal_bt.py`），直接 import 复用，内核一行未改。
- `validate_model` 校验每个动作 `pre∩add=∅` 且 `add∩del=∅`；`action_id` = `name(k=v,...)`。

## 4. 谓词映射

LIBERO 的 6 类谓词映射到 CABTO 原子（`On` 按第二参数区分区域/物体）：

| LIBERO 谓词 | 含义 | CABTO 原子 |
|-------------|------|-----------|
| `On(o, region)` | 物体在表面区域 | `on_region(o, region)` |
| `On(o1, o2)` | 物体叠在物体上 | `on_object(o1, o2)` |
| `In(o, region)` | 物体在容器内 | `in_region(o, region)` |
| `Open(x)` / `Close(x)` | 抽屉/门开/关 | `open(x)` / `closed(x)` |
| `Turnon(x)` / `Turnoff(x)` | 电器开/关 | `turned_on(x)` / `turned_off(x)` |
| 机器人自身 | 手爪状态 | `empty()` / `holding(o)` |

**关节实体判定**（`atoms.py`），靠 fixture 类别：
- `*_cabinet` + 区域名 `top/middle/bottom_region` → 抽屉（可 open/close）
- `microwave` → 门（可 open/close，`In(o, *_heating_region)` 需先 `open(microwave_1)`）
- `flat_stove` → 可 turnon/turnoff
- 其余（basket / wooden_tray / desk_caddy / two_layer_shelf 等）→ 静态容器，无关节

**默认闭态补全**：LIBERO 的 `:init` 只列物体摆放，不列关节默认态。构造 init 时，为每个 open/close
实体补 `closed(e)`、turnable 实体补 `turned_off(e)`（若 init 显式给出如 `(Open microwave_1)` 则尊重）。

实测全量谓词分布（130 任务扫描）：`on` 708 / `in` 64 / `open` 29 / `close` 14 / `turnon` 11 / `turnoff` 1。

## 5. 参考动作库（`library.py`）

**7 类技能模板（工程指定 pre/add/del）+ 机械落地**，不是 LLM 生成：

| program_kind | 模板 |
|---|---|
| `pick(o)` | `empty()` → `holding(o)`；del `empty()` + o 的全部位置原子 |
| `place_on(o,t)` | `holding(o)` → `on_region/on_object(o,t)` + `empty()`；del `holding(o)` |
| `place_in(o,r)` | `holding(o)`（+`open(e)` 若抽屉/微波炉）→ `in_region(o,r)` + `empty()`；del `holding(o)` |
| `open(e)` | `closed(e)`+`empty()` → `open(e)`；del `closed(e)` |
| `close(e)` | `open(e)`+`empty()` → `closed(e)`；del `open(e)` |
| `turn_on(e)` | `turned_off(e)`+`empty()` → `turned_on(e)`；del `turned_off(e)` |
| `turn_off(e)` | `turned_on(e)`+`empty()` → `turned_off(e)`；del `turned_on(e)` |

要点：`pick` 的 `del` 删除该物体的**所有可能位置原子**（source + 目标），因为规划器不知道它在哪、一次只真一个，全删安全。
动作字典结构：

```json
{"name": "pick", "args": {"object": "milk_1"},
 "pre": ["empty()"], "add": ["holding(milk_1)"],
 "del": ["empty()", "on_region(milk_1,floor_target_object_region)", "in_region(milk_1,basket_1_contain_region)"],
 "program_kind": "pick", "description": "..."}
```

## 6. 完整性 / 可达性判定（130/130 的依据）

对每个任务：`build_problem` → `reference_models` → `plancheck`。

- `ModelLibrary.build(init, goal)` 建出形式化行为树；
- **`symbolic_dry_run(bt, init, goal, 256)` 在复制的符号状态上按树的 tick 顺序重放一遍 STRIPS 转移，
  返回 `goal ⊆ 最终状态` 与否**——这就是「130/130 可达」的判定依据。

结果（`runs/plan_all/summary.json` + 单测 `test_all_130_reachable`）：

| suite | 可达 |
|---|---|
| libero_object / spatial / goal / 10 | 各 10/10 |
| libero_90 | 90/90 |
| **合计** | **130/130** |

**判定语义必须说清**：这只证明「参考库够完整，能从 init 规划到 goal」，是**有限符号验证**，
不证明物理可执行、不证明每个 add/del 与物理一致。

## 7. 两级「准确率」判定

LIBERO **不提供官方动作模型**，只提供两样真值：init/goal 状态 + 谓词求值器（`predicates/`、`object_states/`，
从 `sim.data.body_xpos`/接触/包容/关节状态算谓词真假）。判 LLM 生成动作库的准确率分两级：

1. **符号层：可达性/完整性**（现在就能判）——用动作库 + init/goal，看能否规划到 goal。
   **必要但不充分**：能规划 ≠ add/del 物理正确。
2. **物理层：效应保真度**（需模拟器）——真实执行一次，用谓词求值器读真实 before→after，
   与 LLM 声明的 add/del 逐条对（即 `transition_check`）。**这才是真正的「准确率」**，卡在模拟器层。

## 8. 为什么用 BT 扩展而不是 Fast Downward

FD 输出**线性总序计划**，转 BT 只能得到 **Sequence**（弱反应性：中间一步失败→整棵 FAILURE→停机）；
BT 扩展直接产出**带 Fallback/Selector 的反应式树**（失败→回退分支重试）。详见对话中的
「线性 Sequence vs Fallback 反应性」对比。仓库里已有 `BTExpansion-demo/src/pddl_adapter/`
（PDDL→BT，内部求解器也是 BT 扩展内核，非 FD）。结论：FD 适合当**可解性/最优性 oracle**，
不适合替换 CABTO 的 BT 扩展内核。

## 9. 文件清单

```
libero_repro/
├── run_cabto.py            # CLI：plan / prepare / sample / replay
├── README_cabto.md         # 快速上手 + 边界
├── bddl_files/             # 自包含：130 个 .bddl 任务规格（从 LIBERO 拷出，已入库）
├── vendor_libero/          # 自包含：bddl_utils.py 解析器 + libero_suite_task_map.py
└── cabto/
    ├── __init__.py         # 路径 / suite 注册表 / vendor fallback
    ├── bddl.py             # 纯 Python BDDL 解析 + 130 任务枚举
    ├── atoms.py            # 谓词→原子 + 关节实体判定 + 默认闭态
    ├── library.py          # 参考动作库（7 技能模板 + 机械落地）
    ├── planner.py          # formal_bt 封装 + plancheck
    ├── codegen.py          # 运动 API 契约 + def policy(api) 解释器
    ├── checker.py          # transition_check 效果校验
    ├── llm.py              # 可选 mlx-vlm 后端
    ├── pipeline.py         # prepare / plan / sample / replay 编排 + SymbolicRuntime
    ├── test_libero_cabto.py # 15 项单测
    └── DESIGN.md           # 本文档
```

## 10. 边界（诚实声明）

1. 符号层完整可跑；物理接地（真机/VLM 打点/碰撞审计）需 Linux/GPU 栈或 CABTO MuJoCo 3.10 场景 + LIBERO 资产，未做。
2. 动作签名由工程指定，`--backend reference` 是工程参考策略（非模型证据）；真实模型需 mlx-vlm 环境 `--backend llm`。
3. "push" 任务符号层建模为 pick+place_on（目标状态一致，物理推≠抓放）。
4. 所有可达性/一致性均为**有限符号验证**，不证明连续状态空间的普适一致性。
