# CABTO 归因探针（Attribution Probes）

针对 CABTO 第 2 步「VLM 实现动作 + 查一致性」的**归因混淆**问题——执行失败时无法
区分「动作模型 pre 错了（物理不可达）」还是「策略实现能力不足（可达但策略做不到）」。

核心思想：**用对照实验（干预）解耦混淆变量**。本目录的脚本，用底层
`RebuiltSingleArm` 内置的 oracle 上限策略（直接用真值位姿的 pick/place/home）做
「可达性探针」，把失败二分类成「pre 错 / 物理可达」。

## 脚本

| 文件 | 说明 | 状态 |
|---|---|---|
| `probe_oracle.py` | 探针 1：oracle 上限可达性（零模型、零训练、零 VLM） | ✅ 已跑通 |
| `probe_vlm.py` | 探针 2：VLM 策略 + oracle 定位，与探针 1 对照解耦「策略弱」 | ✅ 已跑通 |
| `probe_qwen.py` | 探针 3：VLM 策略 + qwen 定位，三路对照解耦「感知错」 | ✅ 已跑通 |

## 探针 1 用法

```bash
cd project
PY=/Users/yishuaicai/.workbuddy/binaries/python/envs/cabto/bin/python

$PY -B attribution/probe_oracle.py cover            # cover，seed=0（无 jitter）
$PY -B attribution/probe_oracle.py blocks           # blocks
$PY -B attribution/probe_oracle.py cover --seeds 0 1 2   # 多 seed（有扰动）
$PY -B attribution/probe_oracle.py blocks --output out/myblocks
```

## 归因逻辑

对每个动作模型，用 oracle 上限策略执行，检查 STRIPS 转移一致性：

- **oracle 上限也失败** → `pre 错`（物理不可达，该修动作模型）
- **oracle 上限成功** → `pre 对`（物理可达，后续与 VLM 策略对照才判「策略弱」）

一致性检查用 `core.cover_cabto_api.transition_check`，关键违规项：

| 违规 | 含义 | 归因指向 |
|---|---|---|
| `missing_pre` | 前置没满足 | 前置状态构造错 |
| `missing_add` | 声称的 add 没达成 | **add 效应错** |
| `remaining_del` | 声称的 del 没消失 | **del 效应错** |
| `missing_preserved_pre` | pre 中该保持的谓词消失 | **del 漏了某项** |

> 注意：`missing_preserved_pre` 是必须查的。早期只查 add/del，漏掉了「del 漏删」
> 这类错误（如 place 的 del 漏了 `holding(obj)`），会把坏模型误判为可达。

## 已验证

- **标准参考模型**（cover 7 个动作 / blocks 5 个动作）：全部 `effect_ok=True`，
  更严格的检查未因碰撞噪声产生误报。
- **注入坏模型**（给 pick 的 add 塞入物理不可能的 `at_target`；给 place 的 del 漏掉
  `holding`）：均被正确判为 `effect_ok=False`，且 violations 精确指出问题谓词。

## 探针 2 用法

```bash
$PY -B attribution/probe_vlm.py cover
$PY -B attribution/probe_vlm.py blocks
```

对每个动作模型同时跑两条路，输出归因：

| 组合 | 归因 |
|---|---|
| probe1(oracle) 失败 | pre 错（物理不可达，修动作模型） |
| probe1 成功 & probe2(vlm) 失败 | 策略弱（代码能力不足，修策略） |
| 两者都成功 | OK（若之后换 qwen 定位失败 → 感知错，探针 3 处理） |

## 已验证

- **探针 2 标准策略**：cover/blocks 全部动作 `OK(动作模型+策略都通)`。
- **探针 2 坏策略**（给 pick 漏掉 `close_gripper`）：正确判为 `策略弱`，
  `missing_add=['holding(yellow)']` + `RuntimeError: Lift requires closed gripper`。

## 探针 3 用法

```bash
$PY -B attribution/probe_qwen.py cover
$PY -B attribution/probe_qwen.py blocks   # 需加载 Qwen-VL，较慢
```

三路归因决策树：

| 组合 | 归因 |
|---|---|
| p1(oracle) 失败 | pre 错 |
| p1 成功 & p2(vlm+oracle) 失败 | 策略弱 |
| p1 成功 & p2 成功 & p3(vlm+qwen) 失败 | 感知错 |
| 三路全成功 | OK |

## 探针 3 实测结果（关键发现）

cover 与 blocks 的 pick/place 全部归因「感知错」，home 全通。瓶颈 **100% 在感知**，
不在策略、不在动作模型。且感知失败有两种模式：

| 失败模式 | 异常 | 含义 |
|---|---|---|
| 输出格式不稳定 | `No JSON object in model response` | 模型没按 JSON 返回坐标 |
| 定位不一致 | `Cross-view xy disagreement 0.19~0.26m > 0.025m` | 两视角打点相差 20+ cm |

## 下一步（未实现）

1. 把三路归因接入 `feedback.py` / `loop.py`，替代「全 unresolved」的诊断，
   让 `candidate_model_patch` 变成有证据的定向修订。
