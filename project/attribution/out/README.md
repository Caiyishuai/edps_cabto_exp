# attribution / out 目录说明

本目录是 CABTO **失败归因实验** 的输出产物目录，包含两类内容：

1. **手动三探针诊断脚本**（`probe_*.py`）跑出的 JSON 与感知过程图；
2. **归因器 V1**（`core/attribution.py`）跑出的统计化归因结果。

---

## 一、做的是什么

CABTO 第三步「跨层精化」存在**归因混淆**：动作执行失败时，无法区分是下面哪个根因——

| 根因 | 含义 | 正确修法 |
|---|---|---|
| **model** | 动作模型 `pre/add/del` 物理不可达或声明错误 | 修动作模型 |
| **policy** | 模型对，但 VLM 生成的策略代码能力不足 | 重新采样代码 |
| **perception** | 模型对、代码对，但 Qwen 打点/定位误差 | 修感知 |

错误的归因会把「好模型」当「坏模型」修，污染动作库。本目录记录的就是**用对照实验（干预）解耦这三个混淆变量**的过程与结论。

核心思想：**换一个变量，看结果变不变**。

```
失败的动作模型
  ├─ 探针1(oracle上限策略) 失败 ────────────────→ model 错（修动作模型）
  └─ 探针1 成功
       ├─ 探针2(VLM策略+oracle定位) 失败 ───────→ policy 弱（修策略）
       └─ 探针2 成功
            ├─ 探针3(VLM策略+qwen定位) 失败 ────→ perception 错（修感知）
            └─ 三路全成功 ─────────────────────→ ok
```

归因器 V1 在此基础上加了**多 seed 采样 + 二项置信区间（Wilson）**，证据不足时判 `abstain`（保持 unresolved，不硬归因）。

---

## 二、涉及代码在哪里

### 归因器 V1（当前主线）

| 文件 | 作用 |
|---|---|
| `project/core/attribution.py` | 归因核心库：三探针、`wilson()` 置信区间、`classify()` 判决、`attribute()` 主入口、`attribute_accepted()` 批量归因 |
| `project/attribution/probe_attribution.py` | 归因器 CLI 入口 |

### 早期手动三探针脚本（归因器 V1 的前身）

| 文件 | 作用 |
|---|---|
| `project/attribution/probe_oracle.py` | 探针1：oracle 上限策略可达性 |
| `project/attribution/probe_vlm.py` | 探针2：VLM 策略 + oracle 定位 |
| `project/attribution/probe_qwen.py` | 探针3：VLM 策略 + qwen 定位，三路归因 |
| `project/attribution/probe_pointing.py` | 打点精度对比（3B/7B/9B） |
| `project/attribution/probe_crossview.py` | 两视角定位不一致诊断 |

### 底层依赖模块

| 文件 | 作用 |
|---|---|
| `project/core/cover_cabto_api.py` | `transition_check()` 一致性判定、`FineCover` 高层环境、`API_DOCS`、`Perception` |
| `project/core/remaining_blocks_api.py` | `FineBlocks` 环境、`BlocksPerception`（qwen 点 + HSV mask） |
| `project/core/single_arm_rebuilt.py` | `RebuiltSingleArm`（oracle 上限策略 `pick/place/home`） |
| `CABTO/exp4_bt_tasks/stage3_cabto/formal_bt.py` | `action_id()` 动作唯一键 |
| `CABTO/exp4_bt_tasks/stage3_cabto/policy_codegen.py` | `PolicyProgram` 策略代码执行器 |
| `project/run_remaining_cabto.py` | 主流程（`sample` / `replay`），归因器未来的接入点 |

---

## 三、目录结构

```
out/
├── cover/                       # 单臂 Cover 场景（早期手动探针输出）
│   ├── probe_oracle.json        #   探针1 结果
│   ├── probe_vlm.json           #   探针2 结果
│   ├── probe_qwen.json          #   探针3 / 三路归因结果
│   ├── probe_pointing.json      #   打点精度对比
│   ├── pointing/                #   打点过程图
│   ├── qwen/                    #   qwen 感知过程图 + perception.json
│   └── crossview/               #   两视角定位图
│
└── blocks/                      # 单臂 Blocks 场景
    ├── probe_oracle.json
    ├── probe_vlm.json
    ├── probe_qwen.json
    ├── probe_pointing.json
    ├── pointing/
    ├── qwen/
    └── attributor/              # 归因器 V1 输出（新增）
        ├── attribution.json     #   统计化归因结果
        ├── oracle/              #   探针2 感知过程图
        └── qwen/                #   探针3 感知过程图
```

---

## 四、如何复现

### 归因器 V1（三路，含感知层）

```bash
cd /Users/yishuaicai/mywork/edps_cabto_exp

# blocks 全动作库三路归因（加载 Qwen2.5-VL-3B 感知 VLM）
python3 project/attribution/probe_attribution.py blocks

# 只判 model/policy 两层（跳过感知，不需要 VLM，快）
python3 project/attribution/probe_attribution.py blocks --no-perception

# 单个动作 / 调参
python3 project/attribution/probe_attribution.py blocks \
    --action-id "pick(object=red,support=blue)" \
    --seeds 0 1 2 --threshold 0.5 --confidence 0.8
```

### 早期手动探针

```bash
python3 project/attribution/probe_oracle.py blocks      # 探针1
python3 project/attribution/probe_vlm.py   blocks      # 探针2
python3 project/attribution/probe_qwen.py  blocks      # 探针3（三路）
```

---

## 五、关键结论记录

1. **感知瓶颈的三层**（详见 `project/attribution/README.md`）：
   - 解析协议 bug（`parse_pixel` 只认 JSON）——已修，收益最大；
   - 颜色相近物体难辨（potato 棕 vs shrimp 橙，打偏 76px）——换「认知一致」物体（青椒绿）解决；
   - 换大模型不解决可辨性问题（7B 认错、9B 退化）。

2. **归因器 V1 抓到的真实感知缺陷**（`blocks` place 动作）：第一次跑三路时两个 `place` 被判 `perception`，根因是归因器没在抓取前 `prepare_supports()` 缓存支撑块位置，被举起的 red 遮挡了 blue。修复后 `place` 从 `perception` 回到 `ok`。这一步直接演示了归因器「把失败定位到具体层」的价值。

3. **统计化代价**：单 seed（k=1）下 0/1 失败的置信区间上界仍约 0.62，越不过 0.5 阈值，会判 `abstain` 而非定罪；建议 `k≥3`（默认）。
