# CABTO 归因探针与感知瓶颈调查

针对 CABTO 第 2 步「VLM 实现动作 + 查一致性」的**归因混淆**问题——执行失败时无法
区分「动作模型 pre 错了（物理不可达）」还是「策略实现能力不足（可达但策略做不到）」
还是「感知错了（定位不准）」。

核心思想：**用对照实验（干预）解耦混淆变量**。用底层 `RebuiltSingleArm` 内置的
oracle 上限策略（直接用真值位姿的 pick/place/home）做「可达性探针」，再逐层换
策略/感知做对照，把失败归因到「pre 错 / 策略弱 / 感知错」三选一。

---

## 一、探针脚本

| 文件 | 作用 | 状态 |
|---|---|---|
| `probe_oracle.py` | 探针 1：oracle 上限可达性（零模型、零训练、零 VLM） | ✅ |
| `probe_vlm.py` | 探针 2：VLM 策略 + oracle 定位，解耦「策略弱」 | ✅ |
| `probe_qwen.py` | 探针 3：VLM 策略 + qwen 定位，三路归因 | ✅ |
| `probe_pointing.py` | 打点精度对比（3B / 7B / 9B），绕过解析直接测打点 | ✅ |
| `probe_crossview.py` | 两视角定位不一致诊断（拆解 cross-view 来源） | ✅ |

## 二、三路归因决策树

```
失败的动作模型
  ├─ 探针1(oracle上限) 失败 ────────────────→ pre 错（修动作模型）
  └─ 探针1 成功
       ├─ 探针2(VLM策略+oracle定位) 失败 ────→ 策略弱（修策略）
       └─ 探针2 成功
            ├─ 探针3(VLM策略+qwen定位) 失败 ──→ 感知错（修感知）
            └─ 三路全成功 ──────────────────→ OK
```

一致性检查用 `core.cover_cabto_api.transition_check`，关键违规项：

| 违规 | 含义 | 归因 |
|---|---|---|
| `missing_pre` | 前置没满足 | 前置状态构造错 |
| `missing_add` | 声称的 add 没达成 | add 效应错 |
| `remaining_del` | 声称的 del 没消失 | del 效应错 |
| `missing_preserved_pre` | pre 中该保持的谓词消失 | **del 漏了某项**（早期只查 add/del 会漏掉这类） |

---

## 三、核心结论：感知瓶颈的三层

通过逐层对照实验，把「感知失败」拆成了**三个独立层次，每层根因不同**：

| 层 | 现象 | 根因 | 解决 |
|---|---|---|---|
| **1. 解析协议** | `No JSON object in model response` | `parse_pixel` 只认 JSON，模型返回 `(x,y)` 纯文本 / XML `<points x1 y1>` | ✅ 已修：加纯文本/XML 兜底 |
| **2. 单帧打点精度** | 对易辨物体偏 | 对红苹果、橙虾这类区分度高的物体，3B/7B 都够用（1-5px） | 无需解决 |
| **3. 颜色相近物体** | potato 打偏 76px / 混淆成 shrimp | potato（棕）vs shrimp（橙）外观相近，零样本 VLM 单帧打点难区分 | ✅ 已修：换物体（青椒） |

**一句话**：感知瓶颈不是「模型太小」也不是「机制不行」，而是「解析协议 bug（已修）
+ 颜色相近物体的可辨性（换物体解决）」两件事。

---

## 四、关键发现时间线

1. **探针 3 首跑全「感知错」** → 以为是模型打点差。
2. **发现 render=False 导致黑图** → `render_cam` 在 `render=False` 时返回全黑，
   模型看到空场景说 "There are no objects"。这是脚本 bug，不是模型问题。
3. **修复渲染后，模型返回 `(x,y)` 纯文本** → `parse_pixel` 只认 JSON，抛
   "No JSON object"。这是解析协议不匹配。
4. **修复 `parse_pixel`（加纯文本/XML 兜底）后** → blocks 的 pick 全部从失败变通过，
   cover 的 shrimp pick 通过。**解析协议是主要瓶颈，收益最大。**
5. **打点精度对比** → 3B 打点其实很准（易辨物体 1-5px），之前"全挂"是解析 bug 冤枉的。
6. **cross-view 诊断** → shrimp 两视角都准（cross-view 0.005m，机制没问题）；
   potato front 偏 76px（是"打点偏"，不是"几何问题"）。
7. **换物体** → potato（棕）换成 pepper 青椒（绿），front 打点 76px → 6px，
   cross-view 0.163m → 0.031m。**换"认知一致"的物体最有效。**

---

## 五、模型对比（probe_pointing）

| 模型 | 易辨物体（shrimp/apple） | 难辨物体（potato） | 结论 |
|---|---|---|---|
| Qwen2.5-VL-3B | 1-5px ✅ | 76px 偏 | 够用，但对相近颜色偏 |
| Qwen2.5-VL-7B | 2-5px ✅ | 196px 混淆成 shrimp | 易辨更好，难辨反而更差 |
| Qwen3.5-9B（Qwen3-VL） | — | 300px 乱打 | 全面退化，不适合打点 |

**「规模 vs 机制」的答案：都不是。** 换 3B→7B 只是把 potato 的失败从"打偏"变成
"认错"，换 9B 是全面退化。真正的问题是「物体可辨性」，换模型解决不了，换物体才能。

> 附：不同模型输出格式各异——3B 偏 JSON、7B 偏 XML/纯文本、9B 偏纯文本坐标。
> 根治格式问题应做 constrained decoding（强制 JSON 输出），而非逐个加解析兜底。

---

## 六、换物体（potato → pepper 青椒）

用户判断：物体颜色必须符合认知规范（"蓝色土豆"会让模型和人困惑），应换成认知里
颜色本来就对的物体。最终把 potato（棕）换成 **pepper 青椒（绿）**——绿色是青椒的
天然颜色，与虾（橙）、苹果（红）三色区分度最大，几何复用土豆的椭球体。

改动清单（6 处）：

| 文件 | 改动 |
|---|---|
| `scenes/cover_kitchen_sort_v2.json` | items/pairs/参数 + 任务描述 potato→pepper |
| `core/single_arm_rebuilt.py` | body name potato→pepper + rgba 棕→绿 `.13 .74 .07` |
| `core/cover_cabto_api.py` | DESC → "green pepper" |
| `core/cover_visual_localizer.py` | DESCRIPTIONS + HSV 绿色 mask（hue 38-95） |
| `core/cover_cabto_grounding.py` | PAIRS + task context |
| `attribution/probe_crossview.py` | DESC → "green pepper" |

效果：

| 指标 | potato 棕色 | 蓝土豆（硬改色） | **青椒（换物体）** |
|---|---|---|---|
| front 打点误差 | 76px | 65px | **5.9px** ✅ |
| cross-view | 0.163m | 0.138m | **0.031m** ✅ |

**结论**：换"认知一致"的物体（青椒=绿）比硬改颜色（蓝土豆）有效得多——因为
"青椒"这个词的绿色认知与渲染颜色天然对齐，模型不再困惑。

---

## 七、代码改动（parse_pixel 修复）

`core/cover_cabto_api.py` 的 `Perception.parse_pixel` 原来只认 JSON，已加纯文本/XML 兜底：

```python
# 1) JSON 字段 point / point_2d
# 2) XML <points x1=".." y1="..">
# 3) 括号/方括号 (x, y) / [x, y]
# 4) 逗号分隔的纯数字对（用负向前后瞻避免属性名 x1="361" 里的 1 被误解析）
```

---

## 八、下一步

1. **重新采样 pepper 策略**：换物体后，旧的 `policies.json` 还是 potato 的动作模型，
   需跑 `prepare` + `sample` 重新生成 pepper 的 models/policies，才能跑完整闭环。
2. **constrained decoding**：根治输出格式多样性（比"打地鼠式加解析兜底"更彻底）。
3. **三路归因接入 `feedback.py`**：替代「全 unresolved」的诊断，让
   `candidate_model_patch` 变成有证据的定向修订。
4. **（长期）EPM 式可学习记忆**：若还要处理"颜色相近物体"（不换物体），需训练
   感知模型学会区分，而非靠零样本 VLM 打点。
