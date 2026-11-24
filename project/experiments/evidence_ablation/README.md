# Evidence Ablation（视觉证据回看 vs 契约更新质量）

## 目的

在**相同的物理执行经验**下，检验"视觉证据回看"能否减少错误的契约更新，
而不是只提高视觉问答准确率。这是对 VISTA 的 lossless visual memory +
inspect 思想做的一次最小、离线、无训练的机制验证。

对应主问题：有限交互下，建立与实际能力匹配、且能支持任务组合的技能契约。

## 三组方法（只有"提供哪些帧"不同，判读模型完全一致）

| 方法 | 证据 | 可否回看 |
|---|---|---|
| A | 仅 before / after 两张图（当前 EffectChecker 方式） | 否 |
| B | 固定全部阶段帧（approach/descend/close/lift 全给） | 否 |
| C | 初始 before / after；证据不足时按需 inspect 该谓词可见的阶段帧 | 是 |

## 运行

```bash
cd project/experiments/evidence_ablation

# 对比三组，打印表并写 result.json
python3 evaluate.py --output result.json

# 纯逻辑回归（不依赖 Mujoco / VLM / 网络）
python3 -m unittest test_evidence_ablation -v
```

## 结果怎么读

- `wrong_tighten`：谓词实际成立（truth=True）却被判不成立 → 技能被错误排除（覆盖损失）。
- `wrong_accept`：谓词实际不成立（truth=False）却被判成立 → 契约漏检。
- `abstain`：证据不足、无法判定。
- `frames`：证据总帧数（每次 inspect 追加的帧计入成本）。
- `cov`：可执行任务中被正确保留的比例（只由 add 谓词的 `wrong_tighten` 决定）。

## 机制建模（重要边界）

- 协议里 `visibility` 是**冻结的可见性模型**（哪个谓词在哪个阶段帧可见，每个谓词
  恰好 3 帧，奇数投票避免平局），`episodes[].truth` 是注入的 ground truth。
  它们是**合成的机制探针**，不声称物理场景自洽。
- `readout_noise` 模拟判读噪声；`readout_noise=0` 时证据只影响弃权率，
  噪声 > 0 时多帧多数投票才体现"冗余证据降错"。
- 判断器只读给它帧的 `readout`，从不看 `truth`；评分用到的 truth 不泄露给任何方法。
- 方法 C 用冻结可见性索引选帧（对应"知道该回看哪个阶段"），触发条件是
  "无可见帧，或可见帧数 < `min_frames_for_verdict`(2)"。一次回看可能顺带覆盖
  其他谓词（阶段帧被多个谓词共享），所以 C 的 inspect 次数通常少于谓词数。

## 当前初步结果（默认 noise=0.3, seed=0）

```
method label                       corr  wrongT  wrongA  abst  frames  insp    cov
A      before-after only             14       1       2     7      12     0   1.00
B      fixed phase evidence          21       1       2     0      36     0   1.00
C      on-demand inspection          21       1       2     0      34    10   1.00
```

诚实解读（这是机制探针，不是性能宣称）：

1. 证据回看**首先消除的是弃权**：A 无法看到只在中间阶段可见的谓词（7 次
   `abstain`），B/C 都能判定（0 次）。
2. 在本组参数下，**错误数（wrong_tighten + wrong_accept = 3）三组持平**：
   3 帧多数投票在 noise=0.3 下的错误率（约 22%）与 A 的单帧判读（30%）
   差距不大。也就是说，"回看降错"在低冗余、高噪声下并不自动成立。
3. C 用 34 帧（10 次 inspect）达到与 B 的 36 帧相同的判定质量——按需回看
   比无条件全量更省，且一次回看常顺带覆盖多个谓词。

**下一步要检验的变量**：更低噪声或更多可见帧（如 5 帧）是否让"回看"同时降错；
以及真实 VLM 判读噪声、真实阶段帧的可见性到底落在哪个区间。这些决定了该机制
值不值得接真实数据，而不是现在就直接上线。

## 后续接真实数据

本框架只验证机制。接真实 VLM / 物理时：

1. 用 `PhaseRecorder`（`../collect_pick_groups.py`）已采集的多相机阶段帧
   填充 `Frame.readout`（VLM 判读）与 `visible`（是否可见该谓词）。
2. `truth` 仍由独立验收（`world_state.check_effect` / 接触审计 / 安全判据）给出，
   不喂给判断器。
3. 只有当"证据回看在相同物理预算下确实降低错误契约更新"这一机制成立，
   才考虑投入真实 VLM 调用与在线物理交互。
