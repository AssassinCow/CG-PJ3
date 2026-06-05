# CG-Tutor 当前阶段实验报告

本文档同步当前代码框架与这一阶段实验经历。当前结论比较清楚：系统已经能稳定产出可诊断视频，但复杂教学语义仍不能靠“更多 prompt / 更多规则”自然解决。最近一轮工作的重点从单场景修补，转向修复反馈链路、收缩 hard gate、统一输出与实验样本。

## 1. 当前实验集合

当前只保留一个输出根目录 `outputs/`，并且只保留输出中实际出现过的 concept YAML。

版本库中只跟踪每个输出场景的公开复盘 artifact：所有 MP4 视频，以及 final selection、narrative、storyboard、scene profile、success spec、Scene IR、visual contract、最终 scene script、compiled scaffold 和导出 manifest。逐帧 PNG、preview PNG、stdout/stderr、raw model response、repair/verifier/critic per-iteration trace 不作为公开提交内容。

| Concept | final.mp4 | Best iter | Score | Frame block/warn | Concept block/warn | Success hard/soft | Final status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `affine_transformation` | yes | 0 | 0.8125 | 0 / 4 | 7 / 12 | 0 / 0 | `best_with_violations` |
| `forward_kinematics_chain` | yes | 1 | 0.7150 | 7 / 1 | 7 / 20 | 0 / 1 | `best_with_violations` |
| `mirror_reflection` | yes | 2 | 0.7750 | 0 / 2 | 16 / 15 | 0 / 1 | `best_with_violations` |
| `prism_dispersion_teaching` | yes | 1 | 0.8388 | 0 / 8 | 1 / 18 | 0 / 0 | `best_with_violations` |
| `shape_morphing` | yes | 0 | 0.7625 | 1 / 6 | 3 / 14 | 0 / 1 | `best_with_violations` |

所有 5 个保留场景都有 `final.mp4`，但没有一个被系统标记为严格 pass。这是有意的：当前版本不再把“可渲染、看起来还行”的结果误报为达标，而是通过 `final_status=best_with_violations` 明确说明仍有语义或视觉证据不足。

## 2. 当前框架状态

### 2.1 已稳定的部分

- 单一输出目录 `outputs/` 已完成，历史输出目录已合并。
- concept 集合已收敛到 5 个当前提交样本。
- 全量测试通过：`545 passed, 3 skipped`。
- `run_concept.py` 支持 EEVEE / CYCLES、critic ensemble、resume、preview、diff repair、strict best replay。
- Blender 5.x / WSL / Windows Blender 路径兼容和 Cycles device 选择已经接入。
- render 前有 scene verifier、contract validator、concept metrics、preview 多层检查。
- `critic_best.json` 记录 final status、selected reason、failure class counts、fallback degraded 等关键信息。

### 2.2 这一阶段新增或强化的机制

**Failure Class**

旧的 `block/warn` 过粗，会把结构错误、成功标准失败、美学问题混在一起。现在新增：

```text
structural_fatal
success_hard
success_soft
aesthetic_warn
```

selection 和 pass 逻辑优先处理 structural / success hard，再考虑 critic score。

**Success Spec / Auto Success Spec**

早期设想是用户手写 `success_spec:`，但实践中这不适合通用框架：用户不可能为每个概念补 metric。因此当前改成自动生成软规则：

- `success_spec.generated.json`
- `success_spec.validation.json`
- `success_spec.effective.json`

自动 spec 默认只作为 soft evidence，不在 iter00 直接 hard fail。只有连续 critic evidence 且没有 AST 反证时，才在当前 run 内临时升级。

**Critic Ensemble Partial Success**

以前 critic member 一旦有 execution error，可能导致整个 aggregate 丢失 issues，repair plan 变成 0 target。现在只要 member 有 issues，即使同时有 execution error，也会参与聚合，并写出：

```text
critic_iterNN.member_usable_summary.json
```

**Repair Plan 收缩**

retry 目标现在强调“少量高置信修复”，避免 LLM 收到一长串问题后补一堆新对象，导致画面越修越乱。

**Fallback 降级标记**

`compiled_fallback` 保底可渲染，但不会伪装成质量提升。若 fallback 只是诊断视频，`fallback_degraded=true` 且 final status 不会是 pass。

**Vector / Ray Minimum Scaffold**

对 mirror / prism / ray / vector 类场景，系统给 coder 明确的最小骨架要求：thin curve rays、surface normal、angle cue、readable short label。compiler 也可补 placeholder，但 placeholder 不等于语义成功。

## 3. 实验过程回顾

### 3.1 第一阶段：基础教学视频能跑通

早期集中在 `affine_transformation`、`bezier_curve`、`phong_lighting` 等基础概念，核心目标是让 pipeline 从 concept YAML 到 `final.mp4` 完整跑通。这个阶段的主要收益来自：

- API provider fallback。
- Storyboard schema validation。
- Scene IR / visual contract。
- scene verifier。
- preview render。
- critic loop + best replay。

这一阶段证明系统可以自动生成教学视频，但也暴露出一个问题：best iteration 不一定是最后一轮，多轮 retry 不是单调提升。

### 3.2 第二阶段：更真实/更复杂场景暴露 grounding 问题

尝试深海、雨窗、棱镜、镜面反射等场景后，出现了明显退化：

- 自然语言里的“真实场景”容易被压缩成黑背景、发光矩形、透明球。
- ray / normal / angle 这类教学骨架经常缺失或命名混乱。
- label 存在但不可读、镜像、出框或遮挡。
- 代码属性正确并不等于观众能看懂教学目标。

这推动了 scene profile、persistent anchors、visual contracts、contract validator 和 grounding patch 的接入。

### 3.3 第三阶段：Success Spec 方向

`depth_of_field_focus_pull` 暴露了一个关键反例：`focus_distance` 有 keyframe 不代表视觉上能看出焦点移动，text object 存在不代表文字可读。于是提出 Success Spec：

```text
先定义成功状态，再生成；每一层验证成功状态是否保留。
```

MVP 中实现了 Success Spec schema、DoF aperture 阈值从 YAML 读取、text faces camera / HUD placement 静态检查等。但继续推进后发现，手写 spec 不适合普通用户。

### 3.4 第四阶段：Auto Success Spec 软门禁

为避免要求用户手写规则，系统改为自动生成受限 DSL 的软规则，例如：

```text
object_visible
text_readable
stay_in_screen_safe
helper_hidden
animation_coverage
progressive_visual_ordering
```

关键设计是“影子 + 软门禁”：

- 自动规则默认不 hard fail。
- 不写回 YAML。
- 不污染其它场景。
- 只从 concept/profile/storyboard 中已有 token 生成，不凭空创造 anchor。
- 当前 run 内连续确认才升级。

这比给每个场景手写 metric 更符合通用框架目标。

### 3.5 第五阶段：反馈链路全面收口

`mirror_reflection` 和 `texture_mipmap_lod` 暴露出新的框架问题：

- critic 明明报了大量问题，但因为 partial JSON / execution error，aggregate 可能丢 issues。
- Auto Success Spec 把“对象存在但看不见”误升级为 object missing hard failure。
- fallback 可渲染，但语义弱，容易被误当成改进。
- vector/ray placeholder 数量补齐不等于语义正确。

因此当前代码做了反馈链路收口：

- critic partial success 不再丢。
- object visible 升级时引入 AST anchor status 反证。
- 已创建但 critic 看不见的对象归因为 `visibility_unproven success_soft`。
- `helper_hidden` 不升级 hard。
- fallback degraded 明确标记。
- repair plan 增加 source report，并优先 contract / concept mismatch。

## 4. 当前结果评价

### 4.1 好的方面

- 当前 5 个保留场景全部能输出 final video。
- 系统现在更诚实：未达标会标 `best_with_violations`，不把低质量结果包装成 pass。
- 反馈链路更可诊断：critic member 可用性、repair plan、success evidence、failure class 都能在 artifact 中追踪。
- 对 vector/ray 场景，至少有更明确的骨架要求，减少“只放标签不放光线/法线”的失败。

### 4.2 仍然不足

- 复杂教学语义仍不稳定，尤其是 ray / normal / angle / label 的空间关系。
- critic 能指出问题，但 LLM coder 未必能稳定执行最小修复。
- Auto Success Spec 适合提供软信号，但还不能替代真正的像素级 evidence extractor。
- placeholder scaffold 能保证可渲染，但不能证明概念表达正确。
- 多轮 retry 仍可能退化，因此 best selection 和 final status 必须继续保守。

## 5. 对关键场景的观察

### `texture_mipmap_lod`（历史调试样本，当前公开快照未保留）

LOD 场景适合测试 Auto Success Spec：mip stack、LOD readout、纹理 patch、safe frame 都能从普通 concept/storyboard token 中自动抽出。历史结果能渲染，但仍有 concept mismatch，说明“自动 spec 给出修复信号”与“LLM 真正把画面修好”之间还有距离。

### `mirror_reflection`

它是反馈链路最重要的压力测试：同时覆盖 syntax/render verifier、fallback、vector/ray contract、critic partial success。当前能生成 final，但 concept block 很高，说明镜面反射的 ray-normal-angle 关系还没有稳定转化为可执行 scene。

### `prism_dispersion_teaching`

相比 mirror，prism 的画面感更好，score 也较高。它验证了光学台、RGB rays、surface normal 这类教学骨架比较适合当前架构，但仍需防止“有彩色线条但物理关系弱”。

### `bezier_curve`（历史调试样本，当前公开快照未保留）

Bezier 是相对适合该框架的基础教学场景：对象少、关系清晰、动画连续性强。但历史保留结果仍有 frame block，说明架构重跑后的输出不一定优于早期最佳版本。当前公开快照聚焦 5 个 retained scenarios，因此未包含该输出和对应 concept YAML。

## 6. 重要结论

1. 问题不只是 concept 描述不完整。
   自然语言描述需要被编译成可验证约束，否则“代码属性存在”经常不等于“教学目标达成”。

2. 也不能无限加 hard rule。
   太多硬规则会把 LLM 推向 Goodhart：补更多标签、更多 helper，而不是修最小问题。

3. 当前最有价值的是反馈链路质量。
   critic evidence 不能丢，repair plan 必须有目标，fallback 必须诚实降级，selection 必须优先语义硬失败。

4. Auto Success Spec 应继续保持 opt-in/soft-first。
   对普通概念自动生成软规则是合理的；直接把 generated rule 全部 hard gate 会造成负优化。

5. 最终视频质量的瓶颈仍在生成端。
   verifier、metric、critic 可以发现和约束问题，但如果初始 scene generation 无法构造清晰教学骨架，后续 repair 很难完全救回来。

## 7. 当前复现实验命令

```bash
# 全量测试
.venv/bin/python -m pytest -q

# 当前推荐的轻量最终评估
for c in prism_dispersion_teaching mirror_reflection shape_morphing; do
  MPLCONFIGDIR=/tmp CG_TUTOR_API_TIMEOUT=300 \
  .venv/bin/python scripts/run_concept.py "$c" \
    --out-root outputs \
    --critic-ensemble claude,gpt \
    --critic-strictness union \
    --max-critic-iters 3
done

# Cycles 场景
MPLCONFIGDIR=/tmp CG_TUTOR_API_TIMEOUT=300 \
.venv/bin/python scripts/run_concept.py mirror_reflection \
  --out-root outputs \
  --critic-ensemble claude,gpt \
  --critic-strictness union \
  --render-engine CYCLES \
  --cycles-device AUTO \
  --max-critic-iters 3
```

## 8. 后续优先级

短期不建议继续手写场景专用 metric。更合理的下一步是：

- 加强 render evidence extractor，而不是继续只靠 VLM free text。
- 把 vector/ray/normal/angle 的关系从“对象数量”升级为“空间关系证据”。
- 为 Auto Success Spec 增加更强的验证和降级日志，而不是扩大 hard gate。
- 拆小 `pipeline.py` 的控制状态，但应排在 evidence 链路稳定之后。
