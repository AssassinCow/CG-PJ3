# CG-Tutor 当前代码结构与运行说明

本文档记录当前仓库状态。当前阶段的目标已经从“单个场景修补”收敛为：让 pipeline 稳定地产生可诊断视频，并把概念成功标准、critic 证据、repair plan 和 best selection 串成闭环。

## 1. 当前保留的概念与输出目录

仓库目前只保留一个输出根目录：

```text
outputs/
```

`configs/concepts/` 只保留已经在 `outputs/` 中出现过的 5 个概念：

| Concept | 当前用途 |
| --- | --- |
| `affine_transformation` | 几何变换与矩阵路径教学 |
| `forward_kinematics_chain` | 关节链、层级变换、末端执行器 |
| `mirror_reflection` | 镜面反射、入射/反射光线、法线 |
| `prism_dispersion_teaching` | 棱镜分光、RGB 光线、法线与 Snell 关系 |
| `shape_morphing` | 形状插值 / morphing 过程 |

历史 `outputs_dynamic*`、`outputs_final_eval`、`outputs_repair_eval` 等已合并或移除，避免结果目录和概念集合继续分叉。

Git 中只保留 `outputs/` 的公开复盘文件：所有 MP4 视频，以及每个场景的 final selection、narrative、storyboard、scene profile、success spec、Scene IR、visual contract、最终 scene script、compiled scaffold 和导出 manifest。逐帧 PNG、preview PNG、overlay PNG、stdout/stderr、raw model response、per-iteration repair/verifier/critic trace 等运行期中间产物不提交。

## 2. Pipeline 总览

当前主链路：

```text
Concept YAML
  -> Concept Decomposer
  -> Scene Profile
  -> Auto Success Spec
  -> Storyboard
  -> Scene IR / Visual Contract
  -> Deterministic compiler scaffold
  -> Blender coder
  -> Scene verifier + Contract validator
  -> Keyframe preview
  -> Full render
  -> Critic ensemble
  -> Concept metrics / Auto Success evidence / Cross-ref
  -> Repair plan / retry
  -> Best selection
  -> ffmpeg compose final.mp4
```

几个关键原则：

- 用户仍只需要写普通 concept YAML；不要求手写 success spec 或 metric。
- 手写 `success_spec:` 仍受支持，优先级高于自动生成 spec。
- 自动生成的 `success_spec.generated.json` 默认是软门禁，不在 iter00 直接 hard fail。
- `compiled_fallback` 是诊断保底，不再被当作普通质量提升；使用 degraded fallback 时 final status 会明确标记。
- best selection 先看 structural / success hard，再看 semantic blockers 和 score，避免“好看但概念错”的版本胜出。

## 3. 主要源码模块

```text
scripts/
  run_concept.py              # 主入口
  check_llm.py                # provider / 环境健康检查
  eval_concepts.py            # 输出汇总与人工评分表
  continue_critic.py          # 在已有 run 上继续 critic 迭代
  dry_replay_concept_mismatch.py
  generate_architecture_figures.py
  build_ppt.py

src/cg_tutor/
  pipeline.py                 # 总控流程、resume、fallback、best selection
  config.py                   # pipeline-level 配置加载
  llm_client.py               # 统一 LLM 客户端 + provider 链式 fallback
  correction_controller.py    # 局部修复失败时的 retry 路由决策
  terminal_ui.py              # 运行期进度 / 日志输出
  _logging.py                 # 结构化日志初始化
  success_spec.py             # 手写 Success Spec schema 与 coder 格式化
  auto_success_spec.py        # 自动 Success Spec 软规则生成与 critic 连续确认
  concept_metrics.py          # 概念 metric、failure_class、Auto Spec 静态状态
  critic_loop.py              # critic history、quality key、failure class 聚合
  repair_plan.py              # 结构化修复目标
  critic_cross_reference.py   # critic finding 与 AST/contract 交叉验证
  scene_ir.py                 # Storyboard -> Scene IR
  scene_compiler.py           # Scene IR -> bpy scaffold / fallback
  scene_verifier.py           # render safety / AST verifier
  scene_state.py              # 静态 scene-state 审计(对象 / keyframe 通道)
  scene_profiles.py           # scene-level 风格策略 profile
  contract_validator.py       # visual contract / anchor / label / vector 检查
  visual_contract.py          # per-shot anchor / label / vector 定义
  preview.py                  # keyframe preview 与帧级启发式
  failure_memory.py           # 本轮和跨 run 失败记忆
  storyboard_sanitizer.py     # storyboard 清洗
  prompts/                    # 各 agent prompt 模板

src/cg_tutor/agents/
  base.py                     # 共用 agent 工具(artifact 保存、重试)
  concept_decomposer.py
  profile_generator.py        # scene profile 生成 agent
  storyboard.py
  blender_coder.py
  render_critic.py            # critic ensemble、partial success 聚合、member summary
  latex_overlay.py

src/cg_tutor/composer/
  formula_render.py
  compose.py
  ffmpeg_wrapper.py

src/cg_tutor/blender/
  runtime.py                  # headless Blender / WSL / Windows Blender 兼容
  primitives.py
  templates/                  # 内置场景模板

src/cg_tutor/eval/
  metrics.py                  # run 总结指标(用于人工评分汇总)

src/cg_tutor/schemas/
  feedback.py                 # CriticIssue / CriticReport
  narrative.py                # 概念拆解 schema
  storyboard.py               # storyboard schema
```

## 4. 当前模型配置

默认配置文件为 `configs/models_api.yaml`：

| Agent | Primary | Fallback |
| --- | --- | --- |
| concept_decomposer | `openai/gpt-5.5` | `anthropic/claude-sonnet-4.6` |
| scene_profile | `openai/gpt-5.5` | `anthropic/claude-sonnet-4.6` |
| storyboard | `anthropic/claude-sonnet-4.6` | `openai/gpt-5.5` |
| blender_coder | `anthropic/claude-opus-4.7` | `openai/gpt-5.5` |
| render_critic | `anthropic/claude-sonnet-4.6` | `openai/gpt-5.5` |

CLI 默认 critic ensemble 为：

```text
--critic-ensemble claude,gpt
--critic-strictness strict
```

`strict` 会保留 union block，并把 critic 分歧作为通过条件的一部分处理。需要更快实验时可改为 `--critic-ensemble claude --max-critic-iters 1`。

## 5. 主运行命令

默认输出到 `outputs/<concept>/`。当前提交快照建议用 `prism_dispersion_teaching` 作为默认非 Cycles 复跑样本:

```bash
MPLCONFIGDIR=/tmp CG_TUTOR_API_TIMEOUT=300 \
.venv/bin/python scripts/run_concept.py prism_dispersion_teaching \
  --out-root outputs \
  --critic-ensemble claude,gpt \
  --critic-strictness union \
  --max-critic-iters 3
```

Cycles / GPU 尝试：

```bash
MPLCONFIGDIR=/tmp CG_TUTOR_API_TIMEOUT=300 \
.venv/bin/python scripts/run_concept.py mirror_reflection \
  --out-root outputs \
  --critic-ensemble claude,gpt \
  --critic-strictness union \
  --render-engine CYCLES \
  --cycles-device AUTO \
  --max-critic-iters 3
```

从已有结果继续:

```bash
MPLCONFIGDIR=/tmp CG_TUTOR_API_TIMEOUT=300 \
.venv/bin/python scripts/run_concept.py shape_morphing \
  --out-root outputs \
  --resume \
  --max-critic-iters 3
```

历史说明: `texture_mipmap_lod` 和 `bezier_curve` 曾作为架构调试样本使用，但不属于当前公开快照的 retained output set；当前提交不再保留它们的 concept YAML。若后续需要继续实验，可重新添加对应 YAML 后再运行。

## 6. 重要 CLI 参数

| 参数 | 默认 | 说明 |
| --- | ---: | --- |
| `--out-root` | `outputs` | 输出根目录 |
| `--max-critic-iters` | `5` | iter00 后最多 retry 5 次 |
| `--early-stop-stale-iters` | `0` | 默认不早停 |
| `--critic-ensemble` | `claude,gpt` | critic 后端聚合 |
| `--critic-strictness` | `strict` | `consensus` / `union` / `strict` |
| `--best-selection` | `balanced` | final iteration 选择策略 |
| `--compiler-seed` | on | 给 coder 提供 deterministic scaffold |
| `--preview-render` | on | full render 前抽关键帧预览 |
| `--diff-repair` | on | retry 优先尝试 diff repair |
| `--render-engine` | `BLENDER_EEVEE` | 可选 `CYCLES` |
| `--cycles-device` | `AUTO` | `AUTO` / `GPU` / `CPU` |
| `--strict-best-replay` | on | final 视频必须来自 critic 评分过的帧或 hash 一致 replay |
| `--max-verifier-repair-iters` | `2` | render 前 verifier/contract repair 预算 |

## 7. 当前新增的闭环机制

### Failure Class

保留旧的 `block/warn`，新增兼容字段 `failure_class`：

```text
structural_fatal
success_hard
success_soft
aesthetic_warn
```

用途：

- `structural_fatal` 和 `success_hard` 进入 pass / selection 硬隔离。
- `success_soft` 进入 retry 和 selection 次级信号，但不直接一票否决。
- `aesthetic_warn` 用于风格、美观、legacy fallback 阈值。

### Auto Success Spec

每次 run 会生成：

```text
success_spec.generated.json
success_spec.validation.json
success_spec.effective.json
```

自动 spec 的 DSL 目前只使用低误伤规则：

```text
object_visible
text_readable
stay_in_screen_safe
helper_hidden
animation_coverage
progressive_visual_ordering
```

规则从 concept/profile/storyboard token 中产生，不凭空创造 anchor。generated rule 默认 `success_soft` / `aesthetic_warn` / `diagnostic`，只有在当前 run 内被 critic 连续确认且没有 AST 反证时才可能临时升级。

### Critic Ensemble Partial Success

critic member 如果同时包含 `issues` 和 `execution_errors`，现在仍参与 aggregate。只有“没有 issues 且只有 execution error”的 member 才视为 unusable。对应 artifact：

```text
critic_iterNN.member_usable_summary.json
```

这避免了“critic 已经看见问题，但因为一个 JSON/执行错误导致 retry plan 没目标”的失明问题。

### Fallback 与 Final Status

`compiled_fallback` 现在是 render-safe 诊断保底。若使用 degraded fallback 或存在未解决违反项，最终不会伪装成 pass。`critic_best.json` 会记录：

```text
final_status
selected_reason
fallback_degraded
structural_fatal / success_hard / success_soft / aesthetic_warn
```

常见 `final_status`：

```text
pass
best_with_violations
fallback_diagnostic_video
```

### Vector / Ray Scaffold

对于 mirror、prism、ray/vector teaching 类场景，coder addendum 会加入 `VECTOR/RAY MINIMUM SCAFFOLD`，要求至少创建可追踪的 thin curve ray、normal、angle cue 和短标签。compiled scaffold 也会补最小 placeholder，但 placeholder 只作为可渲染骨架，不等价于语义成功。

## 8. 输出文件说明

| 文件 | 含义 |
| --- | --- |
| `narrative.json` | 概念拆解 |
| `scene_profile.json` | 场景类型、anchors、spatial relationships |
| `success_spec.generated.json` | 自动生成的软成功规则 |
| `success_spec.validation.json` | 自动 spec 接受 / 拒绝原因 |
| `success_spec.effective.json` | 手写 + 自动 spec 合并视图 |
| `storyboard.json` | 最终分镜 |
| `scene_ir.json` | 结构化 Scene IR |
| `visual_contracts.json` | shot-level anchor / label / vector contract |
| `scene.compiled.py` | deterministic scaffold |
| `critic_best.json` | final 选择依据 |
| `video_exports.json` | final / alternative 视频导出清单 |
| `final*.mp4` | 最终视频与不同 selection view 的替代视频 |

当前 Git 跟踪策略:

- 保留: 上表中的关键复盘文件与所有 `final*.mp4`。
- 不保留: 逐帧图、preview 图、overlay 图、stdout/stderr、raw response、repair/verifier/critic per-iteration trace。
- 原则: public snapshot 应该能说明结果和复现实验，但不把一次运行的完整调试转储放进版本库。

## 9. 查看结果

查看当前 5 个输出：

```bash
find outputs -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort
```

查看某个 final 的选择原因：

```bash
cat outputs/mirror_reflection/critic_best.json
```

查看视频元数据：

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,nb_frames,r_frame_rate,duration \
  -of default=noprint_wrappers=1 \
  outputs/mirror_reflection/final.mp4
```

运行测试：

```bash
.venv/bin/python -m pytest -q
```

最近一次全量测试结果：

```text
545 passed, 3 skipped
```
