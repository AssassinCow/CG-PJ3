# CG-Tutor Architecture Overview

This document summarizes the current end-to-end framework in three views:

1. Control flow: concept YAML to final MP4
2. Diagnostic and retry loop
3. Code module layering

Paper-ready vector figures generated from the current code structure:

- `report/figures/architecture_pipeline.pdf` / `.svg`
- `report/figures/architecture_feedback_loop.pdf` / `.svg`
- `report/figures/architecture_module_layers.pdf` / `.svg`

Regenerate them with:

```bash
MPLCONFIGDIR=/tmp .venv/bin/python scripts/generate_architecture_figures.py
```

## 1. End-to-End Control Flow

```mermaid
flowchart TD
    A[Concept YAML<br/>configs/concepts/*.yaml] --> B[Pipeline<br/>src/cg_tutor/pipeline.py]

    B --> C1[Concept Decomposer<br/>agents/concept_decomposer.py]
    C1 --> C2[narrative.json]

    C2 --> D1[Profile Generator<br/>agents/profile_generator.py]
    D1 --> D2[scene_profile.json]

    C2 --> S1[Auto Success Spec<br/>auto_success_spec.py]
    D2 --> S1
    S1 --> S2[success_spec.generated / validation / effective]

    C2 --> E1[Storyboard Agent<br/>agents/storyboard.py]
    D2 --> E1
    S2 --> E1
    E1 --> E2[storyboard.json]
    E2 --> E3[Storyboard Sanitizer<br/>storyboard_sanitizer.py]

    E3 --> F1[Deterministic Scene Compiler<br/>scene_compiler.py]
    F1 --> F2[scene.compiled.py<br/>compiled scaffold]

    E3 --> G1[Blender Coder<br/>agents/blender_coder.py]
    D2 --> G1
    H1[Failure Memory<br/>failure_memory.py] --> G1
    H2[Repair Plan<br/>repair_plan.py] --> G1
    H3[Scene IR<br/>scene_ir.py] --> G1
    H4[Visual Contracts<br/>visual_contract.py] --> G1
    H5[Grounding Patch<br/>critic-derived] --> G1
    H6[Concept Metrics Addendum<br/>concept_metrics.py] --> G1
    H7[Critic x AST Cross-Reference<br/>critic_cross_reference.py] --> G1
    H8[Addendum Bundle<br/>priority / metric / cross-ref / auto spec] --> G1

    G1 --> I1[scene.py]

    I1 --> J1[Scene Verifier<br/>scene_verifier.py]
    I1 --> J2[Contract Validator<br/>contract_validator.py]
    I1 --> J3[Scene IR Verify<br/>scene_ir.py]

    J1 --> K{Fatal?}
    J2 --> K
    J3 --> K

    K -- yes --> F2
    K -- no --> L1[Preview Render<br/>preview.py + blender/runtime.py]

    F2 --> L1

    L1 --> L2[Preview Verification<br/>visibility / motion / overlay]
    L2 --> M1[Full Render<br/>blender/runtime.py]
    M1 --> M2[frames/*.png]

    M2 --> N1[Render Critic Ensemble<br/>agents/render_critic.py]
    D2 --> N1
    C2 --> N1
    E3 --> N1

    N1 --> N2[critic_iterNN.json<br/>member_usable_summary]
    N2 --> O1[Best Selection<br/>critic_loop.py]
    O1 --> P1[Overlay + Compose<br/>composer/compose.py + ffmpeg]
    P1 --> Q[final.mp4]
```

## 2. Retry and Diagnostic Loop

```mermaid
flowchart TD
    A[scene.py] --> B[Verifier / Contract / IR checks]
    B --> C{Fatal block?}

    C -- yes --> D[compiled fallback<br/>scene.compiled.py]
    C -- no --> E[preview render]

    E --> F{preview block?}
    F -- yes --> G[stop / fallback / early exit]
    F -- no --> H[full render]

    H --> I[critic ensemble<br/>strict / union / consensus]
    I --> J[critic report]

    J --> K[concept_metrics.py<br/>AST / storyboard / concept / auto-spec checks]
    J --> L[critic_cross_reference.py<br/>critic issue x scene AST]
    J --> M[grounding_patch.iterNN.json]

    K --> N[retry addendum<br/>failure_class aware]
    L --> N
    M --> N

    N --> O[repair_plan.py<br/>history-aware targets]
    O --> P[Blender Coder retry]

    P --> A
```

## 3. Signal Routing

```mermaid
flowchart LR
    A[Static AST checks<br/>scene_verifier.py] --> Z[Unified Retry Addendum]
    B[Contract checks<br/>contract_validator.py] --> Z
    C[Preview image checks<br/>preview.py] --> Z
    D[Vision critic findings<br/>render_critic.py] --> Z
    E[Concept metrics<br/>concept_metrics.py] --> Z
    F[Critic x AST join<br/>critic_cross_reference.py] --> Z
    G[Failure memory<br/>historical regressions] --> Z
    H[Repair plan<br/>trend / best-of-N / missing objects] --> Z

    Z --> Y[LLM coder next iteration]
```

## 4. Module Layering

```text
cg_tutor/
├── pipeline.py                     # Top-level orchestration: YAML -> mp4
├── config.py                       # Pipeline-level configuration loader
├── llm_client.py                   # Unified LLM client with provider-chain fallback
├── correction_controller.py        # Retry routing when local repair stalls
├── terminal_ui.py                  # Run-time progress / log surface
├── _logging.py                     # Structured logging setup
│
├── agents/
│   ├── base.py                     # Shared agent helpers (artifact save, retries)
│   ├── concept_decomposer.py       # Concept decomposition
│   ├── profile_generator.py        # Scene profile generation
│   ├── storyboard.py               # Storyboard generation / patching
│   ├── blender_coder.py            # scene.py generation / diff repair
│   ├── render_critic.py            # Single critic / ensemble critic
│   └── latex_overlay.py            # Formula / annotation overlay agent
│
├── scene_compiler.py               # Deterministic scaffold / compiled fallback
├── scene_verifier.py               # AST / animation / safety verification
├── scene_state.py                  # Static scene-state audit (objects / keyframes)
├── scene_profiles.py               # Scene-level visual policy profiles
├── contract_validator.py           # Visual contract validation
├── visual_contract.py              # Required anchors / labels / vectors by shot
├── preview.py                      # Preview render verification
├── critic_loop.py                  # Best-of-N / trend / critic history logic
├── repair_plan.py                  # Retry target planning
├── failure_memory.py               # Cross-run memory
├── scene_ir.py                     # Intermediate scene representation
├── success_spec.py                 # Manual Success Spec schema / formatting
├── auto_success_spec.py            # Generated soft Success Spec rules
├── concept_metrics.py              # Concept-specific + auto-spec deterministic checks
├── critic_cross_reference.py       # Critic finding x AST evidence join
├── storyboard_sanitizer.py         # Storyboard cleanup
│
├── prompts/                        # Prompt templates per agent
│
├── blender/
│   ├── runtime.py                  # Blender execution wrapper (WSL / Win / Linux)
│   ├── primitives.py               # Reusable bpy primitive builders
│   └── templates/                  # Bundled scene templates
│
├── composer/
│   ├── compose.py                  # Overlay + ffmpeg composition
│   ├── ffmpeg_wrapper.py           # ffmpeg invocation / probe helpers
│   └── formula_render.py           # LaTeX / formula image rendering
│
├── eval/
│   └── metrics.py                  # Run-summary metrics for manual scoring
│
├── schemas/
│   ├── feedback.py                 # CriticIssue / CriticReport
│   ├── narrative.py                # Narrative / concept decomposition schema
│   └── storyboard.py               # Storyboard schema
│
└── configs/concepts/*.yaml         # Concept specifications
```

## 5. Current Control Principles

```text
Concept Spec
   -> LLM proposes candidate visuals and code
   -> Auto Success Spec derives soft, machine-readable success evidence
   -> Deterministic layers reject obvious structural failures
   -> Preview and critic inspect the rendered result
   -> Cross-reference / metrics / repair planning convert symptoms back
      into actionable structural fixes
   -> Next iteration retries with tighter constraints
```

The current selection and pass logic is deliberately conservative:

- Legacy `block` / `warn` is preserved, but `failure_class` separates
  `structural_fatal`, `success_hard`, `success_soft`, and `aesthetic_warn`.
- `structural_fatal` and `success_hard` outrank critic score in best
  selection.
- Generated Auto Success Spec rules start soft and do not hard-fail iter00.
- Critic member reports with useful issues are retained even when that
  backend also has a partial execution/parsing error.
- `compiled_fallback` is a diagnostic safety net, not a quality improvement;
  degraded fallback output cannot silently become `pass`.

## 6. Current Practical Reading

- `pipeline.py` is the control plane.
- `scene_compiler.py` is the deterministic safety net.
- `render_critic.py` is the visual evaluation layer.
- `success_spec.py` / `auto_success_spec.py` define explicit success signals.
- `concept_metrics.py` and `critic_cross_reference.py` are the bridge
  modules that try to turn post-hoc visual findings into reusable,
  machine-actionable repair signals.
