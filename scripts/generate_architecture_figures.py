"""Generate paper-ready CG-Tutor architecture figures.

The figures are generated from the current repository structure. The script
checks that the modules named in the diagrams exist before writing outputs.
It intentionally uses clean stage-level diagrams instead of dense call graphs:
paper figures should explain the architecture, not reproduce every edge in
`pipeline.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
REPORT_FIGURES = ROOT / "report" / "figures"
DOC_FIGURES = ROOT / "docs" / "figures"


REQUIRED_PATHS = [
    "scripts/run_concept.py",
    "src/cg_tutor/pipeline.py",
    "src/cg_tutor/agents/concept_decomposer.py",
    "src/cg_tutor/agents/profile_generator.py",
    "src/cg_tutor/agents/storyboard.py",
    "src/cg_tutor/agents/blender_coder.py",
    "src/cg_tutor/agents/render_critic.py",
    "src/cg_tutor/success_spec.py",
    "src/cg_tutor/auto_success_spec.py",
    "src/cg_tutor/scene_ir.py",
    "src/cg_tutor/visual_contract.py",
    "src/cg_tutor/scene_compiler.py",
    "src/cg_tutor/scene_verifier.py",
    "src/cg_tutor/contract_validator.py",
    "src/cg_tutor/preview.py",
    "src/cg_tutor/concept_metrics.py",
    "src/cg_tutor/critic_cross_reference.py",
    "src/cg_tutor/repair_plan.py",
    "src/cg_tutor/critic_loop.py",
    "src/cg_tutor/failure_memory.py",
    "src/cg_tutor/blender/runtime.py",
    "src/cg_tutor/composer/compose.py",
]


PALETTE = {
    "input": "#EAF0FA",
    "agent": "#EAF6EF",
    "spec": "#FFF3D4",
    "runtime": "#FDEDED",
    "evidence": "#F1ECFA",
    "output": "#E8F5F7",
    "neutral": "#F7F8FA",
    "line": "#2E3338",
    "muted": "#6D747C",
    "border": "#4A5159",
    "accent": "#1F4F8B",
}


@dataclass(frozen=True)
class Node:
    key: str
    title: str
    body: str
    x: float
    y: float
    w: float
    h: float
    kind: str

    @property
    def left(self) -> tuple[float, float]:
        return self.x, self.y + self.h / 2

    @property
    def right(self) -> tuple[float, float]:
        return self.x + self.w, self.y + self.h / 2

    @property
    def top(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h

    @property
    def bottom(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y


def _validate_current_repo() -> None:
    missing = [path for path in REQUIRED_PATHS if not (ROOT / path).exists()]
    if missing:
        joined = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(f"architecture figure paths are stale:\n{joined}")

    concepts = {p.stem for p in (ROOT / "configs" / "concepts").glob("*.yaml")}
    outputs = {p.name for p in (ROOT / "outputs").iterdir() if p.is_dir()}
    if concepts != outputs:
        raise RuntimeError(
            "configs/concepts and outputs are not aligned:\n"
            f"  concepts only: {sorted(concepts - outputs)}\n"
            f"  outputs only: {sorted(outputs - concepts)}"
        )


def _setup(width: float, height: float) -> tuple[plt.Figure, plt.Axes]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    ax.set_axis_off()
    return fig, ax


def _draw_node(ax: plt.Axes, node: Node, number: int | None = None) -> None:
    patch = FancyBboxPatch(
        (node.x, node.y),
        node.w,
        node.h,
        boxstyle="round,pad=0.025,rounding_size=0.065",
        linewidth=0.85,
        edgecolor=PALETTE["border"],
        facecolor=PALETTE[node.kind],
        zorder=2,
    )
    ax.add_patch(patch)

    if number is not None:
        ax.text(
            node.x + 0.14,
            node.y + node.h - 0.14,
            str(number),
            ha="left",
            va="top",
            fontsize=8.0,
            color=PALETTE["accent"],
            weight="bold",
            zorder=3,
        )

    ax.text(
        node.x + node.w / 2,
        node.y + node.h * 0.66,
        node.title,
        ha="center",
        va="center",
        fontsize=8.8,
        weight="bold",
        color="#1F2328",
        zorder=3,
    )
    ax.text(
        node.x + node.w / 2,
        node.y + node.h * 0.30,
        node.body,
        ha="center",
        va="center",
        fontsize=7.0,
        color="#2E3338",
        linespacing=1.18,
        zorder=3,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str | None = None,
    label_offset: tuple[float, float] = (0.0, 0.0),
    dashed: bool = False,
    lw: float = 1.0,
    color: str | None = None,
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=9.5,
        linewidth=lw,
        color=color or PALETTE["line"],
        connectionstyle="arc3,rad=0",
        linestyle=(0, (4, 3)) if dashed else "solid",
        shrinkA=3,
        shrinkB=3,
        zorder=1,
    )
    ax.add_patch(patch)
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=6.8,
            color=PALETTE["muted"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.0},
            zorder=4,
        )


def _l_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    via: str = "horizontal_first",
    label: str | None = None,
    dashed: bool = False,
    lw: float = 1.0,
) -> None:
    """Draw an L-shaped (polyline) arrow with a sharp corner.

    via="horizontal_first": go horizontal from start to (end.x, start.y), then
    vertical up/down to end.
    via="vertical_first": go vertical first, then horizontal.
    The label, when supplied, is placed at the midpoint of the longer leg.
    """
    if via == "horizontal_first":
        corner = (end[0], start[1])
    else:
        corner = (start[0], end[1])

    seg_a = FancyArrowPatch(
        start,
        corner,
        arrowstyle="-",
        linewidth=lw,
        color=PALETTE["line"],
        connectionstyle="arc3,rad=0",
        linestyle=(0, (4, 3)) if dashed else "solid",
        shrinkA=3,
        shrinkB=0,
        zorder=1,
    )
    seg_b = FancyArrowPatch(
        corner,
        end,
        arrowstyle="-|>",
        mutation_scale=9.5,
        linewidth=lw,
        color=PALETTE["line"],
        connectionstyle="arc3,rad=0",
        linestyle=(0, (4, 3)) if dashed else "solid",
        shrinkA=0,
        shrinkB=3,
        zorder=1,
    )
    ax.add_patch(seg_a)
    ax.add_patch(seg_b)

    if label:
        leg_a_len = abs(corner[0] - start[0]) + abs(corner[1] - start[1])
        leg_b_len = abs(end[0] - corner[0]) + abs(end[1] - corner[1])
        if leg_a_len >= leg_b_len:
            mx = (start[0] + corner[0]) / 2
            my = (start[1] + corner[1]) / 2
        else:
            mx = (corner[0] + end[0]) / 2
            my = (corner[1] + end[1]) / 2
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=6.8,
            color=PALETTE["muted"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.0},
            zorder=4,
        )


def _section_label(ax: plt.Axes, x: float, y: float, text: str) -> None:
    ax.text(
        x,
        y,
        text,
        ha="left",
        va="top",
        fontsize=8,
        color=PALETTE["muted"],
        weight="bold",
    )


def _save(fig: plt.Figure, name: str) -> None:
    for out_dir in (REPORT_FIGURES, DOC_FIGURES):
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    if name == "architecture_pipeline":
        fig.savefig(REPORT_FIGURES / "pipeline.pdf", bbox_inches="tight")


def figure_pipeline() -> None:
    fig, ax = _setup(11.6, 4.7)
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 4.7)

    ax.text(
        0.25,
        4.50,
        "CG-Tutor Current Pipeline",
        ha="left",
        va="top",
        fontsize=13.5,
        weight="bold",
        color="#1F2328",
    )
    ax.text(
        0.25,
        4.18,
        "Stage-level view of the current codebase: solid edges are the forward pipeline; dashed edges are evidence-driven feedback.",
        ha="left",
        va="top",
        fontsize=7.8,
        color=PALETTE["muted"],
    )

    # Main pipeline row (1..5, 7). Evidence Loop is #6 below.
    nodes = [
        Node("input", "Concept Input", "configs/concepts\nordinary YAML", 0.25, 2.55, 1.42, 0.95, "input"),
        Node("plan", "Planning Agents", "decomposer\nprofile\nstoryboard", 1.95, 2.55, 1.55, 0.95, "agent"),
        Node("spec", "Success & Contracts", "auto_success_spec\nscene_ir\nvisual_contract", 3.78, 2.55, 1.72, 0.95, "spec"),
        Node("code", "Scene Generation", "blender_coder\ncompiler scaffold\nscene.py", 5.82, 2.55, 1.68, 0.95, "agent"),
        Node("render", "Gate & Render", "verifier · contract\npreview · Blender", 7.78, 2.55, 1.62, 0.95, "runtime"),
        # Evidence loop extended rightward so output's center column also has
        # a valid anchor on its top edge (the previous box ended at x=8.70 but
        # output's center sits at x≈10.50, leaving the selection-signals arrow
        # dangling in mid-air).
        Node("evidence", "Evidence Loop", "critic ensemble · metrics\ncross-ref · repair_plan", 4.10, 0.65, 6.80, 1.05, "evidence"),
        Node("output", "Selection & Export", "critic_best\nfinal*.mp4\nartifacts", 9.72, 2.55, 1.55, 0.95, "output"),
    ]
    by_key = {node.key: node for node in nodes}

    # Manual numbering so Evidence Loop is #6 and Output is #7.
    numbering = {"input": 1, "plan": 2, "spec": 3, "code": 4, "render": 5, "evidence": 6, "output": 7}
    for node in nodes:
        _draw_node(ax, node, numbering.get(node.key))

    # Forward chain (solid).
    main = ["input", "plan", "spec", "code", "render", "output"]
    for a, b in zip(main, main[1:]):
        _arrow(ax, by_key[a].right, by_key[b].left)

    # Render -> Evidence: straight short drop on the right side of the loop box.
    ev = by_key["evidence"]
    rd = by_key["render"]
    cd = by_key["code"]
    out = by_key["output"]

    # Render -> Evidence: straight vertical drop. Endpoint sits at the top
    # edge of evidence directly below render so the line is purely vertical.
    _arrow(
        ax,
        (rd.x + rd.w * 0.5, rd.y),
        (rd.x + rd.w * 0.5, ev.y + ev.h),
        label="sampled frames + reports",
    )

    # Evidence -> Code: straight vertical retry feedback. The start sits on
    # evidence top edge directly below code's center.
    _arrow(
        ax,
        (cd.x + cd.w * 0.5, ev.y + ev.h),
        (cd.x + cd.w * 0.5, cd.y),
        label="retry addendum",
        dashed=True,
    )

    # Evidence -> Output: straight vertical selection signal directly below
    # output's center.
    _arrow(
        ax,
        (out.x + out.w * 0.5, ev.y + ev.h),
        (out.x + out.w * 0.5, out.y),
        label="selection signals",
        dashed=True,
    )

    # Legend strip placed under the Evidence Loop in a single horizontal row
    # so it never overlaps with the loop box, regardless of how wide the loop
    # extends.
    legend_y = 0.32
    ax.text(
        0.30,
        legend_y,
        "Hard gates:",
        ha="left",
        va="center",
        fontsize=7.6,
        color=PALETTE["muted"],
        weight="bold",
    )
    ax.text(
        1.55,
        legend_y,
        "structural_fatal · success_hard",
        ha="left",
        va="center",
        fontsize=7.2,
        color=PALETTE["muted"],
    )
    ax.text(
        5.30,
        legend_y,
        "Soft signals:",
        ha="left",
        va="center",
        fontsize=7.6,
        color=PALETTE["muted"],
        weight="bold",
    )
    ax.text(
        6.55,
        legend_y,
        "success_soft · aesthetic_warn",
        ha="left",
        va="center",
        fontsize=7.2,
        color=PALETTE["muted"],
    )

    _save(fig, "architecture_pipeline")
    plt.close(fig)


def figure_feedback_loop() -> None:
    fig, ax = _setup(9.6, 5.4)
    ax.set_xlim(0, 9.6)
    ax.set_ylim(0, 5.4)

    ax.text(
        0.25,
        5.20,
        "Diagnosis and Repair Loop",
        ha="left",
        va="top",
        fontsize=13.5,
        weight="bold",
    )
    ax.text(
        0.25,
        4.88,
        "Evidence flows render → critic → metrics → repair, with fallback kept strictly diagnostic. No edge crosses another node.",
        ha="left",
        va="top",
        fontsize=7.8,
        color=PALETTE["muted"],
    )

    # Three tiers, each tier shifts down by ~1.55. All vertical connections are
    # short and aligned to a column, all horizontals stay within a single tier.
    nodes = [
        # Top tier: forward path, scene → ... → best.
        Node("scene", "Candidate", "scene.py\nor diff repair", 0.30, 3.30, 1.40, 0.95, "runtime"),
        Node("static", "Static Gates", "scene_verifier\ncontract_validator\nscene_ir verify", 1.95, 3.30, 1.65, 0.95, "evidence"),
        Node("preview", "Preview", "sampled keyframes\nmotion · visibility", 3.85, 3.30, 1.50, 0.95, "runtime"),
        Node("render", "Render", "full frames\nor fallback video", 5.60, 3.30, 1.45, 0.95, "runtime"),
        Node("best", "Best Selection", "hard isolation\nsemantic tie-break\nfinal_status", 7.30, 3.30, 1.85, 0.95, "output"),
        # Middle tier: evidence directly under render and static, fed by them.
        Node("fallback", "Fallback", "compiled scaffold\nfallback_degraded", 1.95, 1.80, 1.65, 0.95, "spec"),
        Node("critic", "Critic Evidence", "member usable\npartial success\nblock / warn", 5.60, 1.80, 1.45, 0.95, "evidence"),
        # Bottom tier: repair plan under fallback, metrics under critic.
        Node("repair", "Repair Plan", "max 6 targets\nsource_report\nminimal fix", 1.95, 0.30, 1.65, 0.95, "agent"),
        Node("metrics", "Static Evidence", "concept_metrics\nauto spec status\ncross-ref", 5.60, 0.30, 1.45, 0.95, "evidence"),
    ]
    by_key = {node.key: node for node in nodes}
    for node in nodes:
        _draw_node(ax, node)

    # Forward top chain (solid).
    for a, b in [("scene", "static"), ("static", "preview"), ("preview", "render"), ("render", "best")]:
        _arrow(ax, by_key[a].right, by_key[b].left)

    # Vertical drops (all straight, label at exact midpoint).
    _arrow(
        ax,
        by_key["render"].bottom,
        by_key["critic"].top,
        label="frames + reports",
    )
    _arrow(
        ax,
        by_key["static"].bottom,
        by_key["fallback"].top,
        label="fatal path",
        dashed=True,
    )
    _arrow(
        ax,
        by_key["critic"].bottom,
        by_key["metrics"].top,
        label="cross-ref",
    )

    # Horizontal in the bottom tier (right→left): metrics → repair.
    _arrow(
        ax,
        by_key["metrics"].left,
        by_key["repair"].right,
        label="repair targets",
    )

    # L-shaped feedback edges (折线): no curves, every segment is orthogonal,
    # corners chosen so segments hug the white space between rows/columns.
    #
    # repair → scene: left along the bottom, then up the left margin.
    _l_arrow(
        ax,
        (by_key["repair"].x, by_key["repair"].y + by_key["repair"].h * 0.5),
        (by_key["scene"].x + by_key["scene"].w * 0.5, by_key["scene"].y),
        via="horizontal_first",
        label="retry",
        dashed=True,
    )
    # fallback → preview: right along the inter-row gap, then up into preview.
    _l_arrow(
        ax,
        (by_key["fallback"].x + by_key["fallback"].w, by_key["fallback"].y + by_key["fallback"].h * 0.5),
        (by_key["preview"].x + by_key["preview"].w * 0.5, by_key["preview"].y),
        via="horizontal_first",
        label="diagnose",
        dashed=True,
    )
    # critic → best: right along the inter-row gap, then up into best.
    _l_arrow(
        ax,
        (by_key["critic"].x + by_key["critic"].w, by_key["critic"].y + by_key["critic"].h * 0.5),
        (by_key["best"].x + by_key["best"].w * 0.35, by_key["best"].y),
        via="horizontal_first",
        label="failure_class",
        dashed=True,
    )

    _save(fig, "architecture_feedback_loop")
    plt.close(fig)


def figure_module_layers() -> None:
    fig, ax = _setup(12.2, 5.0)
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 5.0)

    ax.text(
        0.30,
        4.82,
        "Codebase Module Boundaries",
        ha="left",
        va="top",
        fontsize=13.5,
        weight="bold",
    )
    ax.text(
        0.30,
        4.50,
        "Top row = source modules grouped by responsibility. Bottom row = downstream artifacts, each directly anchored to its owner column.",
        ha="left",
        va="top",
        fontsize=7.8,
        color=PALETTE["muted"],
    )

    # Canvas widened so each column gets a ~0.40-unit gap, leaving room for
    # visible inter-column arrows (previously the 0.15-unit gap clipped them).
    columns = [
        Node("cli", "Entry & Config", "run_concept.py\nmodels_*.yaml\nconcept YAML", 0.30, 2.60, 1.55, 1.30, "input"),
        Node("agents", "LLM Agents", "decomposer · profile\nstoryboard\ncoder · critic", 2.25, 2.60, 1.55, 1.30, "agent"),
        Node("struct", "Typed Structure", "success_spec\nauto_success_spec\nscene_ir\nvisual_contract", 4.20, 2.60, 1.70, 1.30, "spec"),
        Node("runtime", "Execution", "scene_compiler\nscene_verifier\npreview · runtime\ncompose", 6.30, 2.60, 1.55, 1.30, "runtime"),
        Node("diag", "Diagnostics", "concept_metrics\ncross_reference\ncritic_loop\nrepair_plan\nfailure_memory", 8.25, 2.60, 1.70, 1.30, "evidence"),
        Node("art", "Artifacts", "success_spec.*\ncritic_iter*.json\nrepair_plan*.json\nfinal*.mp4", 10.35, 2.60, 1.55, 1.30, "output"),
    ]
    for node in columns:
        _draw_node(ax, node)

    for a, b in zip(columns, columns[1:]):
        _arrow(ax, a.right, b.left)

    # Anchor each artifact directly under its source column so the dashed
    # ownership edges are straight verticals — no crossings.
    by_key = {node.key: node for node in columns}
    lower = [
        Node(
            "tests",
            "Regression Tests",
            "545 passed\n3 skipped",
            by_key["struct"].x,
            0.70,
            by_key["struct"].w,
            0.95,
            "output",
        ),
        Node(
            "repo",
            "Retained Snapshot",
            "5 concepts\n25 MP4s\nno PNG frames",
            by_key["diag"].x,
            0.70,
            by_key["diag"].w,
            0.95,
            "output",
        ),
        Node(
            "docs",
            "Paper Docs",
            "README\narchitecture overview\nexperiment report",
            by_key["art"].x,
            0.70,
            by_key["art"].w,
            0.95,
            "output",
        ),
    ]
    for node in lower:
        _draw_node(ax, node)

    lower_by_key = {node.key: node for node in lower}
    for upper, lower_key, label in [
        (by_key["struct"], "tests", "verified by"),
        (by_key["diag"], "repo", "evaluated on"),
        (by_key["art"], "docs", "documented in"),
    ]:
        target = lower_by_key[lower_key]
        _arrow(
            ax,
            (upper.x + upper.w * 0.5, upper.y),
            (target.x + target.w * 0.5, target.y + target.h),
            label=label,
            dashed=True,
        )

    ax.text(
        0.30,
        0.30,
        "Ownership boundaries only — internal call edges are intentionally omitted.",
        ha="left",
        va="bottom",
        fontsize=7.2,
        color=PALETTE["muted"],
        style="italic",
    )
    _save(fig, "architecture_module_layers")
    plt.close(fig)


def main() -> None:
    _validate_current_repo()
    figure_pipeline()
    figure_feedback_loop()
    figure_module_layers()
    print(f"Wrote figures to {REPORT_FIGURES} and {DOC_FIGURES}")


if __name__ == "__main__":
    main()
