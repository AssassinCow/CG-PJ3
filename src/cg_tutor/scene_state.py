"""Lightweight scene-state audit artifacts.

This is intentionally static for now: it reads scene.py and records object
names plus keyframed channels without launching Blender. The schema is shaped
so a future Blender-backed render-space inspector can add screen bboxes and
visibility samples without changing downstream artifact consumers.
"""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field

from cg_tutor.critic_cross_reference import (
    _all_created_object_names,
    _keyframe_schedule,
)


@dataclass(frozen=True)
class SceneStateKeyframe:
    object_name: str
    data_path: str
    frame: int
    value: object | None = None


@dataclass
class SceneStateReport:
    source: str = "ast_static"
    ok: bool = True
    error: str = ""
    object_names: list[str] = field(default_factory=list)
    keyframes: list[SceneStateKeyframe] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "ok": self.ok,
            "error": self.error,
            "object_names": self.object_names,
            "keyframes": [asdict(k) for k in self.keyframes],
            "metrics": self.metrics,
        }


def inspect_scene_code(scene_code: str) -> SceneStateReport:
    try:
        tree = ast.parse(scene_code)
    except SyntaxError as exc:
        return SceneStateReport(ok=False, error=f"syntax_error: {exc.msg}")

    object_names = sorted(_all_created_object_names(tree))
    schedule = _keyframe_schedule(tree)
    keyframes: list[SceneStateKeyframe] = []
    for object_name, events in sorted(schedule.items()):
        for data_path, frame, value in events:
            keyframes.append(SceneStateKeyframe(
                object_name=object_name,
                data_path=data_path,
                frame=frame,
                value=value,
            ))

    hide_paths = {"hide_render", "hide_viewport"}
    camera_names = {"cam", "camera", "render_camera", "main_camera", "Camera"}
    non_visibility = [k for k in keyframes if k.data_path not in hide_paths]
    non_camera = [
        k for k in non_visibility
        if k.object_name not in camera_names
        and not k.object_name.lower().endswith("camera")
    ]
    metrics = {
        "object_count": len(object_names),
        "keyframe_count": len(keyframes),
        "non_visibility_keyframe_count": len(non_visibility),
        "non_camera_keyframe_count": len(non_camera),
        "animated_objects": sorted({k.object_name for k in non_visibility}),
        "animated_non_camera_objects": sorted({k.object_name for k in non_camera}),
        "animated_channels": sorted({
            f"{k.object_name}.{k.data_path}" for k in non_visibility
        }),
    }
    return SceneStateReport(
        object_names=object_names,
        keyframes=keyframes,
        metrics=metrics,
    )


def scene_state_report_to_json(report: SceneStateReport) -> str:
    return json.dumps(report.to_dict(), indent=2)
