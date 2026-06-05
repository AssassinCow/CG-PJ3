"""Hand-written, parameterised Blender scene template for lighting concepts.

The Agent generates a storyboard JSON; this script reads it (via env var)
and materialises a scene + animation + render.

Run via:
    blender -b --factory-startup -P lighting_scene.py
with env:
    CG_TUTOR_STORYBOARD=/abs/path/to/storyboard.json
    CG_TUTOR_OUT_DIR=/abs/path/to/frames

Supported object primitives: sphere, plane, cube.
Supported light kinds: POINT, SUN, AREA (uppercase to match bpy enum).
Supported keyframe attrs: location, data.energy, data.color,
  material.diffuse_color.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

from mathutils import Vector  # type: ignore


# ---------- bootstrap import path so we can reuse `primitives` ------------

THIS_FILE = Path(__file__).resolve()
# repo_root/src/cg_tutor/blender/templates/lighting_scene.py
SRC_ROOT = THIS_FILE.parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cg_tutor.blender.primitives import (  # noqa: E402
    configure_render,
    create_camera,
    create_cube,
    create_light,
    create_plane,
    create_sphere,
    keyframe,
    render_animation,
    reset_scene,
    set_frame_range,
)


# ---------- utilities ----------------------------------------------------


def _read_storyboard() -> dict:
    sb_path = os.environ.get("CG_TUTOR_STORYBOARD")
    if not sb_path:
        raise RuntimeError("CG_TUTOR_STORYBOARD env var not set")
    with open(sb_path) as fh:
        return json.load(fh)


def _out_dir() -> Path:
    out = os.environ.get("CG_TUTOR_OUT_DIR")
    if not out:
        raise RuntimeError("CG_TUTOR_OUT_DIR env var not set")
    p = Path(out)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _spawn_object(spec: dict):
    name = spec["name"]
    kind = spec["type"]
    loc = tuple(spec.get("location", (0, 0, 0)))
    props = spec.get("properties", {})
    color = tuple(props.get("color", (0.8, 0.2, 0.2, 1.0)))
    if len(color) == 3:
        color = (*color, 1.0)

    if kind == "primitive" or kind == "mesh":
        primitive = spec.get("primitive", "sphere")
        if primitive == "sphere":
            return create_sphere(name, location=loc,
                                 radius=props.get("radius", 1.0),
                                 color=color)
        if primitive == "plane":
            return create_plane(name, location=loc,
                                size=props.get("size", 10.0),
                                color=color)
        if primitive == "cube":
            return create_cube(name, location=loc,
                               size=props.get("size", 1.0),
                               color=color)
        raise ValueError(f"unsupported primitive {primitive!r}")

    if kind == "light":
        return create_light(
            name,
            kind=props.get("light_kind", "POINT"),
            location=loc,
            energy=props.get("energy", 1000.0),
            color=tuple(props.get("light_color", (1, 1, 1))),
        )

    raise ValueError(f"unsupported object type {kind!r}")


def _apply_keyframes(obj, kfs: list[dict], fps: int) -> None:
    for kf in kfs:
        frame = int(round(kf["time_sec"] * fps)) + 1
        attr = kf["attr"]
        value = kf["value"]
        keyframe(obj, attr, value, frame)


def _setup_camera_for_shot(shot: dict, fps: int):
    cam_keys = shot["camera"]
    if not cam_keys:
        raise RuntimeError(f"shot {shot.get('node_id')} has no camera keys")
    first = cam_keys[0]
    cam = create_camera(
        name=f"cam_{shot['node_id']}",
        location=tuple(first["position"]),
        look_at=tuple(first["look_at"]),
        fov_deg=first.get("fov", 50.0),
    )
    # animate further keys
    for key in cam_keys:
        frame = int(round(key["time_sec"] * fps)) + 1
        cam.location = Vector(key["position"])
        cam.keyframe_insert(data_path="location", frame=frame)
        # Aim per-frame: rotate to look at target.
        direction = Vector(key["look_at"]) - Vector(key["position"])
        rot_euler = direction.to_track_quat("-Z", "Y").to_euler()
        cam.rotation_euler = rot_euler
        cam.keyframe_insert(data_path="rotation_euler", frame=frame)
        cam.data.angle = math.radians(key.get("fov", 50.0))
        cam.data.keyframe_insert(data_path="lens", frame=frame)
    return cam


# ---------- main build ---------------------------------------------------


def build() -> None:
    storyboard = _read_storyboard()
    out_dir = _out_dir()
    fps = int(storyboard.get("fps", 30))
    res = tuple(storyboard.get("resolution", (1280, 720)))

    reset_scene()
    configure_render(
        engine=os.environ.get("CG_TUTOR_ENGINE", "BLENDER_EEVEE"),
        resolution=res,
        fps=fps,
        samples=int(os.environ.get("CG_TUTOR_SAMPLES", "32")),
        out_pattern=str(out_dir / "frame_####.png"),
    )

    total_duration = sum(s["duration_sec"] for s in storyboard["shots"])
    total_frames = int(round(total_duration * fps))
    set_frame_range(1, total_frames)

    shots = storyboard["shots"]
    for shot in shots:
        # Spawn this shot's objects (we treat the scene as a union of all shots
        # for W1 simplicity; later shots can deactivate earlier objects via
        # keyframes on `hide_render`).
        for obj_spec in shot["objects"]:
            obj = _spawn_object(obj_spec)
            _apply_keyframes(obj, obj_spec.get("keyframes", []), fps)

        # Each shot has its own camera; for multi-shot videos we'd swap
        # scene.camera over time. W1 supports a single shot; if multiple,
        # the last one wins (good enough for verification).
        _setup_camera_for_shot(shot, fps)

    render_animation()


if __name__ == "__main__":
    build()
