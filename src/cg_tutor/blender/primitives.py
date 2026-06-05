"""Reusable bpy helpers for scene authoring.

Import this module from inside a bpy script (run under Blender). Do NOT
import from regular Python; bpy is only present inside Blender.

The Agent (or hand-written scenes) calls these helpers instead of raw
bpy.ops calls, which sharply reduces error rate compared to LLMs writing
ad-hoc Blender code.
"""

from __future__ import annotations

# These imports only resolve inside Blender's Python.
import bpy  # type: ignore
import math
from mathutils import Vector  # type: ignore


# ----- scene reset ---------------------------------------------------------

def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


# ----- primitives ---------------------------------------------------------

def create_sphere(
    name: str,
    location=(0, 0, 0),
    radius: float = 1.0,
    color=(0.8, 0.2, 0.2, 1.0),
    smooth: bool = True,
):
    bpy.ops.mesh.primitive_uv_sphere_add(location=location, radius=radius)
    obj = bpy.context.active_object
    obj.name = name
    if smooth:
        bpy.ops.object.shade_smooth()
    _assign_material(obj, color)
    return obj


def create_plane(name: str, location=(0, 0, 0), size: float = 10.0,
                 color=(0.5, 0.5, 0.5, 1.0)):
    bpy.ops.mesh.primitive_plane_add(location=location, size=size)
    obj = bpy.context.active_object
    obj.name = name
    _assign_material(obj, color)
    return obj


def create_cube(name: str, location=(0, 0, 0), size: float = 1.0,
                color=(0.5, 0.5, 0.5, 1.0)):
    bpy.ops.mesh.primitive_cube_add(location=location, size=size)
    obj = bpy.context.active_object
    obj.name = name
    _assign_material(obj, color)
    return obj


# ----- lights & camera ----------------------------------------------------

def create_light(name: str, kind: str = "POINT", location=(0, 0, 5),
                 energy: float = 1000.0, color=(1, 1, 1)):
    bpy.ops.object.light_add(type=kind, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.energy = energy
    obj.data.color = color
    return obj


def create_camera(name: str, location=(7, -7, 5), look_at=(0, 0, 0),
                  fov_deg: float = 50.0):
    bpy.ops.object.camera_add(location=location)
    cam = bpy.context.active_object
    cam.name = name
    cam.data.lens_unit = "FOV"
    cam.data.angle = math.radians(fov_deg)
    _aim_at(cam, look_at)
    bpy.context.scene.camera = cam
    return cam


def _aim_at(obj, target) -> None:
    direction = Vector(target) - obj.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    obj.rotation_euler = rot_quat.to_euler()


# ----- keyframes ----------------------------------------------------------

def keyframe(obj, attr: str, value, frame: int) -> None:
    """Set obj.<attr> = value and insert a keyframe at the given frame.

    `attr` may be dotted (e.g. "location" or "data.energy").
    """
    path = attr.split(".")
    target = obj
    for p in path[:-1]:
        target = getattr(target, p)
    setattr(target, path[-1], value)
    target.keyframe_insert(data_path=path[-1], frame=frame)


# ----- material ----------------------------------------------------------

def _assign_material(obj, color) -> None:
    mat = bpy.data.materials.new(name=f"{obj.name}_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
    obj.data.materials.append(mat)


# ----- render config ------------------------------------------------------

def configure_render(
    *,
    engine: str = "BLENDER_EEVEE",
    resolution=(1280, 720),
    fps: int = 30,
    samples: int = 32,
    out_pattern: str = "//frames/frame_####.png",
):
    """Configure render. Default engine `BLENDER_EEVEE` works on Blender 4.0+;
    pass `BLENDER_EEVEE_NEXT` on 4.2+ for higher quality, or `CYCLES` for
    final renders. Unknown engines fall back to BLENDER_EEVEE rather than
    crashing — older Blenders may have renamed the enum.
    """
    scene = bpy.context.scene
    try:
        scene.render.engine = engine
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.fps = fps
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = out_pattern
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = samples
    elif hasattr(scene, "eevee"):
        for attr in ("taa_render_samples", "samples"):
            if hasattr(scene.eevee, attr):
                setattr(scene.eevee, attr, samples)
                break


def set_frame_range(start: int, end: int) -> None:
    scene = bpy.context.scene
    scene.frame_start = start
    scene.frame_end = end


def render_animation() -> None:
    bpy.ops.render.render(animation=True)
