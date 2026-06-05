"""Semi-deterministic Scene IR / storyboard -> bpy compiler.

The compiler intentionally covers stable visual grammar, not complete
concept-specific scenes. It emits a runnable scaffold that the LLM coder can
preserve and enrich: render boilerplate, visibility windows, keyframes, and
reusable teaching helpers such as labeled points, thin tracer lines, image
plane marks, and bracket annotations.
"""

from __future__ import annotations

from copy import deepcopy
from pprint import pformat
import re

from cg_tutor.schemas import Storyboard


_RUNTIME_TEMPLATE = r'''import math
import os
from pathlib import Path

import bpy
from mathutils import Vector


STORYBOARD = __STORYBOARD_JSON__
COMPILED_SCENE_HINTS = __COMPILED_SCENE_HINTS_JSON__


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def frame_for_time(time_sec):
    return round(float(time_sec) * STORYBOARD['fps']) + 1


def lens_for_fov(fov_deg, sensor_mm=36.0):
    angle = math.radians(max(5.0, min(160.0, float(fov_deg))))
    return sensor_mm / (2.0 * math.tan(angle / 2.0))


def make_mat(name, color, roughness=0.45, emission=0.0, alpha=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    rgba = tuple(color[:3]) + (float(alpha) if alpha is not None else float(color[3] if len(color) > 3 else 1.0),)
    if bsdf:
        bsdf.inputs['Base Color'].default_value = rgba
        bsdf.inputs['Roughness'].default_value = roughness
        if 'Alpha' in bsdf.inputs:
            bsdf.inputs['Alpha'].default_value = rgba[3]
        if emission > 0:
            bsdf.inputs['Emission Color'].default_value = rgba
            bsdf.inputs['Emission Strength'].default_value = emission
    if rgba[3] < 0.999:
        mat.blend_method = 'BLEND'
        mat.use_nodes = True
        if hasattr(mat, 'show_transparent_back'):
            mat.show_transparent_back = True
    return mat


def color_from_props(props, fallback=(0.8, 0.84, 0.9, 1.0)):
    raw = props.get('color') or props.get('base_color') or props.get('light_color')
    if isinstance(raw, list) and len(raw) >= 3:
        alpha = float(props.get('alpha', raw[3] if len(raw) > 3 else 1.0))
        return (float(raw[0]), float(raw[1]), float(raw[2]), alpha)
    return fallback


def set_path_value(obj, attr, value):
    target = obj
    parts = attr.split('.')
    for part in parts[:-1]:
        target = getattr(target, part)
    leaf = parts[-1]
    if leaf in {'location', 'scale', 'rotation_euler'} and isinstance(value, list):
        value = tuple(value)
    setattr(target, leaf, value)
    return target, leaf


def apply_initial_props(obj, spec):
    props = spec.get('properties') or {}
    name = spec.get('name', '').lower()
    if obj.type == 'LIGHT':
        if hasattr(obj.data, 'size') and 'size' in props:
            obj.data.size = float(props['size'])
        if hasattr(obj.data, 'shadow_soft_size') and 'shadow_soft_size' in props:
            obj.data.shadow_soft_size = float(props['shadow_soft_size'])
        if 'energy' in props:
            obj.data.energy = float(props['energy'])
        if 'light_color' in props:
            obj.data.color = tuple(props['light_color'][:3])
        return
    if 'rotation_euler' in props:
        obj.rotation_euler = tuple(props['rotation_euler'])
    elif spec.get('primitive') == 'plane' and 'image_plane' in name:
        obj.rotation_euler = (0.0, math.radians(90), 0.0)
    if 'scale' in props:
        obj.scale = tuple(props['scale'])
    if props.get('wireframe'):
        obj.display_type = 'WIRE'
        if hasattr(obj, 'show_wire'):
            obj.show_wire = True


def apply_spec_keyframes(obj, spec):
    for key in spec.get('keyframes') or []:
        frame = frame_for_time(key['time_sec'])
        target, leaf = set_path_value(obj, key['attr'], key['value'])
        target.keyframe_insert(data_path=leaf, frame=frame)


def add_text(name, location, props):
    body = props.get('text') or props.get('label') or name
    bpy.ops.object.text_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.body = str(body)
    obj.data.align_x = 'CENTER'
    obj.data.align_y = 'CENTER'
    obj.data.size = float(props.get('size', 0.22))
    obj.rotation_euler = tuple(props.get('rotation_euler', [math.radians(65), 0, 0]))
    mat = make_mat(
        'mat_' + name,
        color_from_props(props, (1.0, 0.96, 0.72, 1.0)),
        emission=float(props.get('emission', 0.15)),
    )
    obj.data.materials.append(mat)
    return obj


def add_curve_polyline(name, points, props):
    pts = [Vector(p) for p in points if len(p) >= 3]
    if len(pts) < 2:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.04, location=tuple(pts[0] if pts else Vector((0, 0, 0))))
        obj = bpy.context.object
        obj.name = name
        obj.data.materials.append(make_mat('mat_' + name, color_from_props(props), emission=float(props.get('emission', 0.0))))
        return obj
    curve = bpy.data.curves.new(name, type='CURVE')
    curve.dimensions = '3D'
    curve.bevel_factor_mapping_start = 'SPLINE'
    curve.bevel_factor_mapping_end = 'SPLINE'
    curve.bevel_factor_start = 0.0
    curve.bevel_factor_end = float(props.get('bevel_factor_end', 1.0))
    curve.resolution_u = 2
    curve.bevel_depth = float(props.get('bevel_depth', props.get('radius', 0.018)))
    curve.bevel_resolution = int(props.get('bevel_resolution', 3))
    spl = curve.splines.new('POLY')
    spl.points.add(len(pts) - 1)
    for point, co in zip(spl.points, pts):
        point.co = (co.x, co.y, co.z, 1.0)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(make_mat(
        'mat_' + name,
        color_from_props(props, (0.35, 0.7, 1.0, float(props.get('alpha', 1.0)))),
        emission=float(props.get('emission', 0.08)),
        alpha=props.get('alpha'),
    ))
    return obj


def add_line_between(name, start, end, props=None):
    props = props or {}
    return add_curve_polyline(name, [list(start), list(end)], props)


def add_cube_between(name, start, end, props):
    start = Vector(start)
    end = Vector(end)
    direction = end - start
    length = max(direction.length, 1e-4)
    center = start + direction * 0.5
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = (length, float(props.get('thickness', 0.035)), float(props.get('thickness', 0.035)))
    obj.rotation_euler = direction.to_track_quat('X', 'Z').to_euler()
    obj.data.materials.append(make_mat('mat_' + name, color_from_props(props), alpha=props.get('alpha')))
    return obj


def add_vector_shaft_head(name, location, props):
    color = color_from_props(props, (0.15, 0.45, 1.0, 1.0))
    mat = make_mat('mat_' + name, color, roughness=0.35)
    length = float(props.get('length', 1.2))
    radius = float(props.get('radius', 0.035))
    direction = Vector(props.get('direction', [0, 0, 1]))
    if direction.length == 0:
        direction = Vector((0, 0, 1))
    direction.normalize()
    loc = Vector(location)
    mid = loc + direction * (length * 0.45)
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=radius, depth=length * 0.75, location=mid)
    shaft = bpy.context.object
    shaft.name = name
    shaft.data.materials.append(mat)
    shaft.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
    tip = loc + direction * length
    bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=radius * 3.0, radius2=0.0, depth=length * 0.25, location=tip)
    head = bpy.context.object
    head.name = name + '_head'
    head.parent = shaft
    head.data.materials.append(mat)
    head.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
    label = props.get('label')
    if label:
        add_text(name + '_label', tip + direction * 0.25 + Vector((0, 0, 0.12)), {
            'text': label,
            'size': float(props.get('label_size', 0.28)),
            'color': list(color[:3]),
        }).parent = shaft
    return shaft


def add_group_parent(name, location):
    obj = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(obj)
    obj.empty_display_type = 'PLAIN_AXES'
    obj.location = tuple(location)
    return obj


def add_point_group(name, primitive, location, props):
    parent = add_group_parent(name, location)
    points = props.get('points') or []
    radius = float(props.get('radius', props.get('size', 0.08)))
    color = color_from_props(props, (1.0, 0.82, 0.2, 1.0))
    for idx, point in enumerate(points):
        if primitive == 'cube_group':
            bpy.ops.mesh.primitive_cube_add(size=radius, location=tuple(point))
        else:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=radius, location=tuple(point))
        child = bpy.context.object
        child.name = f'{name}_{idx:02d}'
        child.parent = parent
        child.data.materials.append(make_mat('mat_' + child.name, color, alpha=props.get('alpha')))
    return parent


def add_object(spec):
    name = spec['name']
    primitive = spec.get('primitive')
    kind = spec.get('type')
    props = spec.get('properties') or {}
    loc = tuple(spec.get('location') or [0, 0, 0])
    if kind == 'light':
        light_kind = props.get('light_kind', 'AREA')
        data = bpy.data.lights.new(name, type=light_kind)
        data.energy = float(props.get('energy', 1200.0))
        if hasattr(data, 'size') and 'size' in props:
            data.size = float(props['size'])
        if hasattr(data, 'shadow_soft_size') and 'shadow_soft_size' in props:
            data.shadow_soft_size = float(props['shadow_soft_size'])
        if 'light_color' in props:
            data.color = tuple(props['light_color'][:3])
        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
        obj.location = loc
        apply_initial_props(obj, spec)
        return obj
    if kind in ('annotation', 'text') or primitive == 'empty':
        return add_text(name, loc, props)
    if primitive == 'curve_polyline':
        points = props.get('points') or props.get('control_points') or []
        if not points:
            points = [loc, (loc[0] + 1.0, loc[1], loc[2])]
        return add_curve_polyline(name, points, props)
    if primitive in ('sphere_group', 'cube_group'):
        return add_point_group(name, primitive, loc, props)
    if 'ray' in name.lower() and props.get('start') and props.get('end'):
        return add_line_between(name, props['start'], props['end'], props)
    if primitive == 'sphere':
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=float(props.get('radius', 0.7)), location=loc)
    elif primitive == 'plane':
        bpy.ops.mesh.primitive_plane_add(size=float(props.get('size', 6.0)), location=loc)
    elif primitive == 'cube':
        if props.get('start') and props.get('end'):
            return add_cube_between(name, props['start'], props['end'], props)
        bpy.ops.mesh.primitive_cube_add(size=float(props.get('size', 1.0)), location=loc)
    elif primitive == 'cylinder':
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=float(props.get('radius', 0.35)), depth=float(props.get('depth', 1.0)), location=loc)
    elif primitive == 'cone':
        bpy.ops.mesh.primitive_cone_add(vertices=48, radius1=float(props.get('radius', 0.4)), depth=float(props.get('depth', 1.0)), location=loc)
    elif primitive == 'torus':
        bpy.ops.mesh.primitive_torus_add(major_radius=float(props.get('major_radius', 0.7)), minor_radius=float(props.get('minor_radius', 0.08)), location=loc)
    elif primitive == 'arrow':
        return add_vector_shaft_head(name, loc, props)
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.25, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(make_mat('mat_' + name, color_from_props(props), alpha=props.get('alpha')))
    apply_initial_props(obj, spec)
    return obj


def iter_tree(obj):
    yield obj
    for child in obj.children:
        yield from iter_tree(child)


def set_visibility_windows(obj, ranges):
    if not ranges:
        return
    ranges = merge_visibility_ranges(ranges)
    for target in iter_tree(obj):
        target.hide_render = True
        target.hide_viewport = True
        target.keyframe_insert('hide_render', frame=1)
        target.keyframe_insert('hide_viewport', frame=1)
        for start, end in ranges:
            target.hide_render = False
            target.hide_viewport = False
            target.keyframe_insert('hide_render', frame=start)
            target.keyframe_insert('hide_viewport', frame=start)
            target.hide_render = True
            target.hide_viewport = True
            target.keyframe_insert('hide_render', frame=end + 1)
            target.keyframe_insert('hide_viewport', frame=end + 1)


def merge_visibility_ranges(ranges):
    merged = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def readable_label(name):
    low = name.lower()
    if 'camera_center' in low or 'pinhole' in low:
        return 'C'
    if 'projected_point' in low:
        return 'p(x,y)'
    if 'camera_center' in low:
        return 'C'
    if 'image_plane' in low:
        return 'image plane'
    if 'object_point' in low:
        if low.endswith('_b') or '_b_' in low:
            return 'B'
        if low.endswith('_a') or '_a_' in low:
            return 'P(X,Y,Z)'
        return '3D point'
    return None


def label_offset_for_name(name):
    low = name.lower()
    if 'image_plane' in low:
        return (0.0, 0.48, -0.34)
    if 'projected_point' in low:
        return (0.0, 0.34, -0.22)
    if 'camera_center' in low or 'pinhole' in low:
        return (0.0, -0.34, 0.22)
    if 'object_point' in low:
        return (0.0, -0.36, 0.26)
    return (0.0, -0.18, 0.24)


def add_label_for_object(base_name, obj, text, shot_id, offset=None):
    if offset is None:
        offset = label_offset_for_name(base_name)
    loc = obj.location + Vector(offset)
    label_name = f'auto_label_{shot_id}_{base_name}'
    return add_text(label_name, loc, {
        'text': text,
        'size': 0.2 if len(text) <= 8 else 0.17,
        'color': [1.0, 0.96, 0.72],
    })


def find_object_by_names(names, object_map):
    for name in names:
        if name in object_map:
            return name, object_map[name]
    for wanted in names:
        wanted_low = wanted.lower()
        for name, obj in object_map.items():
            if wanted_low in name.lower() or name.lower() in wanted_low:
                return name, obj
    return None, None


def add_projection_plane_marks(shot_id, image_plane, projected_point, object_map):
    center = projected_point.location
    x = image_plane.location.x
    y = center.y
    z = center.z
    size = 0.22
    props = {'color': [1.0, 0.9, 0.15], 'bevel_depth': 0.012, 'emission': 0.12}
    h = add_line_between(f'auto_projected_silhouette_h_{shot_id}', (x, y - size, z), (x, y + size, z), props)
    v = add_line_between(f'auto_projected_silhouette_v_{shot_id}', (x, y, z - size), (x, y, z + size), props)
    object_map[h.name] = h
    object_map[v.name] = v


def add_axis_glyph(shot_id, anchor, object_map):
    origin = anchor.location + Vector((-0.55, -0.62, -0.42))
    axes = [
        ('X', Vector((0.55, 0.0, 0.0)), [1.0, 0.35, 0.25]),
        ('Y', Vector((0.0, 0.45, 0.0)), [0.35, 0.9, 0.45]),
        ('Z', Vector((0.0, 0.0, 0.5)), [0.45, 0.65, 1.0]),
    ]
    for label, delta, color in axes:
        line = add_line_between(
            f'auto_axis_{label}_{shot_id}',
            origin,
            origin + delta,
            {'color': color, 'bevel_depth': 0.01, 'emission': 0.06},
        )
        text = add_text(
            f'auto_axis_{label}_label_{shot_id}',
            origin + delta * 1.15,
            {'text': label, 'size': 0.2, 'color': color},
        )
        object_map[line.name] = line
        object_map[text.name] = text


def reveal_curve(obj, start, end):
    if not hasattr(obj, 'data') or not hasattr(obj.data, 'bevel_factor_end'):
        return
    obj.data.bevel_factor_end = 0.03
    obj.data.keyframe_insert(data_path='bevel_factor_end', frame=start)
    obj.data.bevel_factor_end = 1.0
    obj.data.keyframe_insert(data_path='bevel_factor_end', frame=max(start + 2, end - 3))


def add_focal_bracket(shot_id, pinhole, image_plane, object_map):
    y = max(pinhole.location.y, image_plane.location.y) + 0.35
    z = min(pinhole.location.z, image_plane.location.z) - 0.38
    start = Vector((pinhole.location.x, y, z))
    end = Vector((image_plane.location.x, y, z))
    props = {'color': [0.95, 0.95, 0.85], 'bevel_depth': 0.015, 'emission': 0.08}
    main = add_line_between(f'auto_focal_bracket_{shot_id}', start, end, props)
    tick_a = add_line_between(f'auto_focal_bracket_tick_a_{shot_id}', start + Vector((0, 0, -0.12)), start + Vector((0, 0, 0.12)), props)
    tick_b = add_line_between(f'auto_focal_bracket_tick_b_{shot_id}', end + Vector((0, 0, -0.12)), end + Vector((0, 0, 0.12)), props)
    label = add_text(f'auto_focal_bracket_label_{shot_id}', (start + end) * 0.5 + Vector((0, 0, 0.18)), {'text': 'f', 'size': 0.22, 'color': [0.95, 0.95, 0.85]})
    for obj in (main, tick_a, tick_b, label):
        object_map[obj.name] = obj


def add_teaching_helpers_for_shot(shot, object_map, start_frame, end_frame):
    created = []
    shot_id = shot['node_id']
    shot_names = {spec['name'] for spec in shot.get('objects') or []}
    shot_objects = {
        name: object_map[name]
        for name in shot_names
        if name in object_map
    }
    for name in list(shot_names):
        obj = shot_objects.get(name)
        if obj is None:
            continue
        text = readable_label(name)
        if text:
            label = add_label_for_object(name, obj, text, shot_id)
            object_map[label.name] = label
            created.append(label.name)
    _, pinhole = find_object_by_names(['pinhole_aperture_C', 'pinhole', 'camera_center_marker'], shot_objects)
    _, image_plane = find_object_by_names(['translucent_image_plane', 'image_plane'], shot_objects)
    _, projected_point = find_object_by_names(['projected_point_a', 'projected_point'], shot_objects)
    point_items = [
        (name, shot_objects[name])
        for name in shot_names
        if name in shot_objects and 'object_point' in name.lower()
    ]
    if pinhole is not None and point_items:
        for point_name, point_obj in point_items[:3]:
            path = [tuple(point_obj.location), tuple(pinhole.location)]
            if projected_point is not None:
                path.append(tuple(projected_point.location))
            elif image_plane is not None:
                path.append(tuple(image_plane.location))
            ray = add_curve_polyline(
                f'auto_projection_ray_{shot_id}_{point_name}',
                path,
                {'color': [0.35, 0.7, 1.0], 'bevel_depth': 0.014, 'emission': 0.12},
            )
            reveal_curve(ray, start_frame, end_frame)
            object_map[ray.name] = ray
            created.append(ray.name)
    if pinhole is not None and image_plane is not None:
        before = set(object_map)
        add_focal_bracket(shot_id, pinhole, image_plane, object_map)
        created.extend(name for name in object_map if name not in before)
    if image_plane is not None and projected_point is not None:
        before = set(object_map)
        add_projection_plane_marks(shot_id, image_plane, projected_point, object_map)
        created.extend(name for name in object_map if name not in before)
    anchor = pinhole or image_plane or projected_point
    if anchor is not None:
        before = set(object_map)
        add_axis_glyph(shot_id, anchor, object_map)
        created.extend(name for name in object_map if name not in before)
    return created


bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
if scene.render.engine == 'CYCLES':
    scene.cycles.samples = 32
    if hasattr(scene.cycles, 'use_denoising'):
        scene.cycles.use_denoising = False
    if hasattr(scene.cycles, 'denoiser'):
        try:
            scene.cycles.denoiser = 'NONE'
        except Exception:
            pass
    for _view_layer in scene.view_layers:
        if hasattr(_view_layer, 'cycles') and hasattr(_view_layer.cycles, 'use_denoising'):
            _view_layer.cycles.use_denoising = False
    scene.cycles.device = '__CYCLES_DEVICE__'
    if scene.cycles.device == 'GPU':
        prefs = bpy.context.preferences.addons.get('cycles')
        if prefs is not None:
            cprefs = prefs.preferences
            for compute_type in ('OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI'):
                try:
                    cprefs.compute_device_type = compute_type
                    cprefs.get_devices()
                    enabled = False
                    for dev in cprefs.devices:
                        if getattr(dev, 'type', '') != 'CPU':
                            dev.use = True
                            enabled = True
                        else:
                            dev.use = False
                    if enabled:
                        print('[cg-tutor] cycles GPU', compute_type)
                        break
                except Exception:
                    continue
else:
    scene.eevee.taa_render_samples = 16
if scene.world:
    scene.world.color = (0.035, 0.04, 0.048)
scene.view_settings.view_transform = 'Filmic'
scene.view_settings.look = 'Medium High Contrast'
scene.view_settings.exposure = 0.15
scene.view_settings.gamma = 1.0
if hasattr(scene.eevee, 'use_gtao'):
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 3
    scene.eevee.gtao_factor = 0.7
if hasattr(scene.eevee, 'use_bloom'):
    scene.eevee.use_bloom = False
if hasattr(scene.eevee, 'use_ssr'):
    scene.eevee.use_ssr = False
if hasattr(scene.eevee, 'use_ssr_refraction'):
    scene.eevee.use_ssr_refraction = False
if hasattr(scene.eevee, 'use_motion_blur'):
    scene.eevee.use_motion_blur = False
scene.render.resolution_x = STORYBOARD['resolution'][0]
scene.render.resolution_y = STORYBOARD['resolution'][1]
scene.render.fps = STORYBOARD['fps']
scene.frame_start = 1
scene.frame_end = round(sum(s['duration_sec'] for s in STORYBOARD['shots']) * STORYBOARD['fps'])
out_dir = os.environ['CG_TUTOR_OUT_DIR']
Path(out_dir).mkdir(parents=True, exist_ok=True)
_render_path = os.path.join(out_dir, 'frame_####.png')
if not _render_path.startswith('\\\\'):
    _render_path = _render_path.replace('\\', '/')
scene.render.filepath = _render_path

objects = {}
visibility = {}


def add_visibility(name, start, end):
    visibility.setdefault(name, []).append((start, end))


cursor = 0
for idx, shot in enumerate(STORYBOARD['shots']):
    shot_frames = round(shot['duration_sec'] * STORYBOARD['fps'])
    start = cursor + 1
    end = cursor + shot_frames
    cam_data = bpy.data.cameras.new('camera_' + shot['node_id'])
    cam = bpy.data.objects.new('camera_' + shot['node_id'], cam_data)
    bpy.context.collection.objects.link(cam)
    first_key = shot['camera'][0]
    cam.location = tuple(first_key['position'])
    look_at(cam, first_key['look_at'])
    cam.data.sensor_width = 36.0
    cam.data.lens = lens_for_fov(first_key.get('fov', 50.0), cam.data.sensor_width)
    marker = scene.timeline_markers.new('shot_' + shot['node_id'], frame=start)
    marker.camera = cam
    if idx == 0:
        scene.camera = cam
    for key in shot['camera']:
        frame = frame_for_time(key['time_sec'])
        cam.location = tuple(key['position'])
        look_at(cam, key['look_at'])
        cam.data.lens = lens_for_fov(key.get('fov', first_key.get('fov', 50.0)), cam.data.sensor_width)
        cam.keyframe_insert('location', frame=frame)
        cam.keyframe_insert('rotation_euler', frame=frame)
        cam.data.keyframe_insert('lens', frame=frame)
    for spec in shot['objects']:
        obj = objects.get(spec['name'])
        if obj is None:
            obj = add_object(spec)
            objects[spec['name']] = obj
        apply_initial_props(obj, spec)
        apply_spec_keyframes(obj, spec)
        add_visibility(spec['name'], start, end)
    for helper_name in add_teaching_helpers_for_shot(shot, objects, start, end):
        add_visibility(helper_name, start, end)
    cursor = end

if not any(obj.type == 'LIGHT' for obj in bpy.data.objects):
    data = bpy.data.lights.new('key_light', type='AREA')
    data.energy = 1200
    obj = bpy.data.objects.new('key_light', data)
    bpy.context.collection.objects.link(obj)
    obj.location = (2.5, -4.0, 5.0)
    objects[obj.name] = obj
    visibility[obj.name] = [(scene.frame_start, scene.frame_end)]

for name, obj in objects.items():
    set_visibility_windows(obj, visibility.get(name, []))

preview_raw = os.environ.get('CG_TUTOR_PREVIEW_FRAMES', '').strip()
if preview_raw:
    for raw in preview_raw.split(','):
        if not raw.strip():
            continue
        frame = int(raw)
        scene.frame_set(frame)
        _preview_path = os.path.join(out_dir, f'frame_{frame:04d}.png')
        if not _preview_path.startswith('\\\\'):
            _preview_path = _preview_path.replace('\\', '/')
        scene.render.filepath = _preview_path
        bpy.ops.render.render(write_still=True)
else:
    bpy.ops.render.render(animation=True)
'''


def _runtime_storyboard_data(storyboard: Storyboard, visual_contracts=None) -> dict:
    """Return only fields the compiled bpy runtime actually consumes.

    The compiler does not render formulas/captions itself; leaving them in
    the embedded JSON can trip static checks that forbid formula text inside
    Blender scenes.
    """
    data = storyboard.model_dump(mode="json")
    _ensure_required_vector_placeholders(data, visual_contracts)
    data = _enrich_storyboard_motion(data)
    for shot in data.get("shots", []):
        shot.pop("formula", None)
        shot.pop("caption", None)
        shot.pop("overlay_zone", None)
    return data


def _ensure_required_vector_placeholders(data: dict, visual_contracts) -> None:
    if not visual_contracts:
        return
    shots = data.get("shots")
    if not isinstance(shots, list):
        return
    for shot_index, shot in enumerate(shots):
        if not isinstance(shot, dict):
            continue
        shot_id = str(shot.get("node_id", "") or "")
        contract = visual_contracts.get(shot_id) if hasattr(visual_contracts, "get") else None
        vectors = list(getattr(contract, "required_vectors", []) or [])
        if not vectors:
            continue
        objects = shot.get("objects")
        if not isinstance(objects, list):
            objects = []
            shot["objects"] = objects
        existing = {
            str(obj.get("name", "") or "").lower()
            for obj in objects
            if isinstance(obj, dict)
        }
        start = float(shot.get("start_sec", 0.0) or 0.0)
        duration = max(0.5, float(shot.get("duration_sec", 1.0) or 1.0))
        end = round(start + min(duration * 0.75, max(duration - 0.1, 0.25)), 6)
        for vector_index, raw_name in enumerate(vectors[:10]):
            token = _safe_vector_token(raw_name, shot_index, vector_index)
            token_low = token.lower()
            raw_low = str(raw_name).lower()
            if any(token_low in name or raw_low in name for name in existing):
                continue
            name = f"{token}_placeholder"
            base_x = -1.45 + 0.18 * (vector_index % 4)
            base_y = -0.35 + 0.18 * (vector_index // 4)
            base_z = 1.0 + 0.08 * (vector_index % 3)
            points = [
                [round(base_x, 4), round(base_y, 4), round(base_z, 4)],
                [round(base_x + 0.85, 4), round(base_y + 0.32, 4), round(base_z + 0.04, 4)],
            ]
            color = _placeholder_vector_color(vector_index, raw_name)
            objects.append({
                "name": name,
                "type": "mesh",
                "primitive": "curve_polyline",
                "location": points[0],
                "properties": {
                    "points": points,
                    "color": color,
                    "bevel_depth": 0.014,
                    "bevel_factor_end": 0.04,
                    "emission": 0.18,
                },
                "keyframes": [
                    {
                        "time_sec": round(start, 6),
                        "attr": "data.bevel_factor_end",
                        "value": 0.04,
                    },
                    {
                        "time_sec": end,
                        "attr": "data.bevel_factor_end",
                        "value": 1.0,
                    },
                ],
            })
            existing.add(name.lower())


def _safe_vector_token(raw_name, shot_index: int, vector_index: int) -> str:
    text = str(raw_name or "").strip()
    token = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")
    if not token:
        token = f"required_vector_{shot_index + 1}_{vector_index + 1}"
    if len(token) == 1:
        token = f"required_vector_{token}"
    if not re.match(r"^[A-Za-z_]", token):
        token = "required_vector_" + token
    return token[:64]


def _placeholder_vector_color(index: int, raw_name) -> list[float]:
    low = str(raw_name).lower()
    if "red" in low:
        return [1.0, 0.18, 0.12, 1.0]
    if "green" in low:
        return [0.2, 1.0, 0.32, 1.0]
    if "blue" in low:
        return [0.25, 0.45, 1.0, 1.0]
    if "normal" in low or low in {"n", "required_vector_n"}:
        return [1.0, 0.9, 0.18, 1.0]
    palette = (
        [0.35, 0.7, 1.0, 1.0],
        [1.0, 0.62, 0.18, 1.0],
        [0.9, 0.35, 1.0, 1.0],
        [0.4, 1.0, 0.72, 1.0],
    )
    return palette[index % len(palette)]


def _enrich_storyboard_motion(data: dict) -> dict:
    """Add minimal teaching motion to a runtime copy of the storyboard.

    The original storyboard artifact remains unchanged; this only protects
    compiled scaffolds and external/cached storyboards that lack keyframes.
    """
    out = deepcopy(data)
    concept_id = str(out.get("concept_id", "")).lower()
    for shot_idx, shot in enumerate(out.get("shots", [])):
        if not isinstance(shot, dict):
            continue
        start = float(shot.get("start_sec", 0.0) or 0.0)
        duration = max(0.0, float(shot.get("duration_sec", 0.0) or 0.0))
        is_dolly_zoom = _is_dolly_zoom_scene(concept_id, shot)
        if is_dolly_zoom:
            _normalise_dolly_zoom_layout(shot, shot_idx, start, duration)
        is_shadow_softness = _is_shadow_softness_scene(concept_id, shot)
        if is_shadow_softness:
            _normalise_shadow_softness_layout(shot, start, duration)
        _normalise_teaching_layout(shot)
        if not is_dolly_zoom:
            _ensure_camera_framing(shot)
        if duration <= 2.0:
            continue
        if not is_dolly_zoom:
            _ensure_camera_motion(shot, shot_idx, start, duration)
        if not is_shadow_softness:
            _ensure_object_motion(shot, shot_idx, start, duration)
    return out


def _is_dolly_zoom_scene(concept_id: str, shot: dict) -> bool:
    cid = concept_id.lower()
    if cid == "dolly_zoom" or cid.startswith("dolly_zoom_") or "dolly zoom" in cid:
        return True
    text = " ".join(
        str(shot.get(key, "") or "")
        for key in ("caption", "formula", "node_id")
    ).lower()
    object_names = " ".join(
        str(obj.get("name", "") or "")
        for obj in (shot.get("objects") or [])
        if isinstance(obj, dict)
    ).lower()
    return "dolly zoom" in text or (
        "hero_lighthouse" in object_names
        and "marker_post" in object_names
        and "camera_icon" in object_names
    )


def _is_shadow_softness_scene(concept_id: str, shot: dict) -> bool:
    cid = concept_id.lower()
    if cid == "shadow_softness_radius" or cid.startswith("shadow_softness_radius_"):
        return True
    text = " ".join(
        str(shot.get(key, "") or "")
        for key in ("caption", "formula", "node_id")
    ).lower()
    object_names = " ".join(
        str(obj.get("name", "") or "")
        for obj in (shot.get("objects") or [])
        if isinstance(obj, dict)
    ).lower()
    has_shadow_set = (
        "area_light" in object_names
        and ("ground_plane" in object_names or "floor_plane" in object_names)
        and ("subject_pillar" in object_names or "subject_sphere" in object_names)
    )
    return has_shadow_set and ("softness" in text or "shadow" in object_names)


def _normalise_shadow_softness_layout(
    shot: dict,
    start: float,
    duration: float,
) -> None:
    objects = shot.get("objects")
    if not isinstance(objects, list):
        objects = []
        shot["objects"] = objects

    area_light = None
    retained = []
    for obj in objects:
        if not isinstance(obj, dict):
            retained.append(obj)
            continue
        name = str(obj.get("name", "")).lower()
        kind = str(obj.get("type", "")).lower()
        if name == "area_light":
            if area_light is not None:
                continue
            area_light = obj
            retained.append(obj)
            continue
        if kind == "light" or "key_light" in name:
            if area_light is None:
                area_light = obj
                obj["name"] = "area_light"
                retained.append(obj)
            continue
        retained.append(obj)
    if area_light is None:
        area_light = {
            "name": "area_light",
            "type": "light",
            "primitive": None,
            "location": [-1.7, -2.6, 4.2],
            "properties": {},
            "keyframes": [],
        }
        retained.append(area_light)
    objects[:] = retained

    area_light["name"] = "area_light"
    area_light["type"] = "light"
    area_light["primitive"] = None
    subject_loc = _shadow_subject_location(objects)
    area_light["location"] = [
        round(subject_loc[0] - 1.7, 4),
        round(subject_loc[1] - 2.6, 4),
        round(max(subject_loc[2] + 3.2, 3.8), 4),
    ]
    props = _props(area_light)
    props["light_kind"] = "AREA"
    props["energy"] = max(float(props.get("energy", 1300.0)), 1300.0)
    props["size"] = 0.05
    props.setdefault("light_color", [1.0, 0.93, 0.82])
    end = round(start + max(duration, 0.25), 6)
    area_light["keyframes"] = [
        {"time_sec": round(start, 6), "attr": "data.size", "value": 0.05},
        {"time_sec": end, "attr": "data.size", "value": 1.25},
    ]

    for obj in objects:
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("name", "")).lower()
        if name in {"subject_pillar", "subject_sphere", "ground_plane", "floor_plane"}:
            obj["keyframes"] = []


def _shadow_subject_location(objects: list) -> list[float]:
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if str(obj.get("name", "")).lower() in {"subject_pillar", "subject_sphere"}:
            return _vec3(obj.get("location"), [0.0, 0.0, 0.8])
    return [0.0, 0.0, 0.8]


def _normalise_dolly_zoom_layout(
    shot: dict,
    shot_idx: int,
    start: float,
    duration: float,
) -> None:
    """Make the deterministic seed express the actual Vertigo geometry.

    Dolly zoom is highly sensitive to layout: the foreground subject must be
    centered, background markers must sit behind it along depth, and the render
    camera must animate both distance and FOV. This keeps the compiled scaffold
    useful even when the storyboard patch is underspecified.
    """
    objects = shot.get("objects")
    if not isinstance(objects, list):
        objects = []
        shot["objects"] = objects
    by_name = {
        str(obj.get("name", "")).lower(): obj
        for obj in objects
        if isinstance(obj, dict)
    }

    _ensure_dolly_anchor(objects, by_name, "hero_lighthouse", "cylinder", [0.0, 0.0, 1.0])
    _ensure_dolly_anchor(objects, by_name, "camera_icon", "cube", [0.0, -3.2, 0.55])
    _ensure_dolly_anchor(objects, by_name, "thin_camera_frustum", "curve_polyline", [0.0, -3.2, 0.55])
    _ensure_dolly_anchor(objects, by_name, "camera_frame_inset", "plane", [-2.9, -0.2, 2.0])
    for idx in range(4):
        _ensure_dolly_anchor(
            objects,
            by_name,
            f"marker_post_{idx + 1}",
            "cylinder",
            [-1.8 + idx * 1.2, 3.2 + idx * 1.15, 0.75],
        )

    for name, obj in by_name.items():
        if "hero_lighthouse" in name:
            obj["primitive"] = "cylinder"
            obj["location"] = [0.0, 0.0, 1.0]
            props = _props(obj)
            props.setdefault("radius", 0.36)
            props.setdefault("depth", 2.0)
            props.setdefault("color", [0.92, 0.94, 0.96])
            obj["keyframes"] = []
        elif name.startswith("marker_post_"):
            idx = _trailing_index(name) - 1
            idx = max(0, min(3, idx))
            obj["primitive"] = "cylinder"
            obj["location"] = [-1.8 + idx * 1.2, 3.2 + idx * 1.15, 0.75]
            props = _props(obj)
            props.setdefault("radius", 0.11)
            props.setdefault("depth", 1.5)
            props.setdefault("color", [[0.95, 0.45, 0.55], [0.95, 0.75, 0.35], [0.55, 0.85, 0.7], [0.45, 0.68, 0.95]][idx])
            obj["keyframes"] = []
        elif "camera_icon" in name:
            obj["primitive"] = "cube"
            props = _props(obj)
            props.setdefault("size", 0.42)
            props.setdefault("color", [0.12, 0.13, 0.16])
            _set_dolly_object_motion(obj, start, duration, shot_idx)
        elif "thin_camera_frustum" in name:
            obj["primitive"] = "curve_polyline"
            obj["location"] = [0.0, 0.0, 0.0]
            props = _props(obj)
            props["points"] = [[0.0, -3.2, 0.55], [0.0, 0.0, 1.0], [-1.35, 3.4, 1.1]]
            props.setdefault("bevel_depth", 0.014)
            props.setdefault("color", [0.88, 0.94, 1.0])
            props.setdefault("emission", 0.18)
            obj["keyframes"] = [
                {"time_sec": round(start, 6), "attr": "scale", "value": [1.0, 1.0, 1.0]},
                {"time_sec": round(start + duration, 6), "attr": "scale", "value": _dolly_frustum_scale(shot_idx)},
            ]
        elif "camera_frame_inset" in name:
            obj["primitive"] = "plane"
            obj["location"] = [-2.9, -0.2, 2.0]
            props = _props(obj)
            props.setdefault("size", 1.05)
            props.setdefault("color", [0.05, 0.07, 0.09])
            props.setdefault("alpha", 0.35)

    _set_dolly_camera_keys(shot, shot_idx, start, duration)


def _ensure_dolly_anchor(
    objects: list,
    by_name: dict[str, dict],
    name: str,
    primitive: str,
    location: list[float],
) -> dict:
    existing = _find_spec(by_name, (name,))
    if existing is not None:
        return existing
    spec = {
        "name": name,
        "type": "primitive",
        "primitive": primitive,
        "location": location,
        "properties": {},
        "keyframes": [],
    }
    objects.append(spec)
    by_name[name.lower()] = spec
    return spec


def _trailing_index(name: str) -> int:
    digits = ""
    for ch in reversed(name):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            break
    return int(digits or "1")


def _dolly_mode(shot_idx: int, shot: dict | None = None) -> str:
    text = ""
    if shot is not None:
        text = " ".join(
            str(shot.get(key, "") or "")
            for key in ("caption", "formula", "node_id")
        ).lower()
    if "zoom only" in text:
        return "zoom_only"
    if "dolly only" in text:
        return "dolly_only"
    if "loop" in text or shot_idx >= 4:
        return "loop"
    if "dolly zoom" in text or shot_idx == 3:
        return "dolly_zoom"
    if shot_idx == 1:
        return "dolly_only"
    if shot_idx == 2:
        return "zoom_only"
    return "setup"


def _set_dolly_camera_keys(
    shot: dict,
    shot_idx: int,
    start: float,
    duration: float,
) -> None:
    mode = _dolly_mode(shot_idx, shot)
    z = 2.25
    look = [0.0, 0.0, 1.05]
    if mode == "dolly_only":
        keys = [(-5.0, 48.0), (-9.0, 48.0)]
    elif mode == "zoom_only":
        keys = [(-7.0, 62.0), (-7.0, 27.0)]
    elif mode == "dolly_zoom":
        keys = [(-4.8, 68.0), (-10.2, 27.0)]
    elif mode == "loop":
        mid = round(start + duration * 0.5, 6)
        shot["camera"] = [
            {"time_sec": round(start, 6), "position": [0.0, -5.0, z], "look_at": look, "fov": 66.0},
            {"time_sec": mid, "position": [0.0, -10.0, z], "look_at": look, "fov": 28.0},
            {"time_sec": round(start + duration, 6), "position": [0.0, -5.0, z], "look_at": look, "fov": 66.0},
        ]
        return
    else:
        keys = [(-6.5, 48.0), (-6.5, 48.0)]
    shot["camera"] = [
        {"time_sec": round(start, 6), "position": [0.0, keys[0][0], z], "look_at": look, "fov": keys[0][1]},
        {"time_sec": round(start + duration, 6), "position": [0.0, keys[1][0], z], "look_at": look, "fov": keys[1][1]},
    ]


def _set_dolly_object_motion(
    obj: dict,
    start: float,
    duration: float,
    shot_idx: int,
) -> None:
    mode = _dolly_mode(shot_idx)
    if mode == "dolly_only":
        locs = [[0.0, -3.0, 0.55], [0.0, -4.8, 0.55]]
    elif mode == "zoom_only":
        locs = [[0.0, -3.6, 0.55], [0.0, -3.6, 0.55]]
    elif mode == "dolly_zoom":
        locs = [[0.0, -2.8, 0.55], [0.0, -5.2, 0.55]]
    elif mode == "loop":
        locs = [[0.0, -3.0, 0.55], [0.0, -5.0, 0.55], [0.0, -3.0, 0.55]]
        obj["keyframes"] = [
            {"time_sec": round(start, 6), "attr": "location", "value": locs[0]},
            {"time_sec": round(start + duration * 0.5, 6), "attr": "location", "value": locs[1]},
            {"time_sec": round(start + duration, 6), "attr": "location", "value": locs[2]},
        ]
        return
    else:
        locs = [[0.0, -3.2, 0.55], [0.0, -3.2, 0.55]]
    obj["keyframes"] = [
        {"time_sec": round(start, 6), "attr": "location", "value": locs[0]},
        {"time_sec": round(start + duration, 6), "attr": "location", "value": locs[1]},
    ]


def _dolly_frustum_scale(shot_idx: int) -> list[float]:
    mode = _dolly_mode(shot_idx)
    if mode == "zoom_only":
        return [0.48, 1.0, 0.48]
    if mode in {"dolly_zoom", "loop"}:
        return [0.42, 1.18, 0.42]
    return [1.0, 1.0, 1.0]


def _normalise_teaching_layout(shot: dict) -> None:
    objects = shot.get("objects")
    if not isinstance(objects, list):
        return
    by_name = {
        str(obj.get("name", "")).lower(): obj
        for obj in objects
        if isinstance(obj, dict)
    }
    pinhole = _find_spec(by_name, ("pinhole", "camera_center"))
    image_plane = _find_spec(by_name, ("image_plane",))
    camera_box = _find_spec(by_name, ("camera_box",))
    object_points = [
        obj for name, obj in by_name.items()
        if "object_point" in name
    ]
    projected_points = [
        obj for name, obj in by_name.items()
        if "projected_point" in name
    ]
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if str(obj.get("type", "")).lower() == "light":
            props = _props(obj)
            props["energy"] = max(float(props.get("energy", 900.0)), 1100.0)
            props.setdefault("size", 4.5)
    if camera_box is not None:
        props = _props(camera_box)
        props.setdefault("color", [0.035, 0.035, 0.04])
        props["color"] = [0.035, 0.035, 0.04]
        props["size"] = min(float(props.get("size", 1.0)), 1.1)
        if pinhole is not None:
            _move_spec_location(camera_box, _vec3(
                pinhole.get("location"),
                _vec3(camera_box.get("location"), [0.0, 0.0, 1.0]),
            ))
    if pinhole is not None:
        props = _props(pinhole)
        props.setdefault("radius", 0.075)
        props.setdefault("color", [1.0, 0.88, 0.28])
    if image_plane is not None:
        props = _props(image_plane)
        props.setdefault("color", [0.42, 0.74, 1.0])
        props["alpha"] = min(float(props.get("alpha", 0.32)), 0.42)
        props.setdefault("size", 3.2)
        props.setdefault("rotation_euler", [0.0, 1.5707963268, 0.0])
    for idx, obj in enumerate(object_points):
        props = _props(obj)
        obj["primitive"] = "sphere"
        props["radius"] = min(float(props.get("radius", 0.09)), 0.1)
        props.setdefault("color", [1.0, 0.82, 0.2])
        loc = _vec3(obj.get("location"), [1.6, 0.0, 1.0])
        _move_spec_location(
            obj, _spread_point_location(loc, idx, len(object_points)),
        )
    for idx, obj in enumerate(projected_points):
        props = _props(obj)
        obj["primitive"] = "sphere"
        props["radius"] = min(float(props.get("radius", 0.08)), 0.09)
        props.setdefault("color", [0.9, 0.9, 1.0])
        if image_plane is not None:
            loc = _vec3(obj.get("location"), [-2.0, 0.0, 1.0])
            plane_loc = _vec3(image_plane.get("location"), loc)
            loc[0] = plane_loc[0]
            _move_spec_location(
                obj, _spread_point_location(loc, idx, len(projected_points)),
            )
    _ensure_projected_points_for_object_points(
        objects, object_points, projected_points, image_plane,
    )


def _find_spec(by_name: dict[str, dict], needles: tuple[str, ...]) -> dict | None:
    for needle in needles:
        for name, obj in by_name.items():
            if needle in name:
                return obj
    return None


def _props(obj: dict) -> dict:
    props = obj.get("properties")
    if not isinstance(props, dict):
        props = {}
        obj["properties"] = props
    return props


def _move_spec_location(obj: dict, new_loc: list[float]) -> None:
    old_loc = _vec3(obj.get("location"), new_loc)
    obj["location"] = [round(float(v), 4) for v in new_loc]
    delta = [new_loc[i] - old_loc[i] for i in range(3)]
    if max(abs(v) for v in delta) < 1e-9:
        return
    for key in obj.get("keyframes", []) or []:
        if (
            isinstance(key, dict)
            and key.get("attr") == "location"
            and isinstance(key.get("value"), list)
            and len(key["value"]) == 3
        ):
            key["value"] = [
                round(float(key["value"][i]) + delta[i], 4)
                for i in range(3)
            ]


def _spread_point_location(loc: list[float], idx: int, total: int) -> list[float]:
    if total <= 1:
        return loc
    offsets = [
        (0.0, -0.26, 0.22),
        (0.0, 0.28, -0.16),
        (0.0, 0.02, 0.36),
        (0.0, -0.42, -0.08),
    ]
    ox, oy, oz = offsets[idx % len(offsets)]
    return [
        round(loc[0] + ox, 4),
        round(loc[1] + oy, 4),
        round(loc[2] + oz, 4),
    ]


def _suffix_from_name(name: str, fallback: int) -> str:
    low = name.lower()
    for suffix in ("a", "b", "c", "d"):
        if low.endswith("_" + suffix) or f"_{suffix}_" in low:
            return suffix
    return chr(ord("a") + min(fallback, 25))


def _ensure_projected_points_for_object_points(
    objects: list,
    object_points: list[dict],
    projected_points: list[dict],
    image_plane: dict | None,
) -> None:
    if image_plane is None:
        return
    existing = {
        _suffix_from_name(str(obj.get("name", "")), idx)
        for idx, obj in enumerate(projected_points)
    }
    plane_loc = _vec3(image_plane.get("location"), [-2.0, 0.0, 1.0])
    for idx, obj in enumerate(object_points):
        suffix = _suffix_from_name(str(obj.get("name", "")), idx)
        if suffix in existing:
            continue
        loc = _vec3(obj.get("location"), [2.0, 0.0, 1.0])
        projected = {
            "name": f"projected_point_{suffix}",
            "type": "primitive",
            "primitive": "sphere",
            "location": [
                plane_loc[0],
                round(plane_loc[1] + (loc[1] * 0.55), 4),
                round(plane_loc[2] + ((loc[2] - plane_loc[2]) * 0.55), 4),
            ],
            "properties": {
                "radius": 0.08,
                "color": [0.9, 0.9, 1.0],
            },
            "keyframes": [],
        }
        objects.append(projected)
        projected_points.append(projected)
        existing.add(suffix)


def _ensure_camera_framing(shot: dict) -> None:
    camera = shot.get("camera")
    objects = shot.get("objects")
    if not isinstance(camera, list) or not camera or not isinstance(objects, list):
        return
    spread = _object_spread(objects)
    teaching_scene = _has_teaching_geometry(objects)
    if not teaching_scene and spread < 3.0:
        return
    for key in camera:
        if not isinstance(key, dict):
            continue
        key["fov"] = max(float(key.get("fov", 50.0)), 62.0)
        pos = _vec3(key.get("position"), [0.0, -7.0, 3.2])
        if abs(pos[1]) > 0.1:
            pos[1] = round(pos[1] * 1.18, 4)
        else:
            pos[1] = -7.5
        pos[2] = round(pos[2] + 0.12, 4)
        key["position"] = pos


def _object_spread(objects: list) -> float:
    coords = [
        _vec3(obj.get("location"), [0.0, 0.0, 0.0])
        for obj in objects
        if isinstance(obj, dict)
    ]
    if not coords:
        return 0.0
    return max(max(vals) - min(vals) for vals in zip(*coords))


def _has_teaching_geometry(objects: list) -> bool:
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("name", "")).lower()
        primitive = str(obj.get("primitive", "")).lower()
        if primitive == "curve_polyline" or any(
            word in name
            for word in (
                "ray", "vector", "projected", "object_point",
                "image_plane", "pinhole", "camera_center",
            )
        ):
            return True
    return False


def _ensure_camera_motion(
    shot: dict,
    shot_idx: int,
    start: float,
    duration: float,
) -> None:
    camera = shot.get("camera")
    if not isinstance(camera, list) or not camera:
        return
    if _camera_motion_delta(camera) >= 0.35:
        return
    first = dict(camera[0])
    first["time_sec"] = round(start, 6)
    last = dict(camera[-1])
    pos = _vec3(last.get("position"), [0.0, -7.0, 3.2])
    look = _vec3(last.get("look_at"), [0.0, 0.0, 0.9])
    sign = -1.0 if shot_idx % 2 else 1.0
    last["time_sec"] = round(start + duration, 6)
    last["position"] = [
        round(pos[0] + sign * 0.6, 4),
        round(pos[1] + 0.1, 4),
        round(pos[2] + 0.08, 4),
    ]
    last["look_at"] = [
        round(look[0] + sign * 0.08, 4),
        round(look[1], 4),
        round(look[2], 4),
    ]
    last["fov"] = float(last.get("fov", first.get("fov", 50.0)))
    camera[:] = [first, last]


def _ensure_object_motion(
    shot: dict,
    shot_idx: int,
    start: float,
    duration: float,
) -> None:
    objects = shot.get("objects")
    if not isinstance(objects, list) or _has_object_motion(objects):
        return
    added = 0
    for obj in sorted(objects, key=_motion_candidate_key):
        if not isinstance(obj, dict) or _is_static_anchor(obj):
            continue
        if _add_motion_keyframes(obj, shot_idx, start, duration):
            added += 1
        if added >= 2:
            break


def _has_object_motion(objects: list) -> bool:
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        times = {
            round(float(key.get("time_sec", 0.0)), 4)
            for key in obj.get("keyframes", [])
            if isinstance(key, dict)
            and key.get("attr") not in {"hide_render", "hide_viewport"}
        }
        if len(times) >= 2:
            return True
    return False


def _add_motion_keyframes(
    obj: dict,
    shot_idx: int,
    start: float,
    duration: float,
) -> bool:
    if obj.get("keyframes"):
        return False
    end = round(start + max(duration * 0.8, 0.25), 6)
    name = str(obj.get("name", "")).lower()
    primitive = str(obj.get("primitive", "")).lower()
    obj_type = str(obj.get("type", "")).lower()
    loc = _vec3(obj.get("location"), [0.0, 0.0, 0.0])
    sign = -1.0 if shot_idx % 2 else 1.0
    if primitive == "curve_polyline":
        obj["keyframes"] = [
            {"time_sec": round(start, 6), "attr": "data.bevel_factor_end", "value": 0.03},
            {"time_sec": end, "attr": "data.bevel_factor_end", "value": 1.0},
        ]
    elif any(w in name for w in ("ray", "vector", "arrow")):
        obj["keyframes"] = [
            {"time_sec": round(start, 6), "attr": "scale", "value": [0.72, 0.72, 0.72]},
            {"time_sec": end, "attr": "scale", "value": [1.08, 1.08, 1.08]},
        ]
    elif obj_type == "light":
        obj["keyframes"] = [
            {"time_sec": round(start, 6), "attr": "location", "value": loc},
            {
                "time_sec": end,
                "attr": "location",
                "value": [
                    round(loc[0] + sign * 0.45, 4),
                    round(loc[1] + 0.2, 4),
                    round(loc[2], 4),
                ],
            },
        ]
    else:
        obj["keyframes"] = [
            {"time_sec": round(start, 6), "attr": "location", "value": loc},
            {
                "time_sec": end,
                "attr": "location",
                "value": [
                    round(loc[0] + sign * 0.28, 4),
                    round(loc[1] + 0.18, 4),
                    round(loc[2] + 0.06, 4),
                ],
            },
        ]
    return True


def _motion_candidate_key(obj: dict) -> tuple[int, str]:
    name = str(obj.get("name", "")).lower()
    primitive = str(obj.get("primitive", "")).lower()
    obj_type = str(obj.get("type", "")).lower()
    if any(w in name for w in ("ray", "vector", "arrow", "projected")):
        return (0, name)
    if any(w in name for w in ("point", "object", "marker", "sample")):
        return (10, name)
    if primitive in {"arrow", "curve_polyline", "sphere", "cube", "cone"}:
        return (20, name)
    if obj_type == "light":
        return (40, name)
    return (80, name)


def _is_static_anchor(obj: dict) -> bool:
    name = str(obj.get("name", "")).lower()
    if "marker_post" in name or "background_marker" in name:
        return True
    if any(w in name for w in ("ray", "vector", "arrow", "projected", "point", "object")):
        return False
    return any(
        w in name
        for w in (
            "floor", "grid", "bench", "wall", "window", "frame", "screen",
            "plane", "aperture", "camera_center", "axis", "axes",
        )
    )


def _camera_motion_delta(camera: list) -> float:
    keys = [key for key in camera if isinstance(key, dict)]
    if len(keys) < 2:
        return 0.0
    return max(
        _vec_delta(keys[0].get("position"), keys[-1].get("position")),
        _vec_delta(keys[0].get("look_at"), keys[-1].get("look_at")),
    )


def _vec_delta(a, b) -> float:
    av = _vec3(a, [0.0, 0.0, 0.0])
    bv = _vec3(b, [0.0, 0.0, 0.0])
    return sum((x - y) ** 2 for x, y in zip(av, bv)) ** 0.5


def _vec3(value, default: list[float]) -> list[float]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            pass
    return [float(default[0]), float(default[1]), float(default[2])]


def _name_has_readable_label(name: str) -> bool:
    low = name.lower()
    return any(
        marker in low
        for marker in (
            "pinhole",
            "camera_center",
            "image_plane",
            "projected_point",
            "object_point",
        )
    )


def _shot_has_name(names: set[str], candidates: tuple[str, ...]) -> bool:
    lows = {name.lower() for name in names}
    for candidate in candidates:
        wanted = candidate.lower()
        if any(wanted in name or name in wanted for name in lows):
            return True
    return False


def _compiled_scene_hints(data: dict) -> dict[str, int]:
    """Static hints for validators that cannot evaluate runtime loops."""
    text_objects = 0
    tracer_primitives = 0
    for shot in data.get("shots", []):
        specs = shot.get("objects") or []
        names = {str(spec.get("name", "")) for spec in specs}
        text_objects += sum(
            1
            for spec in specs
            if spec.get("type") in {"annotation", "text"}
            or spec.get("primitive") == "empty"
        )
        text_objects += sum(1 for name in names if _name_has_readable_label(name))
        tracer_primitives += sum(
            1
            for spec in specs
            if spec.get("primitive") == "curve_polyline"
            or "ray" in str(spec.get("name", "")).lower()
            or spec.get("primitive") == "arrow"
        )
        has_pinhole = _shot_has_name(names, ("pinhole_aperture_C", "pinhole", "camera_center_marker"))
        has_image_plane = _shot_has_name(names, ("translucent_image_plane", "image_plane"))
        has_projected_point = _shot_has_name(names, ("projected_point_a", "projected_point"))
        object_points = [
            name for name in names
            if "object_point" in name.lower()
        ]
        if has_pinhole and object_points:
            tracer_primitives += min(3, len(object_points))
        if has_pinhole and has_image_plane:
            tracer_primitives += 3
            text_objects += 1
        if has_image_plane and has_projected_point:
            tracer_primitives += 2
    return {
        "text_objects_created": text_objects,
        "arrow_or_tracer_primitives": tracer_primitives,
    }


def compile_storyboard_to_bpy(
    storyboard: Storyboard,
    *,
    render_engine: str = "BLENDER_EEVEE",
    cycles_device: str = "AUTO",
    visual_contracts=None,
) -> str:
    """Return a conservative runnable Blender script for common visual grammar."""
    if render_engine not in {"BLENDER_EEVEE", "CYCLES"}:
        raise ValueError("render_engine must be one of: BLENDER_EEVEE, CYCLES")
    if cycles_device not in {"AUTO", "GPU", "CPU"}:
        raise ValueError("cycles_device must be one of: AUTO, GPU, CPU")
    resolved_cycles_device = "GPU" if cycles_device == "AUTO" else cycles_device
    data = _runtime_storyboard_data(storyboard, visual_contracts=visual_contracts)
    scene_literal = pformat(data, width=88, sort_dicts=False)
    hints_literal = pformat(_compiled_scene_hints(data), width=88, sort_dicts=False)
    return (
        _RUNTIME_TEMPLATE
        .replace("__STORYBOARD_JSON__", scene_literal)
        .replace("__COMPILED_SCENE_HINTS_JSON__", hints_literal)
        .replace("__CYCLES_DEVICE__", resolved_cycles_device)
        .replace("scene.render.engine = 'BLENDER_EEVEE'",
                 f"scene.render.engine = '{render_engine}'")
    )


def format_compiled_scene_for_coder(code: str) -> str:
    return (
        "DETERMINISTIC SCENE SCAFFOLD:\n"
        "The script below was generated from the storyboard/Scene IR using "
        "the semi-deterministic compiler. Prefer preserving its object names, "
        "visibility windows, camera timing, preview-frame support "
        "(CG_TUTOR_PREVIEW_FRAMES), render boilerplate, and reusable teaching "
        "helpers such as labeled points, thin curve_polyline tracers, focal "
        "brackets, and image-plane marks. Improve it only where the storyboard/"
        "visual contract needs richer teaching visuals.\n\n"
        "```python\n"
        f"{code}\n"
        "```"
    )
