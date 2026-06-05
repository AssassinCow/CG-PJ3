import math
import os
from pathlib import Path

import bpy
from mathutils import Vector


STORYBOARD = {'concept_id': 'shape_morphing',
 'fps': 24,
 'resolution': [960, 540],
 'shots': [{'node_id': 'node_01',
            'start_sec': 0.0,
            'duration_sec': 3.5,
            'camera': [{'time_sec': 0.0,
                        'position': [0.0, -8.26, 3.32],
                        'look_at': [0.0, 0.0, 0.9],
                        'fov': 62.0},
                       {'time_sec': 3.5,
                        'position': [0.8, -8.1184, 3.38],
                        'look_at': [0.08, 0.0, 0.9],
                        'fov': 62.0}],
            'objects': [{'name': 'key_light',
                         'type': 'light',
                         'primitive': None,
                         'location': [0.0, -3.5, 5.0],
                         'properties': {'light_kind': 'AREA',
                                        'energy': 1100.0,
                                        'size': 5.0,
                                        'light_color': [1.0, 0.96, 0.9]},
                         'keyframes': []},
                        {'name': 'hero_mesh',
                         'type': 'primitive',
                         'primitive': 'cube',
                         'location': [2.0, 0.0, 1.0],
                         'properties': {'size': 0.8, 'color': [0.95, 0.35, 0.25]},
                         'keyframes': [{'time_sec': 0.0,
                                        'attr': 'location',
                                        'value': [2.0, 0.0, 1.0]},
                                       {'time_sec': 2.87,
                                        'attr': 'location',
                                        'value': [2.28, 0.18, 1.06]}]},
                        {'name': 'source_shape_A',
                         'type': 'primitive',
                         'primitive': 'cube',
                         'location': [2.0, 0.0, 1.0],
                         'properties': {'size': 0.8, 'color': [0.95, 0.35, 0.25]},
                         'keyframes': []},
                        {'name': 'target_shape_B',
                         'type': 'primitive',
                         'primitive': 'cube',
                         'location': [2.0, 0.0, 1.0],
                         'properties': {'size': 0.8, 'color': [0.95, 0.35, 0.25]},
                         'keyframes': []},
                        {'name': 'sample_vertex_marker',
                         'type': 'primitive',
                         'primitive': 'sphere',
                         'location': [0.15000000000000013, -0.35, 1.24],
                         'properties': {'radius': 0.08, 'color': [0.56, 0.65, 0.7]},
                         'keyframes': [{'time_sec': 0.0,
                                        'attr': 'location',
                                        'value': [0.15000000000000013, -0.35, 1.24]},
                                       {'time_sec': 2.87,
                                        'attr': 'location',
                                        'value': [0.43, -0.17, 1.3]}]},
                        {'name': 'trajectory_trace',
                         'type': 'mesh',
                         'primitive': 'curve_polyline',
                         'location': [0.0, 0.0, 0.0],
                         'properties': {'points': [[2.0, 0.25, 1.35],
                                                   [0.0, 0.0, 1.0],
                                                   [-2.0, -0.25, 0.95]],
                                        'bevel_depth': 0.012,
                                        'color': [0.2, 0.75, 1.0],
                                        'emission': 0.6},
                         'keyframes': []},
                        {'name': 'parameter_panel',
                         'type': 'primitive',
                         'primitive': 'plane',
                         'location': [2.4, 0.05, 1.45],
                         'properties': {'size': 1.2,
                                        'color': [0.08, 0.11, 0.14],
                                        'alpha': 0.35},
                         'keyframes': []},
                        {'name': 'visible_ray_vector_cues_placeholder',
                         'type': 'mesh',
                         'primitive': 'curve_polyline',
                         'location': [-1.45, -0.35, 1.0],
                         'properties': {'points': [[-1.45, -0.35, 1.0],
                                                   [-0.6, -0.03, 1.04]],
                                        'color': [0.35, 0.7, 1.0, 1.0],
                                        'bevel_depth': 0.014,
                                        'bevel_factor_end': 0.04,
                                        'emission': 0.18},
                         'keyframes': [{'time_sec': 0.0,
                                        'attr': 'data.bevel_factor_end',
                                        'value': 0.04},
                                       {'time_sec': 2.625,
                                        'attr': 'data.bevel_factor_end',
                                        'value': 1.0}]}]},
           {'node_id': 'node_02',
            'start_sec': 3.5,
            'duration_sec': 5.0,
            'camera': [{'time_sec': 3.5,
                        'position': [0.0, -8.26, 3.32],
                        'look_at': [0.0, 0.0, 0.9],
                        'fov': 62.0},
                       {'time_sec': 8.5,
                        'position': [-0.9, -8.0184, 3.46],
                        'look_at': [-0.16, 0.0, 0.9],
                        'fov': 62.0}],
            'objects': [{'name': 'key_light',
                         'type': 'light',
                         'primitive': None,
                         'location': [0.0, -3.5, 5.0],
                         'properties': {'light_kind': 'AREA',
                                        'energy': 1100.0,
                                        'size': 5.0,
                                        'light_color': [1.0, 0.96, 0.9]},
                         'keyframes': []},
                        {'name': 'hero_mesh',
                         'type': 'primitive',
                         'primitive': 'cube',
                         'location': [2.15, 0.0, 1.0],
                         'properties': {'size': 0.8, 'color': [0.95, 0.35, 0.25]},
                         'keyframes': [{'time_sec': 3.5,
                                        'attr': 'location',
                                        'value': [2.15, 0.0, 1.0]},
                                       {'time_sec': 7.6,
                                        'attr': 'location',
                                        'value': [1.87, 0.18, 1.06]}]},
                        {'name': 'source_shape_A',
                         'type': 'primitive',
                         'primitive': 'cube',
                         'location': [2.15, 0.0, 1.0],
                         'properties': {'size': 0.8, 'color': [0.95, 0.35, 0.25]},
                         'keyframes': []},
                        {'name': 'target_shape_B',
                         'type': 'primitive',
                         'primitive': 'cube',
                         'location': [2.15, 0.0, 1.0],
                         'properties': {'size': 0.8, 'color': [0.95, 0.35, 0.25]},
                         'keyframes': []},
                        {'name': 'sample_vertex_marker',
                         'type': 'primitive',
                         'primitive': 'sphere',
                         'location': [0.15000000000000013, -0.35, 1.24],
                         'properties': {'radius': 0.08, 'color': [0.56, 0.65, 0.7]},
                         'keyframes': [{'time_sec': 3.5,
                                        'attr': 'location',
                                        'value': [0.15000000000000013, -0.35, 1.24]},
                                       {'time_sec': 7.6,
                                        'attr': 'location',
                                        'value': [-0.13, -0.17, 1.3]}]},
                        {'name': 'trajectory_trace',
                         'type': 'mesh',
                         'primitive': 'curve_polyline',
                         'location': [0.0, 0.0, 0.0],
                         'properties': {'points': [[2.0, 0.25, 1.35],
                                                   [0.0, 0.0, 1.0],
                                                   [-2.0, -0.25, 0.95]],
                                        'bevel_depth': 0.012,
                                        'color': [0.2, 0.75, 1.0],
                                        'emission': 0.6},
                         'keyframes': []},
                        {'name': 'parameter_panel',
                         'type': 'primitive',
                         'primitive': 'plane',
                         'location': [2.4, 0.05, 1.45],
                         'properties': {'size': 1.2,
                                        'color': [0.08, 0.11, 0.14],
                                        'alpha': 0.35},
                         'keyframes': []}]},
           {'node_id': 'node_03',
            'start_sec': 8.5,
            'duration_sec': 4.0,
            'camera': [{'time_sec': 8.5,
                        'position': [0.0, -8.26, 3.32],
                        'look_at': [0.0, 0.0, 0.9],
                        'fov': 62.0},
                       {'time_sec': 12.5,
                        'position': [0.8, -8.1184, 3.38],
                        'look_at': [0.08, 0.0, 0.9],
                        'fov': 62.0}],
            'objects': [{'name': 'key_light',
                         'type': 'light',
                         'primitive': None,
                         'location': [0.0, -3.5, 5.0],
                         'properties': {'light_kind': 'AREA',
                                        'energy': 1100.0,
                                        'size': 5.0,
                                        'light_color': [1.0, 0.96, 0.9]},
                         'keyframes': []},
                        {'name': 'hero_mesh',
                         'type': 'primitive',
                         'primitive': 'cube',
                         'location': [2.3, 0.0, 1.0],
                         'properties': {'size': 0.8, 'color': [0.95, 0.35, 0.25]},
                         'keyframes': [{'time_sec': 8.5,
                                        'attr': 'location',
                                        'value': [2.3, 0.0, 1.0]},
                                       {'time_sec': 11.78,
                                        'attr': 'location',
                                        'value': [2.58, 0.18, 1.06]}]},
                        {'name': 'source_shape_A',
                         'type': 'primitive',
                         'primitive': 'cube',
                         'location': [2.3, 0.0, 1.0],
                         'properties': {'size': 0.8, 'color': [0.95, 0.35, 0.25]},
                         'keyframes': []},
                        {'name': 'target_shape_B',
                         'type': 'primitive',
                         'primitive': 'cube',
                         'location': [2.3, 0.0, 1.0],
                         'properties': {'size': 0.8, 'color': [0.95, 0.35, 0.25]},
                         'keyframes': []},
                        {'name': 'sample_vertex_marker',
                         'type': 'primitive',
                         'primitive': 'sphere',
                         'location': [0.15000000000000013, -0.35, 1.24],
                         'properties': {'radius': 0.08, 'color': [0.56, 0.65, 0.7]},
                         'keyframes': [{'time_sec': 8.5,
                                        'attr': 'location',
                                        'value': [0.15000000000000013, -0.35, 1.24]},
                                       {'time_sec': 11.78,
                                        'attr': 'location',
                                        'value': [0.43, -0.17, 1.3]}]},
                        {'name': 'trajectory_trace',
                         'type': 'mesh',
                         'primitive': 'curve_polyline',
                         'location': [0.0, 0.0, 0.0],
                         'properties': {'points': [[2.0, 0.25, 1.35],
                                                   [0.0, 0.0, 1.0],
                                                   [-2.0, -0.25, 0.95]],
                                        'bevel_depth': 0.012,
                                        'color': [0.2, 0.75, 1.0],
                                        'emission': 0.6},
                         'keyframes': []},
                        {'name': 'parameter_panel',
                         'type': 'primitive',
                         'primitive': 'plane',
                         'location': [2.4, 0.05, 1.45],
                         'properties': {'size': 1.2,
                                        'color': [0.08, 0.11, 0.14],
                                        'alpha': 0.35},
                         'keyframes': []}]},
           {'node_id': 'node_04',
            'start_sec': 12.5,
            'duration_sec': 5.5,
            'camera': [{'time_sec': 12.5,
                        'position': [0.0, -8.26, 3.32],
                        'look_at': [0.0, 0.0, 0.9],
                        'fov': 62.0},
                       {'time_sec': 18.0,
                        'position': [-0.9, -8.0184, 3.46],
                        'look_at': [-0.16, 0.0, 0.9],
                        'fov': 62.0}],
            'objects': [{'name': 'key_light',
                         'type': 'light',
                         'primitive': None,
                         'location': [0.0, -3.5, 5.0],
                         'properties': {'light_kind': 'AREA',
                                        'energy': 1100.0,
                                        'size': 5.0,
                                        'light_color': [1.0, 0.96, 0.9]},
                         'keyframes': []},
                        {'name': 'hero_mesh',
                         'type': 'primitive',
                         'primitive': 'cube',
                         'location': [2.45, 0.0, 1.0],
                         'properties': {'size': 0.8, 'color': [0.95, 0.35, 0.25]},
                         'keyframes': [{'time_sec': 12.5,
                                        'attr': 'location',
                                        'value': [2.45, 0.0, 1.0]},
                                       {'time_sec': 17.01,
                                        'attr': 'location',
                                        'value': [2.17, 0.18, 1.06]}]},
                        {'name': 'source_shape_A',
                         'type': 'primitive',
                         'primitive': 'cube',
                         'location': [2.45, 0.0, 1.0],
                         'properties': {'size': 0.8, 'color': [0.95, 0.35, 0.25]},
                         'keyframes': []},
                        {'name': 'target_shape_B',
                         'type': 'primitive',
                         'primitive': 'cube',
                         'location': [2.45, 0.0, 1.0],
                         'properties': {'size': 0.8, 'color': [0.95, 0.35, 0.25]},
                         'keyframes': []},
                        {'name': 'sample_vertex_marker',
                         'type': 'primitive',
                         'primitive': 'sphere',
                         'location': [0.15000000000000013, -0.35, 1.24],
                         'properties': {'radius': 0.08, 'color': [0.56, 0.65, 0.7]},
                         'keyframes': [{'time_sec': 12.5,
                                        'attr': 'location',
                                        'value': [0.15000000000000013, -0.35, 1.24]},
                                       {'time_sec': 17.01,
                                        'attr': 'location',
                                        'value': [-0.13, -0.17, 1.3]}]},
                        {'name': 'trajectory_trace',
                         'type': 'mesh',
                         'primitive': 'curve_polyline',
                         'location': [0.0, 0.0, 0.0],
                         'properties': {'points': [[2.0, 0.25, 1.35],
                                                   [0.0, 0.0, 1.0],
                                                   [-2.0, -0.25, 0.95]],
                                        'bevel_depth': 0.012,
                                        'color': [0.2, 0.75, 1.0],
                                        'emission': 0.6},
                         'keyframes': []},
                        {'name': 'parameter_panel',
                         'type': 'primitive',
                         'primitive': 'plane',
                         'location': [2.4, 0.05, 1.45],
                         'properties': {'size': 1.2,
                                        'color': [0.08, 0.11, 0.14],
                                        'alpha': 0.35},
                         'keyframes': []}]}]}
COMPILED_SCENE_HINTS = {'text_objects_created': 0, 'arrow_or_tracer_primitives': 5}


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
    scene.cycles.device = 'GPU'
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
