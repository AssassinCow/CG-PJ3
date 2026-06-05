import math
import os
from pathlib import Path

import bpy
from mathutils import Vector


STORYBOARD = {
    'concept_id': 'affine_transformation',
    'fps': 24,
    'resolution': [960, 540],
    'shots': [
        {
            'node_id': 'node_01',
            'start_sec': 0.0,
            'duration_sec': 4.0,
            'kind': 'translation',
            'camera': [
                {'time_sec': 0.0, 'position': [6.0, -7.0, 4.5], 'look_at': [0.0, 0.0, 0.6], 'fov': 42.0},
                {'time_sec': 4.0, 'position': [6.2, -6.9, 4.55], 'look_at': [0.1, 0.05, 0.6], 'fov': 42.0},
            ],
        },
        {
            'node_id': 'node_02',
            'start_sec': 4.0,
            'duration_sec': 5.0,
            'kind': 'rotation',
            'camera': [
                {'time_sec': 4.0, 'position': [6.0, -7.0, 4.5], 'look_at': [0.0, 0.0, 0.6], 'fov': 42.0},
                {'time_sec': 9.0, 'position': [5.8, -7.1, 4.6], 'look_at': [0.0, 0.0, 0.6], 'fov': 42.0},
            ],
        },
        {
            'node_id': 'node_03',
            'start_sec': 9.0,
            'duration_sec': 4.0,
            'kind': 'scale',
            'camera': [
                {'time_sec': 9.0, 'position': [6.5, -7.5, 4.8], 'look_at': [0.0, 0.0, 0.6], 'fov': 44.0},
                {'time_sec': 13.0, 'position': [8.0, -9.0, 5.6], 'look_at': [0.0, 0.0, 0.6], 'fov': 46.0},
            ],
        },
        {
            'node_id': 'node_04',
            'start_sec': 13.0,
            'duration_sec': 5.0,
            'kind': 'composition',
            'camera': [
                {'time_sec': 13.0, 'position': [0.0, -10.5, 5.0], 'look_at': [0.0, 0.0, 1.2], 'fov': 52.0},
                {'time_sec': 18.0, 'position': [0.0, -10.5, 5.0], 'look_at': [0.0, 0.0, 1.2], 'fov': 52.0},
            ],
        },
    ],
}

FPS = STORYBOARD['fps']

_REQUIRED_NAMES = [
    'key_light', 'hero_mesh', 'source_shape_A', 'target_shape_B',
    'sample_vertex_marker', 'trajectory_trace', 'parameter_panel',
]


def frame_for_time(t):
    return round(float(t) * FPS) + 1


def lens_for_fov(fov_deg, sensor=36.0):
    a = math.radians(max(5.0, min(160.0, float(fov_deg))))
    return sensor / (2.0 * math.tan(a / 2.0))


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def make_mat(name, color, roughness=0.45, emission=0.0, alpha=None, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    a = float(alpha) if alpha is not None else (float(color[3]) if len(color) > 3 else 1.0)
    rgba = (float(color[0]), float(color[1]), float(color[2]), a)
    if bsdf:
        bsdf.inputs['Base Color'].default_value = rgba
        bsdf.inputs['Roughness'].default_value = roughness
        try:
            bsdf.inputs['Metallic'].default_value = metallic
        except KeyError:
            pass
        try:
            bsdf.inputs['Alpha'].default_value = a
        except KeyError:
            pass
        if emission > 0:
            try:
                bsdf.inputs['Emission Color'].default_value = rgba
            except KeyError:
                pass
            try:
                bsdf.inputs['Emission Strength'].default_value = emission
            except KeyError:
                pass
    if a < 0.999:
        mat.blend_method = 'BLEND'
        if hasattr(mat, 'show_transparent_back'):
            mat.show_transparent_back = True
    return mat


def add_text(name, location, body, size=0.28, color=(1.0, 0.96, 0.72), emission=0.4, rot=(math.radians(90), 0, 0)):
    bpy.ops.object.text_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.body = str(body)
    obj.data.align_x = 'CENTER'
    obj.data.align_y = 'CENTER'
    obj.data.size = size
    obj.rotation_euler = rot
    obj.data.materials.append(make_mat('mat_' + name, color, emission=emission))
    return obj


def add_curve_polyline(name, points, color=(0.2, 0.75, 1.0), bevel=0.014, emission=0.3, alpha=1.0):
    pts = [Vector(p) for p in points]
    curve = bpy.data.curves.new(name, type='CURVE')
    curve.dimensions = '3D'
    curve.bevel_depth = bevel
    curve.bevel_resolution = 3
    spl = curve.splines.new('POLY')
    spl.points.add(len(pts) - 1)
    for pt, co in zip(spl.points, pts):
        pt.co = (co.x, co.y, co.z, 1.0)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(make_mat('mat_' + name, color, emission=emission, alpha=alpha))
    return obj


def add_arrow(name, start, end, color=(1.0, 0.85, 0.15), shaft_r=0.035, head_r=0.11, head_len=0.22, emission=0.4):
    s = Vector(start)
    e = Vector(end)
    d = e - s
    L = max(d.length, 1e-4)
    direction = d.normalized()
    shaft_end = e - direction * head_len
    shaft_mid = (s + shaft_end) * 0.5
    bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=shaft_r, depth=max((shaft_end - s).length, 1e-4), location=shaft_mid)
    shaft = bpy.context.object
    shaft.name = name + '_shaft'
    shaft.rotation_euler = (shaft_end - s).to_track_quat('Z', 'Y').to_euler()
    mat = make_mat('mat_' + name, color, emission=emission, roughness=0.3)
    shaft.data.materials.append(mat)
    head_mid = shaft_end + direction * (head_len * 0.5)
    bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=head_r, radius2=0.0, depth=head_len, location=head_mid)
    head = bpy.context.object
    head.name = name + '_head'
    head.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
    head.data.materials.append(mat)
    head.parent = shaft
    shaft.name = name
    return shaft


def add_grid_plane(name='ground_grid', size=10.0, divisions=10, z=0.0, color=(0.18, 0.2, 0.24)):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, z - 0.005))
    base = bpy.context.object
    base.name = name + '_base'
    base.data.materials.append(make_mat('mat_' + base.name, color, roughness=0.85))
    half = size / 2.0
    step = size / divisions
    line_color = (0.45, 0.5, 0.55)
    for i in range(divisions + 1):
        x = -half + i * step
        add_curve_polyline(f'{name}_lx_{i}', [(x, -half, z), (x, half, z)],
                           color=line_color, bevel=0.006, emission=0.05)
        add_curve_polyline(f'{name}_ly_{i}', [(-half, x, z), (half, x, z)],
                           color=line_color, bevel=0.006, emission=0.05)
    return base


def add_axes(prefix, origin=(0, 0, 0), length=2.2):
    o = Vector(origin)
    add_arrow(prefix + '_axis_X', o, o + Vector((length, 0, 0)),
              color=(1.0, 0.18, 0.22), shaft_r=0.028, head_r=0.09, head_len=0.18, emission=0.6)
    add_arrow(prefix + '_axis_Y', o, o + Vector((0, length, 0)),
              color=(0.2, 0.9, 0.3), shaft_r=0.028, head_r=0.09, head_len=0.18, emission=0.6)
    add_arrow(prefix + '_axis_Z', o, o + Vector((0, 0, length)),
              color=(0.25, 0.55, 1.0), shaft_r=0.028, head_r=0.09, head_len=0.18, emission=0.6)
    add_text(prefix + '_lblX', o + Vector((length + 0.22, 0, 0.0)), 'x', size=0.26, color=(1.0, 0.4, 0.45))
    add_text(prefix + '_lblY', o + Vector((0, length + 0.22, 0.0)), 'y', size=0.26, color=(0.45, 1.0, 0.55))
    add_text(prefix + '_lblZ', o + Vector((0, 0, length + 0.22)), 'z', size=0.26, color=(0.5, 0.75, 1.0))


def add_cube_with_corner(name, location, size=0.8, color=(0.2, 0.45, 0.95), alpha=1.0,
                         wireframe=False, corner_color=(1.0, 0.85, 0.2)):
    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    obj = bpy.context.object
    obj.name = name
    mat = make_mat('mat_' + name, color, roughness=0.4, alpha=alpha)
    obj.data.materials.append(mat)
    if wireframe:
        obj.display_type = 'WIRE'
        if hasattr(obj, 'show_wire'):
            obj.show_wire = True
    half = size / 2.0
    marker_loc = (location[0] + half, location[1] + half, location[2] + half)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, radius=size * 0.13, location=marker_loc)
    corner = bpy.context.object
    corner.name = name + '_corner'
    corner.data.materials.append(make_mat('mat_' + corner.name, corner_color, emission=0.4))
    corner.parent = obj
    corner.matrix_parent_inverse = obj.matrix_world.inverted()
    return obj


# ----- scene init -----
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE'
scene.eevee.taa_render_samples = 16
if hasattr(scene.eevee, 'use_bloom'):
    scene.eevee.use_bloom = False
if hasattr(scene.eevee, 'use_ssr'):
    scene.eevee.use_ssr = False
if hasattr(scene.eevee, 'use_ssr_refraction'):
    scene.eevee.use_ssr_refraction = False
if hasattr(scene.eevee, 'use_motion_blur'):
    scene.eevee.use_motion_blur = False
if hasattr(scene.eevee, 'use_gtao'):
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 2.0
    scene.eevee.gtao_factor = 0.5

if scene.world is None:
    scene.world = bpy.data.worlds.new('World')
scene.world.use_nodes = True
bg = scene.world.node_tree.nodes.get('Background')
if bg:
    bg.inputs[0].default_value = (0.04, 0.045, 0.055, 1.0)
    bg.inputs[1].default_value = 1.0

scene.view_settings.view_transform = 'Filmic'
try:
    try:
        try:
            scene.view_settings.look = 'Medium High Contrast'
        except TypeError:
            scene.view_settings.look = 'AgX - Medium High Contrast'
    except TypeError:
        scene.view_settings.look = 'AgX - Medium High Contrast'
except TypeError:
    scene.view_settings.look = 'AgX - Medium High Contrast'
scene.view_settings.exposure = 0.2

scene.render.resolution_x = STORYBOARD['resolution'][0]
scene.render.resolution_y = STORYBOARD['resolution'][1]
scene.render.fps = FPS
scene.frame_start = 1
scene.frame_end = round(sum(s['duration_sec'] for s in STORYBOARD['shots']) * FPS)

out_dir = os.environ['CG_TUTOR_OUT_DIR']
Path(out_dir).mkdir(parents=True, exist_ok=True)
_render_path = os.path.join(out_dir, 'frame_####.png')
if not _render_path.startswith('\\\\'):
    _render_path = _render_path.replace('\\', '/')
scene.render.filepath = _render_path


# ----- shared key light -----
key_data = bpy.data.lights.new('key_light', type='AREA')
key_data.energy = 650.0
key_data.size = 5.0
key_data.color = (1.0, 0.96, 0.9)
key_light = bpy.data.objects.new('key_light', key_data)
bpy.context.collection.objects.link(key_light)
key_light.location = (3.0, -4.0, 6.0)

fill_data = bpy.data.lights.new('fill_light', type='AREA')
fill_data.energy = 220.0
fill_data.size = 6.0
fill_data.color = (0.7, 0.85, 1.0)
fill = bpy.data.objects.new('fill_light', fill_data)
bpy.context.collection.objects.link(fill)
fill.location = (-4.0, -3.0, 3.5)


def shot_frames(idx):
    cursor = 0
    for i, s in enumerate(STORYBOARD['shots']):
        n = round(s['duration_sec'] * FPS)
        if i == idx:
            return cursor + 1, cursor + n
        cursor += n
    return 1, FPS


visibility = {}


def add_visibility(name, start, end):
    visibility.setdefault(name, []).append((start, end))


add_visibility('key_light', 1, scene.frame_end)
add_visibility('fill_light', 1, scene.frame_end)


# ============================================================
# SHOT 1 - TRANSLATION
# ============================================================
s1_start, s1_end = shot_frames(0)

add_grid_plane('grid_s1', size=8.0, divisions=8, z=0.0)
add_axes('axes_s1', origin=(0, 0, 0), length=2.0)

t_start = Vector((-1.6, -1.0, 0.4))
t_end = Vector((1.4, 0.6, 1.0))

source_shape_A = add_cube_with_corner(
    'source_shape_A', tuple(t_start), size=0.8,
    color=(0.25, 0.55, 1.0), alpha=0.28, wireframe=False,
)

hero_mesh = add_cube_with_corner(
    'hero_mesh', tuple(t_start), size=0.8,
    color=(0.22, 0.5, 0.98), alpha=1.0,
)
hero_mesh.location = tuple(t_start)
hero_mesh.keyframe_insert('location', frame=s1_start)
hero_mesh.location = tuple(t_end)
hero_mesh.keyframe_insert('location', frame=s1_end)

target_shape_B = add_cube_with_corner(
    'target_shape_B', tuple(t_end), size=0.8,
    color=(1.0, 0.6, 0.2), alpha=0.22, wireframe=True,
)

half = 0.4
bpy.ops.mesh.primitive_uv_sphere_add(segments=22, ring_count=12, radius=0.09,
                                     location=(t_start.x + half, t_start.y + half, t_start.z + half))
sample_vertex_marker = bpy.context.object
sample_vertex_marker.name = 'sample_vertex_marker'
sample_vertex_marker.data.materials.append(make_mat('mat_svm', (1.0, 0.95, 0.3), emission=0.6))
sample_vertex_marker.location = (t_start.x + half, t_start.y + half, t_start.z + half)
sample_vertex_marker.keyframe_insert('location', frame=s1_start)
sample_vertex_marker.location = (t_end.x + half, t_end.y + half, t_end.z + half)
sample_vertex_marker.keyframe_insert('location', frame=s1_end)

trajectory_trace = add_curve_polyline(
    'trajectory_trace',
    [tuple(t_start + Vector((half, half, half))), tuple(t_end + Vector((half, half, half)))],
    color=(1.0, 0.55, 0.18), bevel=0.018, emission=0.6,
)

# Vector T cue 1: main yellow displacement arrow
add_arrow('vector_T_displacement', t_start, t_end,
          color=(1.0, 0.85, 0.15), shaft_r=0.035, head_r=0.12, head_len=0.24, emission=0.7)
# Vector T cue 2: parallel displacement arrow at the corner showing same t applies to every point
corner_off = Vector((half, half, half))
add_arrow('vector_T_corner_parallel', t_start + corner_off, t_end + corner_off,
          color=(1.0, 0.92, 0.4), shaft_r=0.022, head_r=0.09, head_len=0.18, emission=0.6)
add_text('vector_T_label', (t_start + t_end) * 0.5 + Vector((0.0, -0.35, 0.35)),
         't', size=0.34, color=(1.0, 0.9, 0.3), emission=0.6)

bpy.ops.mesh.primitive_plane_add(size=1.4, location=(-3.2, 0.5, 0.6))
parameter_panel = bpy.context.object
parameter_panel.name = 'parameter_panel'
parameter_panel.rotation_euler = (math.radians(90), 0, math.radians(18))
parameter_panel.data.materials.append(make_mat('mat_parameter_panel', (0.08, 0.11, 0.14), alpha=0.55))

shot1_objs = [source_shape_A, hero_mesh, target_shape_B, sample_vertex_marker, trajectory_trace, parameter_panel]
for o in shot1_objs:
    add_visibility(o.name, s1_start, s1_end)
for obj in bpy.data.objects:
    if obj.name.startswith(('grid_s1', 'axes_s1', 'vector_T_')):
        add_visibility(obj.name, s1_start, s1_end)


# ============================================================
# SHOT 2 - ROTATION about z
# ============================================================
s2_start, s2_end = shot_frames(1)

add_grid_plane('grid_s2', size=8.0, divisions=8, z=0.0)
add_axes('axes_s2', origin=(0, 0, 0), length=2.2)

add_arrow('axes_s2_zhighlight', (0, 0, 0), (0, 0, 2.6),
          color=(0.3, 0.6, 1.0), shaft_r=0.04, head_r=0.13, head_len=0.22, emission=0.9)

src2 = add_cube_with_corner(
    'source_shape_A_s2', (1.4, 0.0, 0.5), size=0.8,
    color=(0.25, 0.55, 1.0), alpha=0.25,
)

hero2 = add_cube_with_corner(
    'hero_mesh_s2', (1.4, 0.0, 0.5), size=0.8,
    color=(0.22, 0.5, 0.98), alpha=1.0,
)
hero2.location = (1.4, 0.0, 0.5)
hero2.rotation_euler = (0, 0, 0)
hero2.keyframe_insert('location', frame=s2_start)
hero2.keyframe_insert('rotation_euler', frame=s2_start)
theta_end = math.radians(110)
hx = 1.4 * math.cos(theta_end)
hy = 1.4 * math.sin(theta_end)
hero2.location = (hx, hy, 0.5)
hero2.rotation_euler = (0, 0, theta_end)
hero2.keyframe_insert('location', frame=s2_end)
hero2.keyframe_insert('rotation_euler', frame=s2_end)

tgt2_loc = (hx, hy, 0.5)
bpy.ops.mesh.primitive_cube_add(size=0.8, location=tgt2_loc)
tgt2 = bpy.context.object
tgt2.name = 'target_shape_B_s2'
tgt2.rotation_euler = (0, 0, theta_end)
tgt2.display_type = 'WIRE'
tgt2.data.materials.append(make_mat('mat_tgt2', (1.0, 0.6, 0.2), alpha=0.2))

svm2_start = Vector((1.4 + 0.4, 0.4, 0.9))
bpy.ops.mesh.primitive_uv_sphere_add(segments=22, ring_count=12, radius=0.09, location=tuple(svm2_start))
svm2 = bpy.context.object
svm2.name = 'sample_vertex_marker_s2'
svm2.data.materials.append(make_mat('mat_svm2', (1.0, 0.95, 0.3), emission=0.6))
svm2.keyframe_insert('location', frame=s2_start)
ox, oy = 0.4, 0.4
rx = (1.4 + ox) * math.cos(theta_end) - oy * math.sin(theta_end)
ry = (1.4 + ox) * math.sin(theta_end) + oy * math.cos(theta_end)
svm2.location = (rx, ry, 0.9)
svm2.keyframe_insert('location', frame=s2_end)

# Rotation arc (trajectory_trace cue)
arc_pts = []
arc_r = 1.4
for i in range(33):
    a = (i / 32.0) * theta_end
    arc_pts.append((arc_r * math.cos(a), arc_r * math.sin(a), 0.5))
trajectory_trace2 = add_curve_polyline(
    'trajectory_trace_s2', arc_pts,
    color=(1.0, 1.0, 1.0), bevel=0.02, emission=0.7,
)
# Vector R cue 1: arrowhead at end of arc
tan_dir = Vector((-math.sin(theta_end), math.cos(theta_end), 0)).normalized()
arc_end = Vector((arc_r * math.cos(theta_end), arc_r * math.sin(theta_end), 0.5))
add_arrow('vector_R_arc_head', arc_end - tan_dir * 0.001, arc_end + tan_dir * 0.25,
          color=(1.0, 1.0, 1.0), shaft_r=0.022, head_r=0.11, head_len=0.20, emission=0.8)
# Vector R cue 2: radial arrow from axis to start of arc (radius spoke)
add_arrow('vector_R_radius_spoke', (0, 0, 0.5), (1.4, 0.0, 0.5),
          color=(1.0, 0.9, 0.5), shaft_r=0.02, head_r=0.09, head_len=0.16, emission=0.6)
add_text('vector_R_label', (arc_r * math.cos(theta_end / 2) * 1.25, arc_r * math.sin(theta_end / 2) * 1.25, 0.8),
         'θ', size=0.36, color=(1.0, 1.0, 1.0), emission=0.7)

add_arrow('rh_thumb_up', (0.35, 0.35, 0.0), (0.35, 0.35, 1.6),
          color=(1.0, 0.8, 0.6), shaft_r=0.045, head_r=0.13, head_len=0.22, emission=0.4)
finger_pts = []
fr = 0.55
for i in range(17):
    a = (i / 16.0) * math.radians(220)
    finger_pts.append((0.35 + fr * math.cos(a), 0.35 + fr * math.sin(a), 1.1))
add_curve_polyline('rh_fingers', finger_pts, color=(1.0, 0.8, 0.6), bevel=0.02, emission=0.5, alpha=0.85)

bpy.ops.mesh.primitive_plane_add(size=1.4, location=(-3.2, 0.5, 0.6))
pp2 = bpy.context.object
pp2.name = 'parameter_panel_s2'
pp2.rotation_euler = (math.radians(90), 0, math.radians(18))
pp2.data.materials.append(make_mat('mat_pp2', (0.08, 0.11, 0.14), alpha=0.55))

shot2_local = [src2, hero2, tgt2, svm2, trajectory_trace2, pp2]
for o in shot2_local:
    add_visibility(o.name, s2_start, s2_end)
for obj in bpy.data.objects:
    if obj.name.startswith(('grid_s2', 'axes_s2', 'vector_R_', 'rh_thumb', 'rh_fingers')):
        add_visibility(obj.name, s2_start, s2_end)


# ============================================================
# SHOT 3 - UNIFORM SCALE
# ============================================================
s3_start, s3_end = shot_frames(2)

add_grid_plane('grid_s3', size=10.0, divisions=10, z=0.0)
add_axes('axes_s3', origin=(0, 0, 0), length=2.6)

bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.6))
wire_orig = bpy.context.object
wire_orig.name = 'source_shape_A_s3'
wire_orig.display_type = 'WIRE'
wire_orig.data.materials.append(make_mat('mat_wire3', (0.3, 0.65, 1.0), alpha=0.4))

bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.6))
hero3 = bpy.context.object
hero3.name = 'hero_mesh_s3'
hero3.data.materials.append(make_mat('mat_hero3', (0.22, 0.5, 0.98), roughness=0.4, alpha=0.85))
hero3.scale = (1.0, 1.0, 1.0)
hero3.keyframe_insert('scale', frame=s3_start)
hero3.scale = (2.0, 2.0, 2.0)
hero3.keyframe_insert('scale', frame=s3_end)

bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.6))
tgt3 = bpy.context.object
tgt3.name = 'target_shape_B_s3'
tgt3.scale = (2.0, 2.0, 2.0)
tgt3.display_type = 'WIRE'
tgt3.data.materials.append(make_mat('mat_tgt3', (1.0, 0.6, 0.2), alpha=0.25))

ray_objs = []
for i, (sx, sy, sz) in enumerate([(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]):
    end_pt = (sx * 1.0, sy * 1.0, 0.6 + sz * 1.0)
    r = add_curve_polyline(
        f'scale_ray_{i}',
        [(0, 0, 0.6), end_pt],
        color=(1.0, 0.55, 0.2), bevel=0.012, emission=0.6,
    )
    ray_objs.append(r)
    add_visibility(r.name, s3_start, s3_end)

svm3_start = (0.5, 0.5, 1.1)
bpy.ops.mesh.primitive_uv_sphere_add(segments=22, ring_count=12, radius=0.09, location=svm3_start)
svm3 = bpy.context.object
svm3.name = 'sample_vertex_marker_s3'
svm3.data.materials.append(make_mat('mat_svm3', (1.0, 0.95, 0.3), emission=0.6))
svm3.keyframe_insert('location', frame=s3_start)
svm3.location = (1.0, 1.0, 1.6)
svm3.keyframe_insert('location', frame=s3_end)

trajectory3 = add_curve_polyline(
    'trajectory_trace_s3', [svm3_start, (1.0, 1.0, 1.6)],
    color=(1.0, 0.55, 0.18), bevel=0.018, emission=0.6,
)

add_text('scale_label_x', (2.2, 0, 0.05), 's·x', size=0.28, color=(1.0, 0.4, 0.45))
add_text('scale_label_y', (0, 2.2, 0.05), 's·y', size=0.28, color=(0.45, 1.0, 0.55))
add_text('scale_label_z', (0, 0, 2.2), 's·z', size=0.28, color=(0.5, 0.75, 1.0))

bpy.ops.mesh.primitive_plane_add(size=1.4, location=(-3.6, 0.5, 0.6))
pp3 = bpy.context.object
pp3.name = 'parameter_panel_s3'
pp3.rotation_euler = (math.radians(90), 0, math.radians(18))
pp3.data.materials.append(make_mat('mat_pp3', (0.08, 0.11, 0.14), alpha=0.55))

shot3_local = [wire_orig, hero3, tgt3, svm3, trajectory3, pp3]
for o in shot3_local:
    add_visibility(o.name, s3_start, s3_end)
for obj in bpy.data.objects:
    if obj.name.startswith(('grid_s3', 'axes_s3', 'scale_label_')):
        add_visibility(obj.name, s3_start, s3_end)


# ============================================================
# SHOT 4 - COMPOSITION
# ============================================================
s4_start, s4_end = shot_frames(3)

LEFT_X = -3.2
RIGHT_X = 3.2

add_grid_plane('grid_s4L', size=5.0, divisions=5, z=0.0)
for obj in list(bpy.data.objects):
    if obj.name.startswith('grid_s4L'):
        obj.location = (obj.location.x + LEFT_X, obj.location.y, obj.location.z)
add_axes('axes_s4L', origin=(LEFT_X, 0, 0), length=1.4)

add_grid_plane('grid_s4R', size=5.0, divisions=5, z=0.0)
for obj in list(bpy.data.objects):
    if obj.name.startswith('grid_s4R'):
        obj.location = (obj.location.x + RIGHT_X, obj.location.y, obj.location.z)
add_axes('axes_s4R', origin=(RIGHT_X, 0, 0), length=1.4)

hero4L = add_cube_with_corner('hero_mesh_s4L', (LEFT_X, 0, 0.4), size=0.6,
                              color=(0.22, 0.5, 0.98), alpha=1.0)
srcL = add_cube_with_corner('source_shape_A_s4L', (LEFT_X, 0, 0.4), size=0.6,
                            color=(0.25, 0.55, 1.0), alpha=0.22)

mid1 = s4_start + (s4_end - s4_start) // 3
mid2 = s4_start + 2 * (s4_end - s4_start) // 3

hero4L.scale = (1, 1, 1); hero4L.location = (LEFT_X, 0, 0.4); hero4L.rotation_euler = (0, 0, 0)
hero4L.keyframe_insert('scale', frame=s4_start)
hero4L.keyframe_insert('rotation_euler', frame=s4_start)
hero4L.keyframe_insert('location', frame=s4_start)
hero4L.scale = (1.4, 1.4, 1.4)
hero4L.keyframe_insert('scale', frame=mid1)
hero4L.keyframe_insert('location', frame=mid1)
hero4L.keyframe_insert('rotation_euler', frame=mid1)
hero4L.rotation_euler = (0, 0, math.radians(60))
hero4L.keyframe_insert('rotation_euler', frame=mid2)
hero4L.keyframe_insert('scale', frame=mid2)
hero4L.keyframe_insert('location', frame=mid2)
finalL = (LEFT_X + 1.2, 0.6, 1.2)
hero4L.location = finalL
hero4L.keyframe_insert('location', frame=s4_end)
hero4L.keyframe_insert('rotation_euler', frame=s4_end)
hero4L.keyframe_insert('scale', frame=s4_end)

hero4R = add_cube_with_corner('hero_mesh_s4R', (RIGHT_X, 0, 0.4), size=0.6,
                              color=(0.22, 0.5, 0.98), alpha=1.0)
srcR = add_cube_with_corner('source_shape_A_s4R', (RIGHT_X, 0, 0.4), size=0.6,
                            color=(0.25, 0.55, 1.0), alpha=0.22)

pivotR = bpy.data.objects.new('pivotR', None)
bpy.context.collection.objects.link(pivotR)
pivotR.location = (RIGHT_X, 0, 0.4)
hero4R.parent = pivotR
hero4R.matrix_parent_inverse = pivotR.matrix_world.inverted()

hero4R.scale = (1, 1, 1); pivotR.rotation_euler = (0, 0, 0)
hero4R.keyframe_insert('scale', frame=s4_start)
pivotR.keyframe_insert('rotation_euler', frame=s4_start)
hero4R.location = (0, 0, 0)
hero4R.keyframe_insert('location', frame=s4_start)
hero4R.scale = (1.4, 1.4, 1.4)
hero4R.keyframe_insert('scale', frame=mid1)
hero4R.location = (1.2, 0.6, 0.8)
hero4R.keyframe_insert('location', frame=mid2)
hero4R.keyframe_insert('scale', frame=mid2)
pivotR.keyframe_insert('rotation_euler', frame=mid2)
pivotR.rotation_euler = (0, 0, math.radians(60))
pivotR.keyframe_insert('rotation_euler', frame=s4_end)
hero4R.keyframe_insert('location', frame=s4_end)
hero4R.keyframe_insert('scale', frame=s4_end)

bpy.ops.mesh.primitive_cube_add(size=0.6, location=finalL)
tgtL = bpy.context.object
tgtL.name = 'target_shape_B_s4L'
tgtL.rotation_euler = (0, 0, math.radians(60))
tgtL.scale = (1.4, 1.4, 1.4)
tgtL.display_type = 'WIRE'
tgtL.data.materials.append(make_mat('mat_tgtL', (1.0, 0.6, 0.2), alpha=0.25))

ang = math.radians(60)
lx, ly, lz = 1.2, 0.6, 0.8
fxR = RIGHT_X + (lx * math.cos(ang) - ly * math.sin(ang))
fyR = lx * math.sin(ang) + ly * math.cos(ang)
fzR = 0.4 + lz
finalR = (fxR, fyR, fzR)
bpy.ops.mesh.primitive_cube_add(size=0.6, location=finalR)
tgtR = bpy.context.object
tgtR.name = 'target_shape_B_s4R'
tgtR.rotation_euler = (0, 0, math.radians(60))
tgtR.scale = (1.4, 1.4, 1.4)
tgtR.display_type = 'WIRE'
tgtR.data.materials.append(make_mat('mat_tgtR', (1.0, 0.6, 0.2), alpha=0.25))


def sample_path(eval_fn, n=20):
    return [eval_fn(i / float(n - 1)) for i in range(n)]


def left_path(u):
    if u <= 1/3.0:
        return (LEFT_X, 0, 0.4)
    elif u <= 2/3.0:
        return (LEFT_X, 0, 0.4)
    else:
        v = (u - 2/3.0) / (1/3.0)
        return (LEFT_X + 1.2 * v, 0.6 * v, 0.4 + (1.2 - 0.4) * v)


def right_path(u):
    if u <= 1/3.0:
        return (RIGHT_X, 0, 0.4)
    elif u <= 2/3.0:
        v = (u - 1/3.0) / (1/3.0)
        return (RIGHT_X + 1.2 * v, 0.6 * v, 0.4 + 0.8 * v)
    else:
        v = (u - 2/3.0) / (1/3.0)
        a = math.radians(60) * v
        return (RIGHT_X + (1.2 * math.cos(a) - 0.6 * math.sin(a)),
                1.2 * math.sin(a) + 0.6 * math.cos(a),
                1.2)


trajL = add_curve_polyline('trajectory_trace_s4L', sample_path(left_path, 28),
                           color=(1.0, 0.55, 0.18), bevel=0.018, emission=0.6)
trajR = add_curve_polyline('trajectory_trace_s4R', sample_path(right_path, 28),
                           color=(1.0, 0.55, 0.18), bevel=0.018, emission=0.6)

n_dash = 14
for i in range(n_dash):
    if i % 2 == 0:
        u0 = i / float(n_dash)
        u1 = (i + 1) / float(n_dash)
        a = (finalL[0] + (finalR[0] - finalL[0]) * u0,
             finalL[1] + (finalR[1] - finalL[1]) * u0,
             finalL[2] + (finalR[2] - finalL[2]) * u0)
        b = (finalL[0] + (finalR[0] - finalL[0]) * u1,
             finalL[1] + (finalR[1] - finalL[1]) * u1,
             finalL[2] + (finalR[2] - finalL[2]) * u1)
        seg = add_curve_polyline(f'compare_dash_{i}', [a, b],
                                 color=(1.0, 0.2, 0.25), bevel=0.014, emission=0.7)
        add_visibility(seg.name, s4_start, s4_end)

add_text('label_left_order', (LEFT_X, -2.3, 0.05), 'T·R·S', size=0.32,
         color=(1.0, 0.95, 0.6), emission=0.5)
add_text('label_right_order', (RIGHT_X, -2.3, 0.05), 'R·T·S', size=0.32,
         color=(1.0, 0.95, 0.6), emission=0.5)

bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=10, radius=0.07,
                                     location=(LEFT_X + 0.3, 0.3, 0.7))
svm4L = bpy.context.object
svm4L.name = 'sample_vertex_marker_s4L'
svm4L.data.materials.append(make_mat('mat_svm4L', (1.0, 0.95, 0.3), emission=0.6))

bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=10, radius=0.07,
                                     location=(RIGHT_X + 0.3, 0.3, 0.7))
svm4R = bpy.context.object
svm4R.name = 'sample_vertex_marker_s4R'
svm4R.data.materials.append(make_mat('mat_svm4R', (1.0, 0.95, 0.3), emission=0.6))

bpy.ops.mesh.primitive_plane_add(size=1.6, location=(-5.5, 0.5, 1.2))
pp4 = bpy.context.object
pp4.name = 'parameter_panel_s4'
pp4.rotation_euler = (math.radians(90), 0, math.radians(20))
pp4.data.materials.append(make_mat('mat_pp4', (0.08, 0.11, 0.14), alpha=0.55))

shot4_local = [hero4L, srcL, hero4R, srcR, pivotR, tgtL, tgtR, trajL, trajR,
               svm4L, svm4R, pp4]
for o in shot4_local:
    add_visibility(o.name, s4_start, s4_end)
for obj in bpy.data.objects:
    if obj.name.startswith(('grid_s4', 'axes_s4', 'label_left_order', 'label_right_order')):
        add_visibility(obj.name, s4_start, s4_end)


# ============================================================
# Apply hide_render visibility windows
# ============================================================
for obj in bpy.data.objects:
    if obj.type == 'CAMERA':
        continue
    windows = visibility.get(obj.name)
    if not windows:
        continue
    obj.hide_render = True
    obj.keyframe_insert('hide_render', frame=1)
    for (ws, we) in windows:
        obj.hide_render = False
        obj.keyframe_insert('hide_render', frame=ws)
        obj.hide_render = True
        obj.keyframe_insert('hide_render', frame=we + 1)


# ============================================================
# CAMERAS
# ============================================================
for idx, shot in enumerate(STORYBOARD['shots']):
    s_start, s_end = shot_frames(idx)
    cam_data = bpy.data.cameras.new('cam_' + shot['node_id'])
    cam = bpy.data.objects.new('cam_' + shot['node_id'], cam_data)
    bpy.context.collection.objects.link(cam)
    first = shot['camera'][0]
    cam.location = tuple(first['position'])
    look_at(cam, first['look_at'])
    cam.data.sensor_width = 36.0
    cam.data.lens = lens_for_fov(first['fov'])
    marker = scene.timeline_markers.new('shot_' + shot['node_id'], frame=s_start)
    marker.camera = cam
    if idx == 0:
        scene.camera = cam
    for key in shot['camera']:
        f = frame_for_time(key['time_sec'])
        cam.location = tuple(key['position'])
        look_at(cam, key['look_at'])
        cam.data.lens = lens_for_fov(key['fov'])
        cam.keyframe_insert('location', frame=f)
        cam.keyframe_insert('rotation_euler', frame=f)


# ============================================================
# RENDER (preview frames support)
# ============================================================
_preview = os.environ.get('CG_TUTOR_PREVIEW_FRAMES', '').strip()
if _preview:
    frames = []
    for tok in _preview.split(','):
        tok = tok.strip()
        if tok:
            try:
                frames.append(int(tok))
            except ValueError:
                pass
    for fr in frames:
        scene.frame_set(fr)
        still_path = os.path.join(out_dir, f'frame_{fr:04d}.png')
        if not still_path.startswith('\\\\'):
            still_path = still_path.replace('\\', '/')
        scene.render.filepath = still_path
        bpy.ops.render.render(write_still=True)
else:
    bpy.ops.render.render(animation=True)