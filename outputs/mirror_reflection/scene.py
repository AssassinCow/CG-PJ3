import math
import os
from pathlib import Path

import bpy
from mathutils import Vector


STORYBOARD = {
    'concept_id': 'mirror_reflection',
    'fps': 24,
    'resolution': [960, 540],
    'shots': [
        {'node_id': 'node_01', 'start_sec': 0.0, 'duration_sec': 2.5},
        {'node_id': 'node_02', 'start_sec': 2.5, 'duration_sec': 4.0},
        {'node_id': 'node_03', 'start_sec': 6.5, 'duration_sec': 3.5},
        {'node_id': 'node_04', 'start_sec': 10.0, 'duration_sec': 5.0},
        {'node_id': 'node_05', 'start_sec': 15.0, 'duration_sec': 3.0},
    ],
}

FPS = STORYBOARD['fps']


def t2f(t):
    return round(float(t) * FPS) + 1


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def lens_for_fov(fov_deg, sensor_mm=36.0):
    a = math.radians(max(5.0, min(160.0, float(fov_deg))))
    return sensor_mm / (2.0 * math.tan(a / 2.0))


def make_mat(name, color, roughness=0.45, emission=0.0, alpha=None, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    a = float(alpha) if alpha is not None else (float(color[3]) if len(color) > 3 else 1.0)
    rgba = (float(color[0]), float(color[1]), float(color[2]), a)
    if bsdf:
        bsdf.inputs['Base Color'].default_value = rgba
        bsdf.inputs['Roughness'].default_value = roughness
        bsdf.inputs['Metallic'].default_value = metallic
        if 'Alpha' in bsdf.inputs:
            bsdf.inputs['Alpha'].default_value = a
        if emission > 0:
            bsdf.inputs['Emission Color'].default_value = rgba
            bsdf.inputs['Emission Strength'].default_value = emission
    if a < 0.999:
        mat.blend_method = 'BLEND'
        if hasattr(mat, 'show_transparent_back'):
            mat.show_transparent_back = True
    return mat


def add_text(name, location, body, size=0.22, color=(1.0, 0.96, 0.72), emission=0.6,
             rotation_euler=(math.radians(75), 0, 0), align='CENTER'):
    bpy.ops.object.text_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.body = str(body)
    obj.data.align_x = align
    obj.data.align_y = 'CENTER'
    obj.data.size = float(size)
    obj.rotation_euler = rotation_euler
    mat = make_mat('mat_' + name, color, emission=emission)
    obj.data.materials.append(mat)
    return obj


def add_curve_polyline(name, points, color=(0.95, 0.35, 0.25), bevel_depth=0.018,
                       emission=0.7, alpha=1.0):
    pts = [Vector(p) for p in points]
    curve = bpy.data.curves.new(name + '_curve', type='CURVE')
    curve.dimensions = '3D'
    curve.bevel_factor_mapping_start = 'SPLINE'
    curve.bevel_factor_mapping_end = 'SPLINE'
    curve.bevel_factor_start = 0.0
    curve.bevel_factor_end = 1.0
    curve.resolution_u = 2
    curve.bevel_depth = bevel_depth
    curve.bevel_resolution = 3
    spl = curve.splines.new('POLY')
    spl.points.add(len(pts) - 1)
    for p, co in zip(spl.points, pts):
        p.co = (co.x, co.y, co.z, 1.0)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(make_mat('mat_' + name, color, emission=emission, alpha=alpha))
    return obj


def add_arrow(name, start, end, color=(0.9, 0.9, 0.95), shaft_radius=0.018,
              head_length=0.16, head_radius=0.05, emission=0.4):
    start = Vector(start)
    end = Vector(end)
    direction = end - start
    length = direction.length
    if length < 1e-5:
        return None
    direction.normalize()
    tip = end
    shaft_end = end - direction * head_length
    shaft_mid = (start + shaft_end) * 0.5
    shaft_len = max((shaft_end - start).length, 1e-4)
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=shaft_radius, depth=shaft_len, location=shaft_mid)
    shaft = bpy.context.object
    shaft.name = name + '_shaft'
    shaft.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
    mat = make_mat('mat_' + name, color, emission=emission)
    shaft.data.materials.append(mat)
    head_mid = shaft_end + direction * (head_length * 0.5)
    bpy.ops.mesh.primitive_cone_add(vertices=20, radius1=head_radius, radius2=0.0,
                                    depth=head_length, location=head_mid)
    head = bpy.context.object
    head.name = name + '_head'
    head.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
    head.data.materials.append(mat)
    head.parent = shaft
    shaft.name = name
    return shaft


def add_angle_arc(name, vertex, dir_a, dir_b, radius=0.28, segments=18,
                  color=(1.0, 0.7, 0.2), bevel_depth=0.012, emission=0.6):
    vertex = Vector(vertex)
    a = Vector(dir_a).normalized()
    b = Vector(dir_b).normalized()
    dot = max(-1.0, min(1.0, a.dot(b)))
    angle = math.acos(dot)
    axis = a.cross(b)
    if axis.length < 1e-5:
        axis = Vector((0, 1, 0))
    axis.normalize()
    pts = []
    for i in range(segments + 1):
        t = i / segments
        from mathutils import Quaternion
        q = Quaternion(axis, angle * t)
        d = q @ a
        pts.append(vertex + d * radius)
    return add_curve_polyline(name, pts, color=color, bevel_depth=bevel_depth, emission=emission)


def hide_all_then_show(obj, ranges, total_end):
    """For each child too. Hide outside given (start, end) ranges."""
    targets = [obj]
    def collect(o):
        for c in o.children:
            targets.append(c)
            collect(c)
    collect(obj)
    for t in targets:
        t.hide_render = True
        t.hide_viewport = True
        t.keyframe_insert('hide_render', frame=1)
        t.keyframe_insert('hide_viewport', frame=1)
        for s, e in ranges:
            t.hide_render = False
            t.hide_viewport = False
            t.keyframe_insert('hide_render', frame=s)
            t.keyframe_insert('hide_viewport', frame=s)
            t.hide_render = True
            t.hide_viewport = True
            t.keyframe_insert('hide_render', frame=e + 1)
            t.keyframe_insert('hide_viewport', frame=e + 1)


# ---------------- SCENE BOOT ----------------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 32
if hasattr(scene.cycles, 'use_denoising'):
    scene.cycles.use_denoising = False
if hasattr(scene.cycles, 'denoiser'):
    try:
        scene.cycles.denoiser = 'NONE'
    except Exception:
        pass
for vl in scene.view_layers:
    if hasattr(vl, 'cycles') and hasattr(vl.cycles, 'use_denoising'):
        vl.cycles.use_denoising = False
scene.cycles.device = 'GPU'
prefs = bpy.context.preferences.addons.get('cycles')
if prefs is not None:
    cprefs = prefs.preferences
    for ct in ('OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI'):
        try:
            cprefs.compute_device_type = ct
            cprefs.get_devices()
            enabled = False
            for dev in cprefs.devices:
                if getattr(dev, 'type', '') != 'CPU':
                    dev.use = True
                    enabled = True
                else:
                    dev.use = False
            if enabled:
                print('[cg-tutor] cycles GPU', ct)
                break
        except Exception:
            continue

if scene.world:
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs[0].default_value = (0.06, 0.07, 0.09, 1.0)
        bg.inputs[1].default_value = 0.6

scene.view_settings.view_transform = 'Filmic'
try:
    scene.view_settings.look = 'Medium High Contrast'
except TypeError:
    scene.view_settings.look = 'AgX - Medium High Contrast'
scene.view_settings.exposure = 0.2
scene.render.resolution_x = STORYBOARD['resolution'][0]
scene.render.resolution_y = STORYBOARD['resolution'][1]
scene.render.fps = FPS
scene.frame_start = 1

total_dur = sum(s['duration_sec'] for s in STORYBOARD['shots'])
scene.frame_end = round(total_dur * FPS)

out_dir = os.environ['CG_TUTOR_OUT_DIR']
Path(out_dir).mkdir(parents=True, exist_ok=True)
_render_path = os.path.join(out_dir, 'frame_####.png')
if not _render_path.startswith('\\\\'):
    _render_path = _render_path.replace('\\', '/')
scene.render.filepath = _render_path

# Shot frame ranges
shot_ranges = {}
cursor = 0
for s in STORYBOARD['shots']:
    n = round(s['duration_sec'] * FPS)
    shot_ranges[s['node_id']] = (cursor + 1, cursor + n)
    cursor += n
TOTAL_END = scene.frame_end

# ---------------- LIGHTING ----------------
key_data = bpy.data.lights.new('key_light', type='AREA')
key_data.energy = 800.0
key_data.size = 5.0
key_data.color = (1.0, 0.96, 0.9)
key_light = bpy.data.objects.new('key_light', key_data)
bpy.context.collection.objects.link(key_light)
key_light.location = (0.0, -3.5, 5.0)

fill_data = bpy.data.lights.new('fill_light', type='AREA')
fill_data.energy = 250.0
fill_data.size = 6.0
fill_data.color = (0.85, 0.92, 1.0)
fill = bpy.data.objects.new('fill_light', fill_data)
bpy.context.collection.objects.link(fill)
fill.location = (-4.0, -2.0, 3.5)

# ---------------- PERSISTENT GROUNDING GEOMETRY ----------------

# Soft reflective floor
bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0.0, 0.0, 0.0))
soft_floor_plane = bpy.context.object
soft_floor_plane.name = 'soft_floor_plane'
floor_mat = make_mat('mat_floor', (0.18, 0.22, 0.28, 1.0), roughness=0.35, metallic=0.2)
soft_floor_plane.data.materials.append(floor_mat)

# Mirrors: two upright parallel rectangular panels facing each other
# Place them on left (M1) and right (M2), separated along X.
def make_mirror(name, x, border_color):
    # Glass-like dark panel
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, 0.0, 1.3))
    panel = bpy.context.object
    panel.name = name
    panel.scale = (0.06, 1.6, 1.2)  # thin in X, wide in Y, tall in Z
    panel_mat = make_mat('mat_' + name, (0.05, 0.07, 0.1, 1.0), roughness=0.08, metallic=0.95)
    panel.data.materials.append(panel_mat)
    # Border frame (thin emissive cube edges)
    frame_color = border_color
    fm = make_mat('mat_' + name + '_frame', frame_color, emission=0.9)
    # top/bottom borders
    sx = 0.07
    sy_full = 1.65
    sz_full = 1.25
    border_t = 0.04
    parts = []
    # top
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, 0.0, 1.3 + sz_full))
    o = bpy.context.object; o.name = name + '_border_top'
    o.scale = (sx, sy_full, border_t); o.data.materials.append(fm); parts.append(o)
    # bottom
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, 0.0, 1.3 - sz_full))
    o = bpy.context.object; o.name = name + '_border_bot'
    o.scale = (sx, sy_full, border_t); o.data.materials.append(fm); parts.append(o)
    # left (-Y)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, -sy_full, 1.3))
    o = bpy.context.object; o.name = name + '_border_left'
    o.scale = (sx, border_t, sz_full); o.data.materials.append(fm); parts.append(o)
    # right (+Y)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x, sy_full, 1.3))
    o = bpy.context.object; o.name = name + '_border_right'
    o.scale = (sx, border_t, sz_full); o.data.materials.append(fm); parts.append(o)
    for p in parts:
        p.parent = panel
    return panel

# M1 on left (negative X), M2 on right (positive X)
M1_X = -2.4
M2_X = 2.4
mirror_panel_m1 = make_mirror('mirror_panel_m1', M1_X, (0.95, 0.35, 0.25, 1.0))
mirror_panel_m2 = make_mirror('mirror_panel_m2', M2_X, (0.30, 0.75, 1.0, 1.0))

# Mirror text labels (M1 / M2) - placed above each panel
m1_label = add_text('label_M1', (M1_X, 0.0, 2.85), 'M1', size=0.42,
                    color=(0.95, 0.45, 0.30), emission=1.2,
                    rotation_euler=(math.radians(80), 0, 0))
m2_label = add_text('label_M2', (M2_X, 0.0, 2.85), 'M2', size=0.42,
                    color=(0.40, 0.80, 1.0), emission=1.2,
                    rotation_euler=(math.radians(80), 0, 0))

# Laser source S - small box near front-left
bpy.ops.mesh.primitive_cube_add(size=0.32, location=(M1_X + 0.4, -1.0, 0.9))
laser_source_s = bpy.context.object
laser_source_s.name = 'laser_source_s'
laser_source_s.data.materials.append(make_mat('mat_laser', (0.95, 0.35, 0.25, 1.0),
                                              roughness=0.3, emission=0.3))
s_label = add_text('label_S', (M1_X + 0.4, -1.0, 1.35), 'S', size=0.32,
                   color=(1.0, 0.55, 0.35), emission=1.2,
                   rotation_euler=(math.radians(80), 0, 0))

# Target sphere T - back of the stage
bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.22,
                                     location=(M1_X + 0.4, 1.2, 0.9))
target_sphere_t = bpy.context.object
target_sphere_t.name = 'target_sphere_t'
target_sphere_t.data.materials.append(make_mat('mat_target', (0.30, 0.85, 0.55, 1.0),
                                               roughness=0.3, emission=0.3))
t_label = add_text('label_T', (M1_X + 0.4, 1.2, 1.35), 'T', size=0.32,
                   color=(0.4, 1.0, 0.65), emission=1.2,
                   rotation_euler=(math.radians(80), 0, 0))

# Alternating bounce points along mirrors:
# Path: S(left,-1.0) -> P1 on M2 -> P2 on M1 -> P3 on M2 -> P4 on M1 -> P5 on M2 -> T(left,+1.2)
S_POS = Vector((M1_X + 0.4, -1.0, 0.9))
T_POS = Vector((M1_X + 0.4, 1.2, 0.9))
INSET = 0.04  # slightly off mirror surface
P_POSITIONS = [
    Vector((M2_X - INSET, -0.7, 1.2)),  # P1 on M2
    Vector((M1_X + INSET, -0.3, 1.5)),  # P2 on M1
    Vector((M2_X - INSET, 0.1,  1.1)),  # P3 on M2
    Vector((M1_X + INSET, 0.5,  1.6)),  # P4 on M1
    Vector((M2_X - INSET, 0.9,  1.0)),  # P5 on M2
]

# Group parent for bounce points
bounce_points_p1_to_p5 = bpy.data.objects.new('bounce_points_p1_to_p5', None)
bpy.context.collection.objects.link(bounce_points_p1_to_p5)
bounce_points_p1_to_p5.empty_display_type = 'PLAIN_AXES'

bounce_objs = []
for i, p in enumerate(P_POSITIONS):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, radius=0.07, location=p)
    sp = bpy.context.object
    sp.name = f'bounce_point_p{i+1}'
    sp.data.materials.append(make_mat('mat_' + sp.name, (1.0, 0.85, 0.25, 1.0), emission=1.4))
    sp.parent = bounce_points_p1_to_p5
    bounce_objs.append(sp)
    # Label Pi - offset away from mirror toward camera (-Y) and upward
    on_m2 = (i % 2 == 0)
    label_off = Vector((-0.2 if on_m2 else 0.2, -0.15, 0.22))
    lbl = add_text(f'label_P{i+1}', p + label_off, f'P{i+1}', size=0.22,
                   color=(1.0, 0.92, 0.55), emission=1.2,
                   rotation_euler=(math.radians(80), 0, 0))
    lbl.parent = bounce_points_p1_to_p5

# Normal arrows - one per bounce point, perpendicular to its mirror
normal_arrows_n1_to_n5 = bpy.data.objects.new('normal_arrows_n1_to_n5', None)
bpy.context.collection.objects.link(normal_arrows_n1_to_n5)
normal_arrows_n1_to_n5.empty_display_type = 'PLAIN_AXES'

normal_arrow_objs = []
for i, p in enumerate(P_POSITIONS):
    on_m2 = (i % 2 == 0)
    # Normal points from mirror surface inward into the gap (toward -X if M2, toward +X if M1)
    n_dir = Vector((-1.0, 0.0, 0.0)) if on_m2 else Vector((1.0, 0.0, 0.0))
    n_len = 0.6
    arrow = add_arrow(
        f'normal_arrow_n{i+1}',
        p, p + n_dir * n_len,
        color=(0.55, 0.95, 0.65), shaft_radius=0.022,
        head_length=0.14, head_radius=0.06, emission=1.0,
    )
    if arrow:
        arrow.parent = normal_arrows_n1_to_n5
        normal_arrow_objs.append(arrow)
        # Label N_i at the tip
        tip = p + n_dir * (n_len + 0.12)
        nlbl = add_text(f'label_N{i+1}', tip + Vector((0, -0.05, 0.12)),
                        f'N{i+1}', size=0.18,
                        color=(0.65, 1.0, 0.75), emission=1.2,
                        rotation_euler=(math.radians(80), 0, 0))
        nlbl.parent = normal_arrows_n1_to_n5

# Multi-bounce ray path (full path, will be sequentially revealed via bevel_factor_end)
RAY_POINTS = [S_POS] + P_POSITIONS + [T_POS]
multi_bounce_ray_path = add_curve_polyline(
    'multi_bounce_ray_path', RAY_POINTS,
    color=(1.0, 0.55, 0.2), bevel_depth=0.022, emission=1.6,
)

# Angle arcs at each bounce point (theta_i and theta_r) - tiny pairs
angle_arc_group = bpy.data.objects.new('angle_arc_group', None)
bpy.context.collection.objects.link(angle_arc_group)
angle_arc_group.empty_display_type = 'PLAIN_AXES'

angle_arc_objs = []
ray_seq = [S_POS] + P_POSITIONS + [T_POS]
for i, p in enumerate(P_POSITIONS):
    incoming_from = ray_seq[i]      # previous point
    outgoing_to = ray_seq[i + 2]    # next point
    on_m2 = (i % 2 == 0)
    n_dir = Vector((-1.0, 0.0, 0.0)) if on_m2 else Vector((1.0, 0.0, 0.0))
    d_in = (incoming_from - p).normalized()
    d_out = (outgoing_to - p).normalized()
    arc_i = add_angle_arc(f'angle_arc_theta_i_{i+1}', p, n_dir, d_in,
                          radius=0.22, segments=14,
                          color=(1.0, 0.4, 0.4), bevel_depth=0.012, emission=1.2)
    arc_r = add_angle_arc(f'angle_arc_theta_r_{i+1}', p, n_dir, d_out,
                          radius=0.22, segments=14,
                          color=(0.4, 0.7, 1.0), bevel_depth=0.012, emission=1.2)
    if arc_i: arc_i.parent = angle_arc_group; angle_arc_objs.append(arc_i)
    if arc_r: arc_r.parent = angle_arc_group; angle_arc_objs.append(arc_r)

# Theta labels at first bounce only (avoid clutter)
theta_i_label = add_text('label_theta_i', P_POSITIONS[0] + Vector((-0.45, -0.1, 0.18)),
                         'θi', size=0.18, color=(1.0, 0.55, 0.55), emission=1.2,
                         rotation_euler=(math.radians(80), 0, 0))
theta_r_label = add_text('label_theta_r', P_POSITIONS[0] + Vector((-0.45, -0.1, -0.05)),
                         'θr', size=0.18, color=(0.55, 0.8, 1.0), emission=1.2,
                         rotation_euler=(math.radians(80), 0, 0))
theta_i_label.parent = angle_arc_group
theta_r_label.parent = angle_arc_group

# ---------------- ANIMATION: SEQUENTIAL PATH REVEAL ----------------

# Reveal multi_bounce_ray_path piecewise across shots:
# - shot 1: hidden (path not yet shown)
# - shot 2: reveal segments 1->2 of 6 segments (S->P1, P1->P2 partial)
# - shot 3: reveal up to segment 3 (P2->P3 reaches M2 again)
# - shot 4: reveal full path while source rotates (animate end 0->1)
# - shot 5: full path locked at 1.0
N_SEG = 6  # S-P1, P1-P2, P2-P3, P3-P4, P4-P5, P5-T

def set_bevel_end(curve_data, value, frame):
    curve_data.bevel_factor_end = value
    curve_data.keyframe_insert('bevel_factor_end', frame=frame)

ray_data = multi_bounce_ray_path.data
# initial value at frame 1: 0
ray_data.bevel_factor_end = 0.0
ray_data.keyframe_insert('bevel_factor_end', frame=1)

s2_start, s2_end = shot_ranges['node_02']
s3_start, s3_end = shot_ranges['node_03']
s4_start, s4_end = shot_ranges['node_04']
s5_start, s5_end = shot_ranges['node_05']

# Shot 2: grow from 0 to 2/6 across the shot
set_bevel_end(ray_data, 0.0, s2_start)
set_bevel_end(ray_data, 2.0 / N_SEG, s2_end)
# Shot 3: grow to 3/6 (reaches P3 on M2)
set_bevel_end(ray_data, 2.0 / N_SEG, s3_start)
set_bevel_end(ray_data, 3.0 / N_SEG, s3_end)
# Shot 4: grow to full path while source "pivots"
set_bevel_end(ray_data, 3.0 / N_SEG, s4_start)
set_bevel_end(ray_data, 1.0, s4_end)
# Shot 5: hold full path
set_bevel_end(ray_data, 1.0, s5_start)
set_bevel_end(ray_data, 1.0, s5_end)

# Required: explicit storyboard keyframes on multi_bounce_ray_path and normal_arrows_n1_to_n5
# (location/scale animation, distinct from bevel_factor_end visibility reveal)
def kf_scale(obj, frame, sx, sy, sz):
    obj.scale = (sx, sy, sz)
    obj.keyframe_insert('scale', frame=frame)

# multi_bounce_ray_path scale keyframes per shot
for shot_id, kfs in [
    ('node_01', [(0.0, 0.72), (2.05, 1.0)]),
    ('node_02', [(2.5, 1.0), (5.78, 1.0)]),
    ('node_03', [(6.5, 1.0), (9.37, 1.0)]),
    ('node_04', [(10.0, 1.0), (14.1, 1.0)]),
    ('node_05', [(15.0, 1.0), (17.46, 1.0)]),
]:
    for tsec, sval in kfs:
        kf_scale(multi_bounce_ray_path, t2f(tsec), sval, sval, sval)

# normal_arrows group scale keyframes (gentle pulse for emphasis)
for shot_id, kfs in [
    ('node_01', [(0.0, 0.72), (2.05, 1.08)]),
    ('node_02', [(2.5, 0.85), (5.78, 1.05)]),
    ('node_03', [(6.5, 0.85), (9.37, 1.05)]),
    ('node_04', [(10.0, 0.85), (14.1, 1.05)]),
    ('node_05', [(15.0, 0.95), (17.46, 1.0)]),
]:
    for tsec, sval in kfs:
        kf_scale(normal_arrows_n1_to_n5, t2f(tsec), sval, sval, sval)

# Source pivot in shot 4 (rotate around Z) - small wiggle
laser_source_s.rotation_euler = (0, 0, 0)
laser_source_s.keyframe_insert('rotation_euler', frame=s4_start)
laser_source_s.rotation_euler = (0, 0, math.radians(15))
laser_source_s.keyframe_insert('rotation_euler', frame=(s4_start + s4_end) // 2)
laser_source_s.rotation_euler = (0, 0, math.radians(-10))
laser_source_s.keyframe_insert('rotation_euler', frame=s4_end)

# ---------------- VISIBILITY GATING ----------------

# Persistent objects (visible whole movie)
PERSISTENT = [
    soft_floor_plane, mirror_panel_m1, mirror_panel_m2,
    m1_label, m2_label,
    laser_source_s, target_sphere_t, s_label, t_label,
    bounce_points_p1_to_p5,
    normal_arrows_n1_to_n5,
    multi_bounce_ray_path,
    key_light, fill,
]
# Children of empties propagate visibility via parent's keyframes too,
# but we need explicit keyframes on every renderable since hide_render
# is per-object. Helper:
def show_full_run(obj):
    obj.hide_render = False
    obj.hide_viewport = False
    obj.keyframe_insert('hide_render', frame=1)
    obj.keyframe_insert('hide_viewport', frame=1)

def collect_renderables(root):
    out = [root]
    for c in root.children:
        out.extend(collect_renderables(c))
    return out

for o in PERSISTENT:
    for r in collect_renderables(o):
        show_full_run(r)

# Angle arcs only visible from shot 2 onward
def gate(obj, ranges):
    targets = collect_renderables(obj)
    for t in targets:
        t.hide_render = True
        t.hide_viewport = True
        t.keyframe_insert('hide_render', frame=1)
        t.keyframe_insert('hide_viewport', frame=1)
        for s, e in ranges:
            t.hide_render = False
            t.hide_viewport = False
            t.keyframe_insert('hide_render', frame=s)
            t.keyframe_insert('hide_viewport', frame=s)
            t.hide_render = True
            t.hide_viewport = True
            t.keyframe_insert('hide_render', frame=e + 1)
            t.keyframe_insert('hide_viewport', frame=e + 1)

# Angle arcs and theta labels visible from shot2..shot5
gate(angle_arc_group, [(s2_start, s5_end)])

# ---------------- CAMERAS PER SHOT ----------------

# Camera plan - simple stable three-quarter view
SHOT_CAMS = {
    'node_01': [
        {'time_sec': 0.0, 'pos': (0.0, -7.5, 3.4), 'look': (0.0, 0.0, 1.2), 'fov': 48.0},
        {'time_sec': 2.5, 'pos': (0.6, -7.4, 3.45), 'look': (0.05, 0.0, 1.2), 'fov': 48.0},
    ],
    'node_02': [
        {'time_sec': 2.5, 'pos': (-0.8, -6.6, 3.0), 'look': (M2_X * 0.4, -0.4, 1.3), 'fov': 46.0},
        {'time_sec': 6.5, 'pos': (-0.4, -6.7, 3.1), 'look': (M2_X * 0.3, -0.2, 1.3), 'fov': 46.0},
    ],
    'node_03': [
        {'time_sec': 6.5, 'pos': (0.4, -6.8, 3.1), 'look': (0.0, 0.0, 1.3), 'fov': 47.0},
        {'time_sec': 10.0, 'pos': (0.8, -6.7, 3.2), 'look': (0.1, 0.05, 1.3), 'fov': 47.0},
    ],
    'node_04': [
        {'time_sec': 10.0, 'pos': (0.0, -7.6, 3.6), 'look': (0.0, 0.0, 1.25), 'fov': 50.0},
        {'time_sec': 15.0, 'pos': (-0.6, -7.4, 3.55), 'look': (-0.1, 0.05, 1.25), 'fov': 50.0},
    ],
    'node_05': [
        {'time_sec': 15.0, 'pos': (0.0, -7.4, 3.4), 'look': (0.0, 0.0, 1.25), 'fov': 48.0},
        {'time_sec': 18.0, 'pos': (1.6, -7.0, 3.5), 'look': (0.2, 0.0, 1.25), 'fov': 48.0},
    ],
}

shot_cams = {}
for idx, sb_shot in enumerate(STORYBOARD['shots']):
    sid = sb_shot['node_id']
    cam_data = bpy.data.cameras.new('camera_' + sid)
    cam_data.sensor_width = 36.0
    cam = bpy.data.objects.new('camera_' + sid, cam_data)
    bpy.context.collection.objects.link(cam)
    keys = SHOT_CAMS[sid]
    first = keys[0]
    cam.location = first['pos']
    look_at(cam, first['look'])
    cam.data.lens = lens_for_fov(first['fov'])
    for k in keys:
        f = t2f(k['time_sec'])
        cam.location = k['pos']
        look_at(cam, k['look'])
        cam.data.lens = lens_for_fov(k['fov'])
        cam.keyframe_insert('location', frame=f)
        cam.keyframe_insert('rotation_euler', frame=f)
        cam.data.keyframe_insert('lens', frame=f)
    s, _e = shot_ranges[sid]
    marker = scene.timeline_markers.new('shot_' + sid, frame=s)
    marker.camera = cam
    if idx == 0:
        scene.camera = cam
    shot_cams[sid] = cam

# ---------------- PREVIEW / RENDER ----------------
preview_raw = os.environ.get('CG_TUTOR_PREVIEW_FRAMES', '').strip()
if preview_raw:
    for raw in preview_raw.split(','):
        if not raw.strip():
            continue
        frame = int(raw)
        scene.frame_set(frame)
        _p = os.path.join(out_dir, f'frame_{frame:04d}.png')
        if not _p.startswith('\\\\'):
            _p = _p.replace('\\', '/')
        scene.render.filepath = _p
        bpy.ops.render.render(write_still=True)
else:
    bpy.ops.render.render(animation=True)