import math
import os
from pathlib import Path

import bpy
import bmesh
from mathutils import Vector, Quaternion


STORYBOARD = {
    'concept_id': 'prism_dispersion_teaching',
    'fps': 24,
    'resolution': [960, 540],
    'shots': [
        {'node_id': 'node_01', 'duration_sec': 3.5,
         'camera': [{'time_sec': 0.0, 'position': [0.0, -7.0, 3.2], 'look_at': [0.0, 0.0, 0.9], 'fov': 48.0},
                    {'time_sec': 3.5, 'position': [0.8, -6.88, 3.26], 'look_at': [0.08, 0.0, 0.9], 'fov': 48.0}]},
        {'node_id': 'node_02', 'duration_sec': 4.0,
         'camera': [{'time_sec': 3.5, 'position': [-1.6, -5.4, 2.6], 'look_at': [-0.6, 0.0, 1.1], 'fov': 40.0},
                    {'time_sec': 7.5, 'position': [-1.8, -5.2, 2.65], 'look_at': [-0.6, 0.0, 1.1], 'fov': 40.0}]},
        {'node_id': 'node_03', 'duration_sec': 4.5,
         'camera': [{'time_sec': 7.5, 'position': [0.6, -6.0, 2.7], 'look_at': [0.5, 0.0, 1.0], 'fov': 42.0},
                    {'time_sec': 12.0, 'position': [1.0, -5.9, 2.75], 'look_at': [0.7, 0.0, 1.0], 'fov': 42.0}]},
        {'node_id': 'node_04', 'duration_sec': 4.0,
         'camera': [{'time_sec': 12.0, 'position': [0.0, -7.2, 3.3], 'look_at': [0.0, 0.0, 1.0], 'fov': 50.0},
                    {'time_sec': 16.0, 'position': [-0.4, -7.1, 3.35], 'look_at': [-0.05, 0.0, 1.0], 'fov': 50.0}]},
    ],
}

FPS = STORYBOARD['fps']


def frame_for_time(t):
    return round(float(t) * FPS) + 1


def lens_for_fov(fov_deg, sensor_mm=36.0):
    a = math.radians(max(5.0, min(160.0, float(fov_deg))))
    return sensor_mm / (2.0 * math.tan(a / 2.0))


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def make_mat(name, color, roughness=0.5, emission=0.0, alpha=1.0, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    rgba = (color[0], color[1], color[2], alpha)
    if bsdf:
        bsdf.inputs['Base Color'].default_value = rgba
        bsdf.inputs['Roughness'].default_value = roughness
        try:
            bsdf.inputs['Metallic'].default_value = metallic
        except KeyError:
            pass
        try:
            bsdf.inputs['Alpha'].default_value = alpha
        except KeyError:
            pass
        if emission > 0:
            try:
                bsdf.inputs['Emission Color'].default_value = rgba
                bsdf.inputs['Emission Strength'].default_value = emission
            except KeyError:
                pass
    if alpha < 0.999:
        mat.blend_method = 'BLEND'
        if hasattr(mat, 'show_transparent_back'):
            mat.show_transparent_back = False
    return mat


def add_curve_polyline(name, points, color=(1, 1, 1), bevel=0.012, emission=1.0, alpha=1.0):
    curve = bpy.data.curves.new(name, type='CURVE')
    curve.dimensions = '3D'
    curve.bevel_depth = bevel
    curve.bevel_resolution = 3
    spl = curve.splines.new('POLY')
    spl.points.add(len(points) - 1)
    for p, co in zip(spl.points, points):
        p.co = (co[0], co[1], co[2], 1.0)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    mat = make_mat('mat_' + name, color, roughness=0.3, emission=emission, alpha=alpha)
    obj.data.materials.append(mat)
    return obj


def add_dashed_line(name, start, end, segments=10, color=(0.85, 0.9, 1.0), bevel=0.008, emission=0.6):
    parent = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(parent)
    parent.empty_display_type = 'PLAIN_AXES'
    s = Vector(start)
    e = Vector(end)
    direction = e - s
    total = direction.length
    if total < 1e-5:
        return parent
    direction.normalize()
    seg_len = total / (segments * 2 - 1)
    for i in range(segments):
        a = s + direction * (seg_len * (2 * i))
        b = s + direction * (seg_len * (2 * i + 1))
        seg = add_curve_polyline(f'{name}_seg{i:02d}', [a, b], color=color, bevel=bevel, emission=emission)
        seg.parent = parent
    return parent


def add_text(name, location, body, size=0.18, color=(1.0, 0.97, 0.85), emission=0.5, rotation=None):
    bpy.ops.object.text_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.body = str(body)
    obj.data.align_x = 'CENTER'
    obj.data.align_y = 'CENTER'
    obj.data.size = size
    if rotation is None:
        obj.rotation_euler = (math.radians(78), 0, 0)
    else:
        obj.rotation_euler = rotation
    mat = make_mat('mat_' + name, color, roughness=0.4, emission=emission)
    obj.data.materials.append(mat)
    return obj


def set_visibility_windows(obj, ranges, total_end):
    if not ranges:
        return
    targets = [obj]
    def collect(o):
        for c in o.children:
            targets.append(c)
            collect(c)
    collect(obj)
    merged = []
    for s, e in sorted(ranges):
        if not merged or s > merged[-1][1] + 1:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    for t in targets:
        t.hide_render = True
        t.hide_viewport = True
        t.keyframe_insert('hide_render', frame=1)
        t.keyframe_insert('hide_viewport', frame=1)
        for s, e in merged:
            t.hide_render = False
            t.hide_viewport = False
            t.keyframe_insert('hide_render', frame=s)
            t.keyframe_insert('hide_viewport', frame=s)
            t.hide_render = True
            t.hide_viewport = True
            t.keyframe_insert('hide_render', frame=e + 1)
            t.keyframe_insert('hide_viewport', frame=e + 1)


# ------------------------------------------------------------------
# Scene init
# ------------------------------------------------------------------
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
    scene.eevee.gtao_factor = 0.6

scene.render.resolution_x = STORYBOARD['resolution'][0]
scene.render.resolution_y = STORYBOARD['resolution'][1]
scene.render.fps = FPS
scene.frame_start = 1
total_duration = sum(s['duration_sec'] for s in STORYBOARD['shots'])
scene.frame_end = round(total_duration * FPS)

if scene.world:
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs[0].default_value = (0.025, 0.028, 0.035, 1.0)
        bg.inputs[1].default_value = 1.0

scene.view_settings.view_transform = 'Filmic'
scene.view_settings.look = 'Medium Contrast'
scene.view_settings.exposure = 0.2

out_dir = os.environ['CG_TUTOR_OUT_DIR']
Path(out_dir).mkdir(parents=True, exist_ok=True)
_render_path = os.path.join(out_dir, 'frame_####.png')
if not _render_path.startswith('\\\\'):
    _render_path = _render_path.replace('\\', '/')
scene.render.filepath = _render_path

# ------------------------------------------------------------------
# Optical bench
# ------------------------------------------------------------------
bpy.ops.mesh.primitive_plane_add(size=14.0, location=(0, 0, 0))
bench = bpy.context.object
bench.name = 'optical_bench'
bench.data.materials.append(make_mat('mat_bench', (0.05, 0.06, 0.075), roughness=0.85))

bench_ticks = []
for i, x in enumerate([-3, -2, -1, 0, 1, 2, 3]):
    t = add_curve_polyline(f'bench_tick_x_{i}', [(x, -2.5, 0.005), (x, 2.5, 0.005)],
                           color=(0.18, 0.22, 0.28), bevel=0.004, emission=0.05)
    bench_ticks.append(t)
for i, y in enumerate([-2, -1, 0, 1, 2]):
    t = add_curve_polyline(f'bench_tick_y_{i}', [(-3.2, y, 0.005), (3.2, y, 0.005)],
                           color=(0.18, 0.22, 0.28), bevel=0.004, emission=0.05)
    bench_ticks.append(t)

# Lights
key_data = bpy.data.lights.new('key_light', type='AREA')
key_data.energy = 650
key_data.size = 5.0
key_data.color = (1.0, 0.96, 0.9)
key_light = bpy.data.objects.new('key_light', key_data)
bpy.context.collection.objects.link(key_light)
key_light.location = (0.0, -3.5, 5.0)
look_at(key_light, (0, 0, 1.0))

fill_data = bpy.data.lights.new('fill_light', type='AREA')
fill_data.energy = 220
fill_data.size = 6.0
fill_data.color = (0.7, 0.85, 1.0)
fill_light = bpy.data.objects.new('fill_light', fill_data)
bpy.context.collection.objects.link(fill_light)
fill_light.location = (-3.0, -2.0, 4.0)
look_at(fill_light, (0, 0, 1.0))

# ------------------------------------------------------------------
# Glass prism (transparent triangular cross-section)
# ------------------------------------------------------------------
mesh = bpy.data.meshes.new('glass_prism_mesh')
prism_obj = bpy.data.objects.new('glass_prism', mesh)
bpy.context.collection.objects.link(prism_obj)

bm = bmesh.new()
height = 1.1
half_base = 0.7
depth = 0.55
z_base = 0.5
z_top = z_base + height
front = [
    bm.verts.new((-half_base, -depth, z_base)),
    bm.verts.new(( half_base, -depth, z_base)),
    bm.verts.new(( 0.0,       -depth, z_top)),
]
back = [
    bm.verts.new((-half_base,  depth, z_base)),
    bm.verts.new(( half_base,  depth, z_base)),
    bm.verts.new(( 0.0,        depth, z_top)),
]
bm.faces.new(front)
bm.faces.new(list(reversed(back)))
bm.faces.new([front[0], front[1], back[1], back[0]])
bm.faces.new([front[1], front[2], back[2], back[1]])
bm.faces.new([front[2], front[0], back[0], back[2]])
bm.normal_update()
bm.to_mesh(mesh)
bm.free()

glass_mat = bpy.data.materials.new('mat_glass_prism')
glass_mat.use_nodes = True
bsdf = glass_mat.node_tree.nodes['Principled BSDF']
bsdf.inputs['Base Color'].default_value = (0.72, 0.92, 1.0, 1.0)
bsdf.inputs['Roughness'].default_value = 0.05
try:
    bsdf.inputs['Alpha'].default_value = 0.28
except KeyError:
    pass
try:
    bsdf.inputs['IOR'].default_value = 1.5
except KeyError:
    pass
try:
    bsdf.inputs['Emission Color'].default_value = (0.45, 0.75, 0.95, 1.0)
    bsdf.inputs['Emission Strength'].default_value = 0.18
except KeyError:
    pass
glass_mat.blend_method = 'BLEND'
glass_mat.show_transparent_back = False
prism_obj.data.materials.append(glass_mat)

# Bright beveled edges
edge_color = (0.7, 0.95, 1.0)
edge_pts_front = [
    [(-half_base, -depth, z_base), (half_base, -depth, z_base)],
    [(half_base, -depth, z_base), (0.0, -depth, z_top)],
    [(0.0, -depth, z_top), (-half_base, -depth, z_base)],
]
edge_pts_back = [
    [(-half_base, depth, z_base), (half_base, depth, z_base)],
    [(half_base, depth, z_base), (0.0, depth, z_top)],
    [(0.0, depth, z_top), (-half_base, depth, z_base)],
]
prism_edges = []
for i, e in enumerate(edge_pts_front):
    ed = add_curve_polyline(f'prism_edge_f_{i}', e, color=edge_color, bevel=0.013, emission=2.0)
    prism_edges.append(ed)
for i, e in enumerate(edge_pts_back):
    ed = add_curve_polyline(f'prism_edge_b_{i}', e, color=edge_color, bevel=0.013, emission=2.0)
    prism_edges.append(ed)

# ------------------------------------------------------------------
# Geometry: face midpoints and normals
# ------------------------------------------------------------------
# Left slope face: between (-half_base, z_base) and (0, z_top), at y=0 mid plane
entry_pt = Vector((-half_base / 2.0, 0.0, (z_base + z_top) / 2.0))
exit_pt  = Vector(( half_base / 2.0, 0.0, (z_base + z_top) / 2.0))

left_dir = Vector((0.0 - (-half_base), 0.0, z_top - z_base)).normalized()
left_normal = Vector((-left_dir.z, 0.0, left_dir.x)).normalized()
if left_normal.x > 0:
    left_normal = -left_normal

right_dir = Vector((0.0 - half_base, 0.0, z_top - z_base)).normalized()
right_normal = Vector((-right_dir.z, 0.0, right_dir.x)).normalized()
if right_normal.x < 0:
    right_normal = -right_normal

# ------------------------------------------------------------------
# Rays
# ------------------------------------------------------------------
# Incident white ray from upper-left
incident_start = Vector((-3.0, 0.0, entry_pt.z + 0.6))
incident_white_ray = add_curve_polyline(
    'incident_white_ray',
    [incident_start, entry_pt],
    color=(1.0, 1.0, 1.0), bevel=0.014, emission=2.8,
)

# Internal ray bends downward toward the entry normal (toward normal = away from incoming side)
# Original incident dir
incident_dir = (entry_pt - incident_start).normalized()
# Internal ray endpoint = exit_pt, but bent (lower than direct)
# The "bend toward normal" means smaller angle with normal; create internal_mid below straight line to exit
internal_target = exit_pt + Vector((0, 0, -0.18))  # slightly downward bend visible
internal_ray = add_curve_polyline(
    'internal_ray',
    [entry_pt, internal_target],
    color=(0.92, 0.96, 1.0), bevel=0.012, emission=2.2,
)

# Surface normals as dashed lines
surface_normal_entry = add_dashed_line(
    'surface_normal_entry',
    entry_pt - left_normal * 0.95,
    entry_pt + left_normal * 0.95,
    segments=10, color=(0.85, 0.92, 1.0), bevel=0.008, emission=1.0,
)
surface_normal_exit = add_dashed_line(
    'surface_normal_exit',
    exit_pt - right_normal * 0.95,
    exit_pt + right_normal * 0.95,
    segments=10, color=(0.85, 0.92, 1.0), bevel=0.008, emission=1.0,
)

# ------------------------------------------------------------------
# Projection screen
# ------------------------------------------------------------------
bpy.ops.mesh.primitive_plane_add(size=1.6, location=(2.6, 0.0, 1.1))
projection_screen = bpy.context.object
projection_screen.name = 'projection_screen'
projection_screen.rotation_euler = (math.radians(90), 0, math.radians(90))
projection_screen.data.materials.append(make_mat('mat_screen', (0.92, 0.92, 0.95), roughness=0.7, emission=0.05))

screen_frame_pts = [
    (2.6, -0.8, 0.3), (2.6, 0.8, 0.3),
    (2.6, 0.8, 1.9), (2.6, -0.8, 1.9), (2.6, -0.8, 0.3)
]
screen_frame = add_curve_polyline('screen_frame', screen_frame_pts, color=(0.5, 0.6, 0.8), bevel=0.012, emission=0.5)

# ------------------------------------------------------------------
# Spectrum exit rays
# ------------------------------------------------------------------
screen_x = 2.6
# Exit point of internal ray - they all start near exit_pt but slightly fanned
# Red bends LEAST -> highest exit (closest to extending internal direction with little extra bend)
# Blue bends MOST -> lowest exit on screen
red_end = Vector((screen_x, 0.0, 1.32))
green_end = Vector((screen_x, 0.0, 1.10))
blue_end = Vector((screen_x, 0.0, 0.82))

spectrum_red_ray = add_curve_polyline(
    'spectrum_red_ray', [internal_target, red_end],
    color=(1.0, 0.22, 0.18), bevel=0.012, emission=2.8,
)
spectrum_green_ray = add_curve_polyline(
    'spectrum_green_ray', [internal_target, green_end],
    color=(0.22, 1.0, 0.32), bevel=0.012, emission=2.8,
)
spectrum_blue_ray = add_curve_polyline(
    'spectrum_blue_ray', [internal_target, blue_end],
    color=(0.28, 0.45, 1.0), bevel=0.012, emission=2.8,
)

def make_screen_dot(name, location, color):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.07, location=location, segments=20, ring_count=10)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (0.35, 1.0, 1.0)
    obj.data.materials.append(make_mat('mat_' + name, color, roughness=0.3, emission=5.0))
    return obj

red_screen_dot = make_screen_dot('red_screen_dot', red_end, (1.0, 0.22, 0.18))
green_screen_dot = make_screen_dot('green_screen_dot', green_end, (0.22, 1.0, 0.32))
blue_screen_dot = make_screen_dot('blue_screen_dot', blue_end, (0.28, 0.45, 1.0))

# ------------------------------------------------------------------
# Angle arcs at entry face
# ------------------------------------------------------------------
def add_arc(name, center, normal_dir, ray_dir, radius=0.22, segments=18, color=(1, 0.9, 0.5), emission=1.0):
    n = normal_dir.normalized()
    r = ray_dir.normalized()
    angle = math.acos(max(-1.0, min(1.0, n.dot(r))))
    axis = n.cross(r)
    if axis.length < 1e-5:
        axis = Vector((0, 1, 0))
    axis.normalize()
    pts = []
    for i in range(segments + 1):
        t = i / segments
        q = Quaternion(axis, angle * t)
        v = q @ n
        pts.append(center + v * radius)
    return add_curve_polyline(name, pts, color=color, bevel=0.005, emission=emission)

arc_theta_i = add_arc('arc_theta_i', entry_pt, left_normal, -incident_dir, radius=0.30,
                      color=(1.0, 0.88, 0.4), emission=1.2)
internal_dir_vec = (internal_target - entry_pt).normalized()
arc_theta_t = add_arc('arc_theta_t', entry_pt, -left_normal, internal_dir_vec, radius=0.28,
                      color=(1.0, 0.88, 0.4), emission=1.2)

# ------------------------------------------------------------------
# Labels (NO formulas - those are added by overlay stage)
# ------------------------------------------------------------------
# Shot 1 labels
label_white_light = add_text('label_white_light',
                             (-2.4, 0.5, entry_pt.z + 0.85),
                             'white light', size=0.14, color=(1, 1, 1), emission=0.8)
label_bench_shot1 = add_text('label_bench',
                             (1.3, 0.5, 0.12),
                             'optical bench', size=0.12, color=(0.7, 0.8, 0.95), emission=0.4)

# Shot 2 labels (Snell scene)
label_incident_ray = add_text('label_incident_ray',
                              (-2.0, 0.6, entry_pt.z + 0.55),
                              'incident ray', size=0.13, color=(1, 1, 1), emission=0.7)
label_normal_entry = add_text('label_normal_entry',
                              entry_pt + left_normal * 1.05 + Vector((0, 0.5, 0.05)),
                              'normal', size=0.12, color=(0.85, 0.95, 1.0), emission=0.6)
label_theta_i = add_text('label_theta_i',
                         entry_pt - incident_dir * 0.40 + Vector((0, 0.5, 0.0)),
                         'theta_i', size=0.11, color=(1.0, 0.92, 0.55), emission=0.7)
label_theta_t = add_text('label_theta_t',
                         entry_pt + internal_dir_vec * 0.40 + Vector((0, 0.5, -0.05)),
                         'theta_t', size=0.11, color=(1.0, 0.92, 0.55), emission=0.7)
label_internal_ray = add_text('label_internal_ray',
                              ((entry_pt + internal_target) * 0.5) + Vector((0, 0.55, -0.25)),
                              'refracted', size=0.11, color=(0.9, 0.95, 1.0), emission=0.5)

# Shot 3 labels (RGB) - short colored words near each ray
label_red = add_text('label_red',
                     (internal_target * 0.5 + red_end * 0.5) + Vector((0, 0.5, 0.10)),
                     'red', size=0.12, color=(1.0, 0.4, 0.35), emission=0.9)
label_green = add_text('label_green',
                       (internal_target * 0.5 + green_end * 0.5) + Vector((0, 0.5, 0.0)),
                       'green', size=0.12, color=(0.4, 1.0, 0.5), emission=0.9)
label_blue = add_text('label_blue',
                      (internal_target * 0.5 + blue_end * 0.5) + Vector((0, 0.5, -0.10)),
                      'blue', size=0.12, color=(0.45, 0.6, 1.0), emission=0.9)
label_normal_exit = add_text('label_normal_exit',
                             exit_pt + right_normal * 1.05 + Vector((0, 0.5, 0.05)),
                             'normal', size=0.12, color=(0.85, 0.95, 1.0), emission=0.6)

# Shot 4 summary labels
label_same_incident = add_text('label_same_incident',
                               (-2.0, 0.6, entry_pt.z + 0.7),
                               'same incident ray', size=0.13, color=(1, 1, 1), emission=0.8)
label_diff_index = add_text('label_diff_index',
                            (1.0, 0.6, 1.85),
                            'different refractive index', size=0.13, color=(1.0, 0.92, 0.7), emission=0.8)
label_screen = add_text('label_screen',
                        (2.6, -0.95, 1.95),
                        'screen', size=0.13, color=(0.9, 0.95, 1.0), emission=0.7)

# ------------------------------------------------------------------
# Cameras and shot timing
# ------------------------------------------------------------------
shots = STORYBOARD['shots']
shot_ranges = {}
cursor = 0
cameras = {}
for idx, shot in enumerate(shots):
    duration_frames = round(shot['duration_sec'] * FPS)
    start = cursor + 1
    end = cursor + duration_frames
    shot_ranges[shot['node_id']] = (start, end)

    cam_data = bpy.data.cameras.new('cam_' + shot['node_id'])
    cam = bpy.data.objects.new('cam_' + shot['node_id'], cam_data)
    bpy.context.collection.objects.link(cam)
    cam_data.sensor_width = 36.0
    first = shot['camera'][0]
    cam.location = tuple(first['position'])
    look_at(cam, first['look_at'])
    cam_data.lens = lens_for_fov(first['fov'])

    for key in shot['camera']:
        f = frame_for_time(key['time_sec'])
        cam.location = tuple(key['position'])
        look_at(cam, key['look_at'])
        cam_data.lens = lens_for_fov(key['fov'])
        cam.keyframe_insert('location', frame=f)
        cam.keyframe_insert('rotation_euler', frame=f)
        cam_data.keyframe_insert('lens', frame=f)

    marker = scene.timeline_markers.new('shot_' + shot['node_id'], frame=start)
    marker.camera = cam
    if idx == 0:
        scene.camera = cam
    cameras[shot['node_id']] = cam
    cursor = end

total_end = scene.frame_end

# ------------------------------------------------------------------
# Visibility windows
# ------------------------------------------------------------------
r1 = shot_ranges['node_01']
r2 = shot_ranges['node_02']
r3 = shot_ranges['node_03']
r4 = shot_ranges['node_04']
all_range = (r1[0], r4[1])

# Always on across all shots
always_on = [bench, key_light, fill_light, prism_obj, projection_screen, screen_frame,
             incident_white_ray, internal_ray,
             surface_normal_entry, surface_normal_exit,
             spectrum_red_ray, spectrum_green_ray, spectrum_blue_ray,
             red_screen_dot, green_screen_dot, blue_screen_dot]
always_on.extend(bench_ticks)
always_on.extend(prism_edges)

for obj in always_on:
    set_visibility_windows(obj, [all_range], total_end)

# Shot 1
set_visibility_windows(label_white_light, [r1, r4], total_end)
set_visibility_windows(label_bench_shot1, [r1], total_end)

# Shot 2
for o in [label_incident_ray, label_normal_entry, label_theta_i, label_theta_t,
          label_internal_ray, arc_theta_i, arc_theta_t]:
    set_visibility_windows(o, [r2], total_end)

# Shot 3
for o in [label_red, label_green, label_blue, label_normal_exit, arc_theta_i, arc_theta_t]:
    if o == arc_theta_i or o == arc_theta_t:
        # already keyed for r2; extend to r3 by re-keying
        set_visibility_windows(o, [r2, r3], total_end)
    else:
        set_visibility_windows(o, [r3], total_end)

# Shot 4
for o in [label_same_incident, label_diff_index, label_screen]:
    set_visibility_windows(o, [r4], total_end)

# ------------------------------------------------------------------
# Animate ray reveal (animation_coverage requirement)
# ------------------------------------------------------------------
def animate_bevel_reveal(curve_obj, ranges):
    for s, e in ranges:
        curve_obj.data.bevel_factor_end = 0.05
        curve_obj.data.keyframe_insert('bevel_factor_end', frame=s)
        curve_obj.data.bevel_factor_end = 1.0
        end_f = min(e, s + max(8, (e - s) // 2))
        curve_obj.data.keyframe_insert('bevel_factor_end', frame=end_f)

animate_bevel_reveal(incident_white_ray, [r1, r2, r3, r4])
animate_bevel_reveal(internal_ray, [r2, r3, r4])
animate_bevel_reveal(spectrum_red_ray, [r3, r4])
animate_bevel_reveal(spectrum_green_ray, [r3, r4])
animate_bevel_reveal(spectrum_blue_ray, [r3, r4])

# ------------------------------------------------------------------
# Render
# ------------------------------------------------------------------
preview_raw = os.environ.get('CG_TUTOR_PREVIEW_FRAMES', '').strip()
if preview_raw:
    for raw in preview_raw.split(','):
        raw = raw.strip()
        if not raw:
            continue
        f = int(raw)
        scene.frame_set(f)
        p = os.path.join(out_dir, f'frame_{f:04d}.png')
        if not p.startswith('\\\\'):
            p = p.replace('\\', '/')
        scene.render.filepath = p
        bpy.ops.render.render(write_still=True)
else:
    bpy.ops.render.render(animation=True)