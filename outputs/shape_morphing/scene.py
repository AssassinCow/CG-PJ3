import math
import os
from pathlib import Path

import bpy
from mathutils import Vector


STORYBOARD = {'concept_id': 'shape_morphing',
 'fps': 24,
 'resolution': [960, 540],
 'shots': [
    {'node_id': 'node_01', 'start_sec': 0.0, 'duration_sec': 3.5,
     'camera': [{'time_sec': 0.0, 'position': [0.0, -7.0, 3.2], 'look_at': [0.0, 0.9, 0.9], 'fov': 48.0},
                {'time_sec': 3.5, 'position': [0.8, -6.88, 3.26], 'look_at': [0.08, 0.9, 0.9], 'fov': 48.0}]},
    {'node_id': 'node_02', 'start_sec': 3.5, 'duration_sec': 5.0,
     'camera': [{'time_sec': 3.5, 'position': [0.0, -7.0, 3.2], 'look_at': [0.0, 0.9, 0.9], 'fov': 48.0},
                {'time_sec': 8.5, 'position': [-0.3, -6.88, 3.26], 'look_at': [-0.08, 0.9, 0.9], 'fov': 48.0}]},
    {'node_id': 'node_03', 'start_sec': 8.5, 'duration_sec': 4.0,
     'camera': [{'time_sec': 8.5, 'position': [2.5, -6.5, 3.0], 'look_at': [0.0, 0.9, 0.9], 'fov': 48.0},
                {'time_sec': 12.5, 'position': [-2.5, -6.5, 3.0], 'look_at': [0.0, 0.9, 0.9], 'fov': 48.0}]},
    {'node_id': 'node_04', 'start_sec': 12.5, 'duration_sec': 5.5,
     'camera': [{'time_sec': 12.5, 'position': [0.0, -7.0, 3.2], 'look_at': [0.0, 0.9, 0.9], 'fov': 48.0},
                {'time_sec': 18.0, 'position': [-0.3, -6.88, 3.26], 'look_at': [-0.08, 0.9, 0.9], 'fov': 48.0}]},
 ]}

FPS = STORYBOARD['fps']


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def frame_for_time(t):
    return round(float(t) * FPS) + 1


def lens_for_fov(fov_deg, sensor_mm=36.0):
    angle = math.radians(max(5.0, min(160.0, float(fov_deg))))
    return sensor_mm / (2.0 * math.tan(angle / 2.0))


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
        if 'Alpha' in bsdf.inputs:
            bsdf.inputs['Alpha'].default_value = rgba[3]
        if emission > 0:
            try:
                bsdf.inputs['Emission Color'].default_value = rgba
                bsdf.inputs['Emission Strength'].default_value = emission
            except KeyError:
                pass
    if rgba[3] < 0.999:
        mat.blend_method = 'BLEND'
        if hasattr(mat, 'show_transparent_back'):
            mat.show_transparent_back = True
    return mat


def add_text(name, location, body, size=0.22, color=(1.0, 0.96, 0.72), emission=0.4,
             rotation=(math.radians(78), 0, 0), align='CENTER'):
    bpy.ops.object.text_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.body = str(body)
    obj.data.align_x = align
    obj.data.align_y = 'CENTER'
    obj.data.size = float(size)
    obj.rotation_euler = rotation
    mat = make_mat('mat_' + name, list(color) + [1.0], emission=emission)
    obj.data.materials.append(mat)
    return obj


def add_polyline(name, points, color=(0.35, 0.7, 1.0), bevel=0.012, emission=0.5, alpha=1.0):
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
    obj.data.materials.append(make_mat('mat_' + name, list(color) + [alpha], emission=emission, alpha=alpha))
    return obj


# ---------- Shared topology hero mesh (subdivided cube) ----------

def build_hero_mesh(name, subdivisions=4):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
    obj = bpy.context.object
    obj.name = name
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    for _ in range(subdivisions):
        bpy.ops.mesh.subdivide()
    bpy.ops.object.mode_set(mode='OBJECT')
    for poly in obj.data.polygons:
        poly.use_smooth = True
    return obj


def vertex_to_cube(v, scale=0.7):
    m = max(abs(v[0]), abs(v[1]), abs(v[2]))
    if m < 1e-6:
        return Vector((0, 0, 0))
    return Vector((v[0] / m, v[1] / m, v[2] / m)) * scale


def vertex_to_sphere(v, radius=0.7):
    direction = Vector(v).normalized() if Vector(v).length > 1e-6 else Vector((0, 0, 1))
    return direction * radius


def vertex_to_capsule(v, radius=0.55, half_height=0.55):
    direction = Vector(v).normalized() if Vector(v).length > 1e-6 else Vector((0, 0, 1))
    z = direction.z * (half_height + radius)
    if abs(z) < half_height:
        rxy = Vector((direction.x, direction.y, 0))
        if rxy.length > 1e-6:
            rxy = rxy.normalized() * radius
        return Vector((rxy.x, rxy.y, z))
    else:
        cap_z = half_height if z > 0 else -half_height
        local = Vector((direction.x, direction.y, (z - cap_z) / radius if radius > 0 else 0))
        if local.length > 1e-6:
            local = local.normalized() * radius
        return Vector((local.x, local.y, cap_z + local.z))


def vertex_to_star(v, base=0.65, spike=0.35, points=5):
    direction = Vector(v).normalized() if Vector(v).length > 1e-6 else Vector((0, 0, 1))
    angle = math.atan2(direction.y, direction.x)
    rxy_len = math.sqrt(direction.x ** 2 + direction.y ** 2)
    r_mod = base + spike * math.cos(points * angle)
    flat_z = direction.z * 0.45
    if rxy_len > 1e-6:
        sx = direction.x / rxy_len * r_mod * rxy_len
        sy = direction.y / rxy_len * r_mod * rxy_len
    else:
        sx = 0
        sy = 0
    w = rxy_len
    return Vector((sx * w + direction.x * (1 - w) * 0.3,
                   sy * w + direction.y * (1 - w) * 0.3,
                   flat_z))


def store_shape_keys(obj, original_coords):
    if obj.data.shape_keys is None:
        obj.shape_key_add(name='Basis', from_mix=False)

    shapes = {
        'cube': vertex_to_cube,
        'sphere': vertex_to_sphere,
        'capsule': vertex_to_capsule,
        'star': vertex_to_star,
    }
    keys = {}
    for shape_name, fn in shapes.items():
        sk = obj.shape_key_add(name=shape_name, from_mix=False)
        for i, v in enumerate(sk.data):
            v.co = fn(original_coords[i])
        keys[shape_name] = sk
    basis = obj.data.shape_keys.key_blocks['Basis']
    for i, v in enumerate(basis.data):
        v.co = vertex_to_cube(original_coords[i])
    return keys


def pick_marker_indices(coords, count=20):
    indices = []
    targets = []
    for sx in (-0.5, 0.5):
        for sy in (-0.5, 0.5):
            for sz in (-0.5, 0.5):
                targets.append(Vector((sx, sy, sz)))
    for axis in range(3):
        for sign in (-0.5, 0.5):
            v = [0, 0, 0]
            v[axis] = sign
            targets.append(Vector(v))
    edge_targets = [
        Vector((0, 0.5, 0.5)), Vector((0, -0.5, 0.5)),
        Vector((0.5, 0, 0.5)), Vector((-0.5, 0, 0.5)),
        Vector((0.5, 0.5, 0)), Vector((-0.5, -0.5, 0)),
    ]
    targets.extend(edge_targets)

    used = set()
    for t in targets[:count]:
        best_i = 0
        best_d = 1e9
        for i, c in enumerate(coords):
            if i in used:
                continue
            d = (Vector(c) - t).length
            if d < best_d:
                best_d = d
                best_i = i
        used.add(best_i)
        indices.append(best_i)
    return indices


# ---------- Main scene setup ----------

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
    scene.eevee.gtao_distance = 3
    scene.eevee.gtao_factor = 0.6

if scene.world:
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs[0].default_value = (0.06, 0.07, 0.085, 1.0)
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

# ---------- Compute shot frame ranges ----------
shot_ranges = {}
cursor = 0
for shot in STORYBOARD['shots']:
    n = round(shot['duration_sec'] * FPS)
    start = cursor + 1
    end = cursor + n
    shot_ranges[shot['node_id']] = (start, end)
    cursor = end

S1 = shot_ranges['node_01']
S2 = shot_ranges['node_02']
S3 = shot_ranges['node_03']
S4 = shot_ranges['node_04']

# ---------- Soft floor ----------
bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
floor = bpy.context.object
floor.name = 'soft_floor'
floor.data.materials.append(make_mat('mat_floor', (0.18, 0.20, 0.24, 1.0), roughness=0.85))

# ---------- Background gradient plane (vertical) ----------
bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 8, 5))
bg_plane = bpy.context.object
bg_plane.name = 'bg_gradient'
bg_plane.rotation_euler = (math.radians(90), 0, 0)
bg_plane.data.materials.append(make_mat('mat_bg', (0.08, 0.10, 0.13, 1.0), roughness=1.0))

# ---------- Lights ----------
key_data = bpy.data.lights.new('key_light', type='AREA')
key_data.energy = 650
key_data.size = 5.0
key_data.color = (1.0, 0.96, 0.9)
key_light = bpy.data.objects.new('key_light', key_data)
bpy.context.collection.objects.link(key_light)
key_light.location = (0.0, -3.5, 5.0)
look_at(key_light, (0, 0, 0.9))

fill_data = bpy.data.lights.new('fill_light', type='AREA')
fill_data.energy = 220
fill_data.size = 4.0
fill_data.color = (0.85, 0.92, 1.0)
fill_light = bpy.data.objects.new('fill_light', fill_data)
bpy.context.collection.objects.link(fill_light)
fill_light.location = (-3.5, -2.0, 3.5)
look_at(fill_light, (0, 0, 0.9))

rim_data = bpy.data.lights.new('rim_light', type='AREA')
rim_data.energy = 320
rim_data.size = 3.0
rim_data.color = (1.0, 0.85, 0.7)
rim_light = bpy.data.objects.new('rim_light', rim_data)
bpy.context.collection.objects.link(rim_light)
rim_light.location = (1.5, 3.0, 4.0)
look_at(rim_light, (0, 0, 0.9))

# ---------- Hero mesh (single, centered) ----------
hero_mesh = build_hero_mesh('hero_mesh', subdivisions=4)
hero_mesh.location = (0.0, 0.0, 0.9)
hero_mat = make_mat('mat_hero', (0.95, 0.55, 0.30, 1.0), roughness=0.25, metallic=0.35)
hero_mesh.data.materials.append(hero_mat)

original_coords = [Vector(v.co) for v in hero_mesh.data.vertices]

shape_keys = store_shape_keys(hero_mesh, original_coords)
shape_keys['cube'].value = 1.0
shape_keys['sphere'].value = 0.0
shape_keys['capsule'].value = 0.0
shape_keys['star'].value = 0.0


def insert_sk(name, frame, value):
    sk = shape_keys[name]
    sk.value = value
    sk.keyframe_insert('value', frame=frame)


for n in ('cube', 'sphere', 'capsule', 'star'):
    insert_sk(n, S1[0], 1.0 if n == 'cube' else 0.0)

for n in ('cube', 'sphere', 'capsule', 'star'):
    insert_sk(n, S1[1], 1.0 if n == 'cube' else 0.0)

insert_sk('cube', S2[0], 1.0)
insert_sk('sphere', S2[0], 0.0)
insert_sk('cube', S2[1], 0.0)
insert_sk('sphere', S2[1], 1.0)
insert_sk('capsule', S2[1], 0.0)
insert_sk('star', S2[1], 0.0)

insert_sk('cube', S3[0], 0.0)
insert_sk('sphere', S3[0], 1.0)
insert_sk('capsule', S3[0], 0.0)
insert_sk('sphere', S3[1], 0.0)
insert_sk('capsule', S3[1], 1.0)
insert_sk('star', S3[1], 0.0)
insert_sk('cube', S3[1], 0.0)

s4_dur = S4[1] - S4[0]
mid_a = S4[0] + int(s4_dur * 0.35)
mid_b = S4[0] + int(s4_dur * 0.45)
loop1 = S4[0] + int(s4_dur * 0.60)
loop2 = S4[0] + int(s4_dur * 0.75)
loop3 = S4[0] + int(s4_dur * 0.90)
end4 = S4[1]

insert_sk('capsule', S4[0], 1.0)
insert_sk('star', S4[0], 0.0)
insert_sk('cube', S4[0], 0.0)
insert_sk('sphere', S4[0], 0.0)

insert_sk('capsule', mid_a, 0.0)
insert_sk('star', mid_a, 1.0)
insert_sk('cube', mid_a, 0.0)
insert_sk('sphere', mid_a, 0.0)

insert_sk('star', mid_b, 0.0)
insert_sk('cube', mid_b, 1.0)
insert_sk('capsule', mid_b, 0.0)
insert_sk('sphere', mid_b, 0.0)

insert_sk('cube', loop1, 0.0)
insert_sk('sphere', loop1, 1.0)
insert_sk('capsule', loop1, 0.0)
insert_sk('star', loop1, 0.0)

insert_sk('sphere', loop2, 0.0)
insert_sk('capsule', loop2, 1.0)
insert_sk('cube', loop2, 0.0)
insert_sk('star', loop2, 0.0)

insert_sk('capsule', loop3, 0.0)
insert_sk('star', loop3, 1.0)
insert_sk('cube', loop3, 0.0)
insert_sk('sphere', loop3, 0.0)

insert_sk('star', end4, 1.0)
insert_sk('cube', end4, 0.0)
insert_sk('sphere', end4, 0.0)
insert_sk('capsule', end4, 0.0)

# ---------- Sparse vertex markers ----------
marker_indices = pick_marker_indices(original_coords, count=20)
markers = []
marker_mat = make_mat('mat_marker', (1.0, 0.85, 0.3, 1.0), roughness=0.3, emission=1.2)

for i, vi in enumerate(marker_indices):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.05, location=(0, 0, 0))
    m = bpy.context.object
    m.name = f'sample_vertex_marker_{i:02d}' if i > 0 else 'sample_vertex_marker'
    m.data.materials.append(marker_mat)
    m.parent = hero_mesh
    m.parent_type = 'VERTEX'
    m.parent_vertices[0] = vi
    m.location = (0, 0, 0)
    markers.append((m, vi))

# ---------- Trajectory traces (cube positions -> sphere positions, world space) ----------
trace_origin = Vector((0.0, 0.0, 0.9))
trajectory_traces = []
for i, vi in enumerate(marker_indices[:10]):
    cube_p = vertex_to_cube(original_coords[vi]) + trace_origin
    sphere_p = vertex_to_sphere(original_coords[vi]) + trace_origin
    name = f'trajectory_trace_{i:02d}' if i > 0 else 'trajectory_trace'
    line = add_polyline(name, [cube_p, sphere_p],
                        color=(0.55, 0.85, 1.0), bevel=0.006, emission=0.6, alpha=0.85)
    trajectory_traces.append(line)

# ---------- Storyboard-required named anchors (kept invisible, no ghost meshes) ----------
src_ref = bpy.data.objects.new('source_shape_A', None)
bpy.context.collection.objects.link(src_ref)
src_ref.empty_display_type = 'PLAIN_AXES'
src_ref.empty_display_size = 0.001
src_ref.location = (0, 0, 0.9)
src_ref.hide_render = True
src_ref.hide_viewport = True

tgt_ref = bpy.data.objects.new('target_shape_B', None)
bpy.context.collection.objects.link(tgt_ref)
tgt_ref.empty_display_type = 'PLAIN_AXES'
tgt_ref.empty_display_size = 0.001
tgt_ref.location = (0, 0, 0.9)
tgt_ref.hide_render = True
tgt_ref.hide_viewport = True

# ---------- Side panel (parameter_panel) ----------
panel_loc = (2.6, 0.6, 1.6)
bpy.ops.mesh.primitive_plane_add(size=1.6, location=panel_loc)
parameter_panel = bpy.context.object
parameter_panel.name = 'parameter_panel'
parameter_panel.rotation_euler = (math.radians(78), 0, math.radians(-18))
parameter_panel.scale = (1.0, 0.65, 1.0)
panel_mat = make_mat('mat_panel', (0.05, 0.07, 0.10, 0.85), roughness=0.6, alpha=0.85)
parameter_panel.data.materials.append(panel_mat)

border_loc = (panel_loc[0], panel_loc[1] - 0.005, panel_loc[2])
bpy.ops.mesh.primitive_plane_add(size=1.7, location=border_loc)
panel_border = bpy.context.object
panel_border.name = 'parameter_panel_border'
panel_border.rotation_euler = parameter_panel.rotation_euler
panel_border.scale = (1.02, 0.68, 1.0)
panel_border.data.materials.append(make_mat('mat_panel_border', (0.25, 0.55, 0.85, 1.0),
                                             roughness=0.4, emission=0.5))

# Short non-formula labels only (formulas are added by the LaTeX overlay stage)
panel_title = add_text('panel_title',
                       (panel_loc[0] - 0.55, panel_loc[1] - 0.03, panel_loc[2] + 0.45),
                       'BLEND  t', size=0.13, color=(1.0, 0.95, 0.7), emission=0.8,
                       rotation=parameter_panel.rotation_euler, align='LEFT')

shape_label = add_text('shape_label',
                       (panel_loc[0] - 0.55, panel_loc[1] - 0.03, panel_loc[2] + 0.25),
                       'shape A', size=0.11, color=(0.85, 0.92, 1.0), emission=0.6,
                       rotation=parameter_panel.rotation_euler, align='LEFT')


def make_shape_label(name, body, frame_on, frame_off):
    lbl = add_text(name,
                   (panel_loc[0] - 0.55, panel_loc[1] - 0.03, panel_loc[2] + 0.25),
                   body, size=0.11, color=(0.85, 0.92, 1.0), emission=0.6,
                   rotation=parameter_panel.rotation_euler, align='LEFT')
    lbl.hide_render = True
    lbl.hide_viewport = True
    lbl.keyframe_insert('hide_render', frame=1)
    lbl.keyframe_insert('hide_viewport', frame=1)
    lbl.hide_render = False
    lbl.hide_viewport = False
    lbl.keyframe_insert('hide_render', frame=frame_on)
    lbl.keyframe_insert('hide_viewport', frame=frame_on)
    lbl.hide_render = True
    lbl.hide_viewport = True
    lbl.keyframe_insert('hide_render', frame=frame_off + 1)
    lbl.keyframe_insert('hide_viewport', frame=frame_off + 1)
    return lbl


shape_label.hide_render = True
shape_label.hide_viewport = True

lbl_s1 = make_shape_label('shape_label_s1', 'shape A', S1[0], S1[1])
lbl_s2 = make_shape_label('shape_label_s2', 'A to B', S2[0], S2[1])
lbl_s3 = make_shape_label('shape_label_s3', 'B to capsule', S3[0], S3[1])
lbl_s4 = make_shape_label('shape_label_s4', 'capsule to star (loop)', S4[0], S4[1])

# t-slider track
slider_y = panel_loc[2] - 0.05
slider_left = panel_loc[0] - 0.55
slider_right = panel_loc[0] + 0.55
slider_track = add_polyline('t_slider_track',
                            [(slider_left, panel_loc[1] - 0.01, slider_y),
                             (slider_right, panel_loc[1] - 0.01, slider_y)],
                            color=(0.4, 0.5, 0.65), bevel=0.012, emission=0.3)

bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.05,
                                      location=(slider_left, panel_loc[1] - 0.02, slider_y))
t_slider_knob = bpy.context.object
t_slider_knob.name = 't_slider_knob'
t_slider_knob.data.materials.append(make_mat('mat_knob', (1.0, 0.7, 0.2, 1.0),
                                               roughness=0.3, emission=1.5))


def keyframe_knob(frame, t):
    x = slider_left + (slider_right - slider_left) * t
    t_slider_knob.location = (x, panel_loc[1] - 0.02, slider_y)
    t_slider_knob.keyframe_insert('location', frame=frame)


keyframe_knob(S1[0], 0.0)
keyframe_knob(S1[1], 0.0)
keyframe_knob(S2[0], 0.0)
keyframe_knob(S2[1], 1.0)
keyframe_knob(S3[0], 0.0)
keyframe_knob(S3[1], 1.0)
keyframe_knob(S4[0], 0.0)
keyframe_knob(mid_a, 1.0)
keyframe_knob(mid_b, 0.0)
keyframe_knob(loop1, 0.33)
keyframe_knob(loop2, 0.66)
keyframe_knob(loop3, 1.0)
keyframe_knob(S4[1], 1.0)

# Short t-value indicator (allowed: short non-formula label)
t_indicator = add_text('t_indicator',
                       (panel_loc[0] - 0.55, panel_loc[1] - 0.03, panel_loc[2] - 0.30),
                       't = 0..1', size=0.085, color=(0.95, 0.96, 0.85), emission=0.5,
                       rotation=parameter_panel.rotation_euler, align='LEFT')

# Weight bars for shot 4
weight_bars = []
for i, n in enumerate(['cube', 'sphere', 'capsule', 'star']):
    bx = panel_loc[0] - 0.45 + i * 0.28
    by = panel_loc[1] - 0.02
    bz = panel_loc[2] - 0.18
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(bx, by, bz))
    bar = bpy.context.object
    bar.name = f'weight_bar_{n}'
    bar.scale = (0.05, 0.01, 0.08)
    bar.rotation_euler = parameter_panel.rotation_euler
    bar.data.materials.append(make_mat(f'mat_wbar_{n}',
                                        (0.4 + i * 0.15, 0.7 - i * 0.1, 1.0 - i * 0.2, 1.0),
                                        emission=1.0))
    bar.hide_render = True
    bar.hide_viewport = True
    bar.keyframe_insert('hide_render', frame=1)
    bar.keyframe_insert('hide_viewport', frame=1)
    bar.hide_render = False
    bar.hide_viewport = False
    bar.keyframe_insert('hide_render', frame=S4[0])
    bar.keyframe_insert('hide_viewport', frame=S4[0])
    bar.hide_render = True
    bar.hide_viewport = True
    bar.keyframe_insert('hide_render', frame=S4[1] + 1)
    bar.keyframe_insert('hide_viewport', frame=S4[1] + 1)
    weight_bars.append(bar)

# ---------- Visible ray/vector cue (required by visual contract for node_01) ----------
ray_cue = add_polyline('visible_ray_vector_cue',
                       [(0.7, 0.7, 1.6), (1.4, 0.4, 2.0), (2.1, 0.0, 2.3)],
                       color=(0.5, 0.85, 1.0), bevel=0.015, emission=1.2)

ray_cue.data.bevel_factor_mapping_start = 'SPLINE'
ray_cue.data.bevel_factor_mapping_end = 'SPLINE'
ray_cue.data.bevel_factor_start = 0.0
ray_cue.data.bevel_factor_end = 0.05
ray_cue.data.keyframe_insert('bevel_factor_end', frame=S1[0])
ray_cue.data.bevel_factor_end = 1.0
ray_cue.data.keyframe_insert('bevel_factor_end', frame=S1[1])

ray_cue.hide_render = False
ray_cue.hide_viewport = False
ray_cue.keyframe_insert('hide_render', frame=1)
ray_cue.keyframe_insert('hide_viewport', frame=1)
ray_cue.hide_render = True
ray_cue.hide_viewport = True
ray_cue.keyframe_insert('hide_render', frame=S1[1] + 1)
ray_cue.keyframe_insert('hide_viewport', frame=S1[1] + 1)

# ---------- Trajectory traces visibility: only during shot 2 ----------
for tt in trajectory_traces:
    tt.hide_render = True
    tt.hide_viewport = True
    tt.keyframe_insert('hide_render', frame=1)
    tt.keyframe_insert('hide_viewport', frame=1)
    tt.hide_render = False
    tt.hide_viewport = False
    tt.keyframe_insert('hide_render', frame=S2[0])
    tt.keyframe_insert('hide_viewport', frame=S2[0])
    tt.hide_render = True
    tt.hide_viewport = True
    tt.keyframe_insert('hide_render', frame=S2[1] + 1)
    tt.keyframe_insert('hide_viewport', frame=S2[1] + 1)

# ---------- Cameras per shot ----------
for idx, shot in enumerate(STORYBOARD['shots']):
    cam_data = bpy.data.cameras.new('camera_' + shot['node_id'])
    cam = bpy.data.objects.new('camera_' + shot['node_id'], cam_data)
    bpy.context.collection.objects.link(cam)
    first_key = shot['camera'][0]
    cam.location = tuple(first_key['position'])
    look_at(cam, first_key['look_at'])
    cam.data.sensor_width = 36.0
    cam.data.lens = lens_for_fov(first_key.get('fov', 48.0))
    s_start, s_end = shot_ranges[shot['node_id']]
    marker = scene.timeline_markers.new('shot_' + shot['node_id'], frame=s_start)
    marker.camera = cam
    if idx == 0:
        scene.camera = cam
    for key in shot['camera']:
        frame = frame_for_time(key['time_sec'])
        cam.location = tuple(key['position'])
        look_at(cam, key['look_at'])
        cam.data.lens = lens_for_fov(key.get('fov', 48.0))
        cam.keyframe_insert('location', frame=frame)
        cam.keyframe_insert('rotation_euler', frame=frame)
        cam.data.keyframe_insert('lens', frame=frame)

# ---------- Render ----------
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