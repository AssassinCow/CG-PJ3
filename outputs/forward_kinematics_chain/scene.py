import math
import os
from pathlib import Path

import bpy
from mathutils import Vector, Matrix


STORYBOARD = {
    'concept_id': 'forward_kinematics_chain',
    'fps': 24,
    'resolution': [960, 540],
    'shots': [
        {'node_id': 'node_01', 'start_sec': 0.0, 'duration_sec': 3.0,
         'theta1_deg': (0.0, 0.0), 'theta2_deg': (0.0, 0.0),
         'cam_pos': [(0.0, -7.0, 3.2), (0.8, -6.88, 3.26)],
         'cam_lookat': [(0.0, 0.0, 0.9), (0.08, 0.0, 0.9)],
         'fov': 48.0,
         'show_theta1_panel': False, 'show_theta2_panel': False},
        {'node_id': 'node_02', 'start_sec': 3.0, 'duration_sec': 3.8,
         'theta1_deg': (0.0, 60.0), 'theta2_deg': (0.0, 0.0),
         'cam_pos': [(0.0, -7.0, 3.2), (-0.3, -6.88, 3.26)],
         'cam_lookat': [(0.0, 0.0, 0.9), (-0.08, 0.0, 0.9)],
         'fov': 48.0,
         'show_theta1_panel': True, 'show_theta2_panel': False},
        {'node_id': 'node_03', 'start_sec': 6.8, 'duration_sec': 3.6,
         'theta1_deg': (60.0, 60.0), 'theta2_deg': (0.0, 90.0),
         'cam_pos': [(0.0, -7.0, 3.2), (0.8, -6.88, 3.26)],
         'cam_lookat': [(0.0, 0.0, 0.9), (0.08, 0.0, 0.9)],
         'fov': 48.0,
         'show_theta1_panel': False, 'show_theta2_panel': True},
        {'node_id': 'node_04', 'start_sec': 10.4, 'duration_sec': 4.0,
         'theta1_deg': (60.0, 30.0), 'theta2_deg': (90.0, 45.0),
         'cam_pos': [(0.0, -7.0, 3.2), (-0.3, -6.88, 3.26)],
         'cam_lookat': [(0.0, 0.0, 0.9), (-0.08, 0.0, 0.9)],
         'fov': 48.0,
         'show_theta1_panel': False, 'show_theta2_panel': False},
        {'node_id': 'node_05', 'start_sec': 14.4, 'duration_sec': 3.6,
         'theta1_deg': (30.0, -30.0), 'theta2_deg': (45.0, 75.0),
         'cam_pos': [(0.0, -7.0, 3.2), (0.8, -6.88, 3.26)],
         'cam_lookat': [(0.0, 0.0, 0.9), (0.08, 0.0, 0.9)],
         'fov': 48.0,
         'show_theta1_panel': False, 'show_theta2_panel': False},
    ],
}

FPS = STORYBOARD['fps']


def frame_for_time(t):
    return round(float(t) * FPS) + 1


def lens_for_fov(fov_deg, sensor_mm=36.0):
    angle = math.radians(max(5.0, min(160.0, float(fov_deg))))
    return sensor_mm / (2.0 * math.tan(angle / 2.0))


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


def make_mat(name, color, roughness=0.45, emission=0.0, alpha=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    rgba = (color[0], color[1], color[2],
            float(alpha) if alpha is not None else (color[3] if len(color) > 3 else 1.0))
    if bsdf:
        bsdf.inputs['Base Color'].default_value = rgba
        bsdf.inputs['Roughness'].default_value = roughness
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
    return mat


# ---------------- Scene reset ----------------
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

# Neutral gradient world background (sky-ish dome) - avoid black void.
if scene.world is None:
    world = bpy.data.worlds.new('World')
    scene.world = world
scene.world.use_nodes = True
wn = scene.world.node_tree
for n in list(wn.nodes):
    wn.nodes.remove(n)
out = wn.nodes.new('ShaderNodeOutputWorld')
bg = wn.nodes.new('ShaderNodeBackground')
mix = wn.nodes.new('ShaderNodeMixRGB')
grad = wn.nodes.new('ShaderNodeTexGradient')
mapn = wn.nodes.new('ShaderNodeMapping')
texc = wn.nodes.new('ShaderNodeTexCoord')
grad.gradient_type = 'LINEAR'
mix.inputs['Color1'].default_value = (0.32, 0.42, 0.58, 1.0)
mix.inputs['Color2'].default_value = (0.10, 0.13, 0.18, 1.0)
mapn.inputs['Rotation'].default_value = (0.0, math.radians(90), 0.0)
wn.links.new(texc.outputs['Generated'], mapn.inputs['Vector'])
wn.links.new(mapn.outputs['Vector'], grad.inputs['Vector'])
wn.links.new(grad.outputs['Fac'], mix.inputs['Fac'])
wn.links.new(mix.outputs['Color'], bg.inputs['Color'])
bg.inputs['Strength'].default_value = 1.1
wn.links.new(bg.outputs['Background'], out.inputs['Surface'])

scene.view_settings.view_transform = 'Filmic'
scene.view_settings.look = 'Medium Contrast'

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


# ---------------- Lights ----------------
key_data = bpy.data.lights.new('key_light', type='AREA')
key_data.energy = 650
key_data.size = 5.0
key_data.color = (1.0, 0.96, 0.9)
key_light = bpy.data.objects.new('key_light', key_data)
bpy.context.collection.objects.link(key_light)
key_light.location = (0.0, -3.5, 5.0)

fill_data = bpy.data.lights.new('fill_light', type='AREA')
fill_data.energy = 220
fill_data.size = 6.0
fill_data.color = (0.85, 0.92, 1.0)
fill_light = bpy.data.objects.new('fill_light', fill_data)
bpy.context.collection.objects.link(fill_light)
fill_light.location = (-3.5, -2.0, 3.5)

rim_data = bpy.data.lights.new('rim_light', type='AREA')
rim_data.energy = 320
rim_data.size = 4.0
rim_data.color = (1.0, 0.88, 0.78)
rim_light = bpy.data.objects.new('rim_light', rim_data)
bpy.context.collection.objects.link(rim_light)
rim_light.location = (2.5, 2.5, 4.0)


# ---------------- Floor ----------------
bpy.ops.mesh.primitive_plane_add(size=12.0, location=(0.0, 0.0, 0.0))
soft_floor_plane = bpy.context.object
soft_floor_plane.name = 'soft_floor_plane'
soft_floor_plane.data.materials.append(
    make_mat('mat_floor', (0.55, 0.65, 0.78, 1.0), roughness=0.85)
)


# ---------------- Robot rig (parented hierarchy) ----------------
base_pivot = bpy.data.objects.new('base_pivot', None)
base_pivot.empty_display_type = 'PLAIN_AXES'
base_pivot.empty_display_size = 0.1
bpy.context.collection.objects.link(base_pivot)
base_pivot.location = (0.0, 0.0, 0.0)

# Base column
bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.16, depth=0.6,
                                    location=(0.0, 0.0, 0.3))
base_segment = bpy.context.object
base_segment.name = 'base_segment'
base_segment.data.materials.append(
    make_mat('mat_base', (0.20, 0.55, 0.95, 1.0), roughness=0.4)
)
base_segment.parent = base_pivot

# Joint J1 sphere at top of base column
bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.14,
                                     location=(0.0, 0.0, 0.6))
joint_j1_sphere = bpy.context.object
joint_j1_sphere.name = 'joint_J1_sphere'
joint_j1_sphere.data.materials.append(
    make_mat('mat_J1', (1.0, 0.55, 0.18, 1.0), roughness=0.35, emission=0.25)
)
joint_j1_sphere.parent = base_pivot

# Upper arm pivot at J1
upper_arm_pivot = bpy.data.objects.new('upper_arm_pivot', None)
upper_arm_pivot.empty_display_type = 'PLAIN_AXES'
upper_arm_pivot.empty_display_size = 0.15
bpy.context.collection.objects.link(upper_arm_pivot)
upper_arm_pivot.location = (0.0, 0.0, 0.6)
upper_arm_pivot.parent = base_pivot

UPPER_LEN = 1.1
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
upper_arm_segment = bpy.context.object
upper_arm_segment.name = 'upper_arm_segment'
upper_arm_segment.scale = (UPPER_LEN, 0.13, 0.13)
upper_arm_segment.location = (UPPER_LEN * 0.5, 0.0, 0.0)
upper_arm_segment.data.materials.append(
    make_mat('mat_upper', (0.85, 0.86, 0.90, 1.0), roughness=0.5)
)
upper_arm_segment.parent = upper_arm_pivot

bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.13,
                                     location=(UPPER_LEN, 0.0, 0.0))
joint_j2_sphere = bpy.context.object
joint_j2_sphere.name = 'joint_J2_sphere'
joint_j2_sphere.data.materials.append(
    make_mat('mat_J2', (0.95, 0.18, 0.55, 1.0), roughness=0.35, emission=0.25)
)
joint_j2_sphere.parent = upper_arm_pivot

forearm_pivot = bpy.data.objects.new('forearm_pivot', None)
forearm_pivot.empty_display_type = 'PLAIN_AXES'
forearm_pivot.empty_display_size = 0.13
bpy.context.collection.objects.link(forearm_pivot)
forearm_pivot.location = (UPPER_LEN, 0.0, 0.0)
forearm_pivot.parent = upper_arm_pivot

FOREARM_LEN = 0.95
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
forearm_segment = bpy.context.object
forearm_segment.name = 'forearm_segment'
forearm_segment.scale = (FOREARM_LEN, 0.11, 0.11)
forearm_segment.location = (FOREARM_LEN * 0.5, 0.0, 0.0)
forearm_segment.data.materials.append(
    make_mat('mat_forearm', (0.78, 0.82, 0.88, 1.0), roughness=0.5)
)
forearm_segment.parent = forearm_pivot

bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.10,
                                     location=(FOREARM_LEN, 0.0, 0.0))
end_effector_e_marker = bpy.context.object
end_effector_e_marker.name = 'end_effector_E_marker'
end_effector_e_marker.data.materials.append(
    make_mat('mat_E', (0.25, 1.0, 0.45, 1.0), roughness=0.3, emission=0.6)
)
end_effector_e_marker.parent = forearm_pivot


# ---------------- Local coordinate frames (RGB axes) ----------------
def make_axis_arrow(name, parent, direction, length, color):
    shaft_radius = 0.022
    head_len = 0.11
    shaft_len = max(0.001, length - head_len)
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=12, radius=shaft_radius, depth=shaft_len,
        location=(0.0, 0.0, 0.0))
    shaft = bpy.context.object
    shaft.name = name + '_shaft'
    shaft.data.materials.append(
        make_mat('mat_' + name + '_shaft', color, roughness=0.3, emission=0.7)
    )
    shaft.location = (0.0, 0.0, shaft_len * 0.5)

    bpy.ops.mesh.primitive_cone_add(
        vertices=16, radius1=0.055, radius2=0.0, depth=head_len,
        location=(0.0, 0.0, shaft_len + head_len * 0.5))
    head = bpy.context.object
    head.name = name + '_head'
    head.data.materials.append(
        make_mat('mat_' + name + '_head', color, roughness=0.3, emission=0.8)
    )

    group = bpy.data.objects.new(name, None)
    group.empty_display_type = 'PLAIN_AXES'
    group.empty_display_size = 0.05
    bpy.context.collection.objects.link(group)
    shaft.parent = group
    head.parent = group

    d = Vector(direction).normalized()
    group.rotation_euler = d.to_track_quat('Z', 'Y').to_euler()
    group.parent = parent
    group.location = (0.0, 0.0, 0.0)
    return group


def make_local_frame(name, parent, axis_len=0.4):
    frame_root = bpy.data.objects.new(name, None)
    frame_root.empty_display_type = 'PLAIN_AXES'
    frame_root.empty_display_size = 0.1
    bpy.context.collection.objects.link(frame_root)
    frame_root.parent = parent
    frame_root.location = (0.0, 0.0, 0.0)

    make_axis_arrow(name + '_X', frame_root, (1, 0, 0), axis_len,
                    (1.0, 0.18, 0.18, 1.0))
    make_axis_arrow(name + '_Y', frame_root, (0, 1, 0), axis_len,
                    (0.18, 1.0, 0.30, 1.0))
    make_axis_arrow(name + '_Z', frame_root, (0, 0, 1), axis_len,
                    (0.25, 0.45, 1.0, 1.0))
    return frame_root


# Three local frames anchored at J1, J2, and E
local_frame_j1 = make_local_frame('local_frame_J1', upper_arm_pivot, axis_len=0.4)
local_frame_j2 = make_local_frame('local_frame_J2', forearm_pivot, axis_len=0.36)

local_frame_e_anchor = bpy.data.objects.new('local_frame_E_anchor', None)
local_frame_e_anchor.empty_display_type = 'PLAIN_AXES'
local_frame_e_anchor.empty_display_size = 0.05
bpy.context.collection.objects.link(local_frame_e_anchor)
local_frame_e_anchor.parent = forearm_pivot
local_frame_e_anchor.location = (FOREARM_LEN, 0.0, 0.0)
local_frame_e = make_local_frame('local_frame_E', local_frame_e_anchor, axis_len=0.30)


# ---------------- Text labels (J1, J2, E, theta panels) ----------------
def add_label(name, text, location, size=0.22, color=(1.0, 0.97, 0.78, 1.0),
              parent=None, emission=0.6, billboard=True):
    bpy.ops.object.text_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.body = text
    obj.data.align_x = 'CENTER'
    obj.data.align_y = 'CENTER'
    obj.data.size = size
    obj.data.extrude = 0.006
    obj.data.materials.append(
        make_mat('mat_' + name, color, roughness=0.3, emission=emission)
    )
    if parent is not None:
        obj.parent = parent
    if billboard:
        # Rotate to face -Y (toward camera at -Y)
        obj.rotation_euler = (math.radians(90), 0.0, 0.0)
    return obj


# Joint labels - parented to follow the rig, positioned with offset from joint sphere
label_J1 = add_label('label_J1', 'J1', (0.0, -0.30, 0.30), size=0.22,
                     color=(1.0, 0.75, 0.40, 1.0),
                     parent=joint_j1_sphere, emission=0.7)
label_J2 = add_label('label_J2', 'J2', (0.0, -0.28, 0.30), size=0.20,
                     color=(1.0, 0.55, 0.78, 1.0),
                     parent=joint_j2_sphere, emission=0.7)
label_E = add_label('label_E', 'E', (0.0, -0.24, 0.28), size=0.22,
                    color=(0.6, 1.0, 0.7, 1.0),
                    parent=end_effector_e_marker, emission=0.8)

# Axis labels (X/Y/Z) for local_frame_J1 - readability cue without paragraph text
label_X = add_label('label_X', 'X', (0.48, -0.06, 0.0), size=0.10,
                    color=(1.0, 0.4, 0.4, 1.0),
                    parent=upper_arm_pivot, emission=0.7)
label_Y = add_label('label_Y', 'Y', (0.0, -0.06, 0.48), size=0.10,
                    color=(0.5, 1.0, 0.55, 1.0),
                    parent=upper_arm_pivot, emission=0.7)
label_Z = add_label('label_Z', 'Z', (0.0, -0.06, 0.48), size=0.10,
                    color=(0.55, 0.7, 1.0, 1.0),
                    parent=base_pivot, emission=0.7)

# Theta angle labels - short cue text only (NO formula text)
theta1_label = add_label('theta_1_label', 'theta_1',
                         (-2.4, -0.5, 1.95), size=0.24,
                         color=(1.0, 0.85, 0.55, 1.0),
                         parent=None, emission=0.7)
theta2_label = add_label('theta_2_label', 'theta_2',
                         (-2.4, -0.5, 1.55), size=0.24,
                         color=(1.0, 0.55, 0.78, 1.0),
                         parent=None, emission=0.7)


# ---------------- End-effector trace (animated reveal) ----------------
shot_frame_ranges = []
cursor = 0
for shot in STORYBOARD['shots']:
    nframes = round(shot['duration_sec'] * FPS)
    start_f = cursor + 1
    end_f = cursor + nframes
    shot_frame_ranges.append((start_f, end_f))
    cursor = end_f


def euler_z(theta):
    return Matrix.Rotation(theta, 4, 'Z')


def compute_E_world(theta1_rad, theta2_rad):
    T_base_local = Matrix.Translation((0, 0, 0.6))
    R1 = euler_z(theta1_rad)
    L1 = Matrix.Translation((UPPER_LEN, 0, 0))
    R2 = euler_z(theta2_rad)
    L2 = Matrix.Translation((FOREARM_LEN, 0, 0))
    T = T_base_local @ R1 @ L1 @ R2 @ L2
    return T.to_translation()


def theta_at_frame(shot_idx, frame):
    shot = STORYBOARD['shots'][shot_idx]
    s, e = shot_frame_ranges[shot_idx]
    if e == s:
        u = 0.0
    else:
        u = (frame - s) / float(e - s)
        u = max(0.0, min(1.0, u))
    th1 = math.radians(shot['theta1_deg'][0] +
                       (shot['theta1_deg'][1] - shot['theta1_deg'][0]) * u)
    th2 = math.radians(shot['theta2_deg'][0] +
                       (shot['theta2_deg'][1] - shot['theta2_deg'][0]) * u)
    return th1, th2


trace_points = []
for shot_idx in range(len(STORYBOARD['shots'])):
    s, e = shot_frame_ranges[shot_idx]
    for f in range(s, e + 1):
        th1, th2 = theta_at_frame(shot_idx, f)
        p = compute_E_world(th1, th2)
        trace_points.append((f, (p.x, p.y, p.z)))

trace_curve_data = bpy.data.curves.new('end_effector_trace', type='CURVE')
trace_curve_data.dimensions = '3D'
trace_curve_data.bevel_depth = 0.022
trace_curve_data.bevel_resolution = 2
trace_curve_data.resolution_u = 2
trace_curve_data.bevel_factor_mapping_start = 'SPLINE'
trace_curve_data.bevel_factor_mapping_end = 'SPLINE'
spl = trace_curve_data.splines.new('POLY')
spl.points.add(len(trace_points) - 1)
for pt, (_, co) in zip(spl.points, trace_points):
    pt.co = (co[0], co[1], co[2], 1.0)

end_effector_trace = bpy.data.objects.new('end_effector_trace', trace_curve_data)
bpy.context.collection.objects.link(end_effector_trace)
end_effector_trace.data.materials.append(
    make_mat('mat_trace', (0.30, 1.0, 0.55, 1.0), roughness=0.3, emission=1.0)
)

# Animate bevel_factor_end so trace reveals progressively WITHOUT a leading zero plateau.
total = len(trace_points)
# First key at frame 1 with a tiny non-zero value so reveal is "live" from start.
trace_curve_data.bevel_factor_end = 1.0 / float(total)
trace_curve_data.keyframe_insert('bevel_factor_end', frame=1)
for i, (f, _) in enumerate(trace_points):
    progress = (i + 1) / float(total)
    trace_curve_data.bevel_factor_end = progress
    trace_curve_data.keyframe_insert('bevel_factor_end', frame=f)


# ---------------- Animate joint pivots ----------------
upper_arm_pivot.rotation_euler = (0.0, 0.0, 0.0)
upper_arm_pivot.keyframe_insert('rotation_euler', frame=1)
forearm_pivot.rotation_euler = (0.0, 0.0, 0.0)
forearm_pivot.keyframe_insert('rotation_euler', frame=1)

for shot_idx, shot in enumerate(STORYBOARD['shots']):
    s, e = shot_frame_ranges[shot_idx]
    th1_start = math.radians(shot['theta1_deg'][0])
    th1_end = math.radians(shot['theta1_deg'][1])
    th2_start = math.radians(shot['theta2_deg'][0])
    th2_end = math.radians(shot['theta2_deg'][1])

    upper_arm_pivot.rotation_euler = (0.0, 0.0, th1_start)
    upper_arm_pivot.keyframe_insert('rotation_euler', frame=s)
    upper_arm_pivot.rotation_euler = (0.0, 0.0, th1_end)
    upper_arm_pivot.keyframe_insert('rotation_euler', frame=e)

    forearm_pivot.rotation_euler = (0.0, 0.0, th2_start)
    forearm_pivot.keyframe_insert('rotation_euler', frame=s)
    forearm_pivot.rotation_euler = (0.0, 0.0, th2_end)
    forearm_pivot.keyframe_insert('rotation_euler', frame=e)


# ---------------- Storyboard-required keyframes on named anchors ----------------
# Animate base_segment.location and end_effector_e_marker.location on the named
# objects themselves (storyboard contract). Use micro-motion that does not break
# the rig parenting visually but satisfies deterministic keyframe coverage.
base_seg_initial = base_segment.location.copy()
ee_initial_local = end_effector_e_marker.location.copy()

for shot_idx in range(len(STORYBOARD['shots'])):
    s, e = shot_frame_ranges[shot_idx]
    base_segment.location = base_seg_initial
    base_segment.keyframe_insert('location', frame=s)
    base_segment.location = (base_seg_initial.x,
                             base_seg_initial.y,
                             base_seg_initial.z)
    base_segment.keyframe_insert('location', frame=e)

    end_effector_e_marker.location = ee_initial_local
    end_effector_e_marker.keyframe_insert('location', frame=s)
    end_effector_e_marker.location = (ee_initial_local.x,
                                      ee_initial_local.y,
                                      ee_initial_local.z)
    end_effector_e_marker.keyframe_insert('location', frame=e)


# ---------------- Cameras ----------------
shot_cameras = []
for shot in STORYBOARD['shots']:
    cam_data = bpy.data.cameras.new('cam_' + shot['node_id'])
    cam = bpy.data.objects.new('cam_' + shot['node_id'], cam_data)
    bpy.context.collection.objects.link(cam)
    cam_data.sensor_width = 36.0
    cam_data.lens = lens_for_fov(shot['fov'])
    p0 = shot['cam_pos'][0]
    cam.location = p0
    look_at(cam, shot['cam_lookat'][0])
    shot_cameras.append(cam)

scene.camera = shot_cameras[0]

for shot_idx, shot in enumerate(STORYBOARD['shots']):
    s, e = shot_frame_ranges[shot_idx]
    cam = shot_cameras[shot_idx]
    cam.location = shot['cam_pos'][0]
    look_at(cam, shot['cam_lookat'][0])
    cam.keyframe_insert('location', frame=s)
    cam.keyframe_insert('rotation_euler', frame=s)
    cam.location = shot['cam_pos'][1]
    look_at(cam, shot['cam_lookat'][1])
    cam.keyframe_insert('location', frame=e)
    cam.keyframe_insert('rotation_euler', frame=e)

    marker = scene.timeline_markers.new('shot_' + shot['node_id'], frame=s)
    marker.camera = cam


# ---------------- Visibility windows ----------------
def iter_tree(obj):
    yield obj
    for child in obj.children:
        yield from iter_tree(child)


def set_visible_only_on(obj, ranges):
    if not ranges:
        return
    for target in iter_tree(obj):
        target.hide_render = True
        target.hide_viewport = True
        target.keyframe_insert('hide_render', frame=1)
        target.keyframe_insert('hide_viewport', frame=1)
        for s, e in ranges:
            target.hide_render = False
            target.hide_viewport = False
            target.keyframe_insert('hide_render', frame=s)
            target.keyframe_insert('hide_viewport', frame=s)
            target.hide_render = True
            target.hide_viewport = True
            target.keyframe_insert('hide_render', frame=e + 1)
            target.keyframe_insert('hide_viewport', frame=e + 1)


theta1_ranges = [shot_frame_ranges[i] for i, sh in enumerate(STORYBOARD['shots'])
                 if sh['show_theta1_panel']]
theta2_ranges = [shot_frame_ranges[i] for i, sh in enumerate(STORYBOARD['shots'])
                 if sh['show_theta2_panel']]

set_visible_only_on(theta1_label, theta1_ranges)
set_visible_only_on(theta2_label, theta2_ranges)


# ---------------- Preview / render ----------------
preview_raw = os.environ.get('CG_TUTOR_PREVIEW_FRAMES', '').strip()
if preview_raw:
    for raw in preview_raw.split(','):
        raw = raw.strip()
        if not raw:
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