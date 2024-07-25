import argparse
import os
import time
import copy
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from haven import haven_utils as hu
import open3d as o3d

o3d.visualization.webrtc_server.enable_webrtc()

import numpy as np
import torch
import torch.nn

import matplotlib.pyplot as plt

from src import models
from src.scenes import NeuralScene
from src.pointLF.ptlf_vis import get_pcd_vis
from src import utils as ut

from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles, \
    matrix_to_quaternion, quaternion_to_matrix
from pytorch3d.transforms.so3 import so3_log_map, so3_exponential_map


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='nplf')
    parser.add_argument('--savedir', type=str, default='./save',
                        help='The directory of saved results.')
    parser.add_argument('--datadir', type=str, default='./data',
                        help='The directory of data.')
    return parser.parse_args()
args = parse_args()

exp_dict = hu.load_json(os.path.join(args.savedir, 'exp_dict.json'))
model_state_dict = hu.torch_load(os.path.join(args.savedir, 'model.pth'))

exp_dict['n_rays'] = 8192*2
# exp_dict['chunk'] = 512
lf_config = exp_dict["lightfield"]
# exp_dict["lightfield"]["n_sample_pts"] = 7000
lf_config
# exp_dict["scale"] = 0.0625

#
seed = 42 + exp_dict.get("runs", 0)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

scene = NeuralScene(
    scene_list=exp_dict["scenes"],
    datadir=args.datadir,
    args=args,
    exp_dict=exp_dict,
)

model = models.Model(scene, exp_dict, 'cuda:0', precache=exp_dict.get("precache"))

# Load checkpoint
model.renderer.load_state_dict(
        model_state_dict["renderer"], strict=False
    )

batch_size = 1
epoch_size = 1
rand_sampler = torch.utils.data.RandomSampler(
    scene, num_samples=epoch_size * batch_size, replacement=True
)

scene_loader = torch.utils.data.DataLoader(
    scene,
    sampler=rand_sampler,
    collate_fn=ut.collate_fn_dict_of_lists,
    batch_size=batch_size,
    num_workers=0,
    # pin_memory=True,
    drop_last=True,
)



reference_id = 29
# reference_id = 25
# reference_id = 129

ind_list = list(np.arange(len(scene)))
bi = ind_list[reference_id]

exp_name = 'middle_curve_01'

euler_mid = None
translation_mid = None

# Rotation ('ZYX'/"ROLL", "YAW", "PITCH")
euler_start = np.array([np.deg2rad(0.),np.deg2rad(0.),np.deg2rad(0.)])
euler_mid = np.array([np.deg2rad(0.),np.deg2rad(0.),np.deg2rad(0.)])
euler_end = np.array([np.deg2rad(0.),np.deg2rad(0.),np.deg2rad(0.)])

# euler_start = np.array([np.deg2rad(0.),np.deg2rad(-8.),np.deg2rad(0.)])
# euler_end = np.array([np.deg2rad(0.),np.deg2rad(8.),np.deg2rad(0.)])

# Translation in camera space; when level: 1.-"LEFT", 2.-"UP", 3.-"FORWARD"
translation_start = np.array([0.,0.0,0.0])
translation_mid = np.array([0.025,0.0,0.10])
translation_end = np.array([0.075,0.0,0.15])

# translation_start = np.array([3,0.0,0.0])
# translation_end = np.array([-3,0.0,0.0])

n_steps = 12

render_reference = False

pose_only = False

img_name = os.path.join(args.savedir, 'images/interpolation/', (exp_name + '_'))

ind_list = list(np.arange(len(scene)))
bi = ind_list[reference_id]
print(img_name)


with torch.no_grad():
    time0 = time.time()
    b = ut.collate_fn_dict_of_lists([scene.__getitem__(bi, intersections_only=False, random_rays=False,)])
    print(time.time()-time0)
    time1 = time.time()
    rgb_out = model.renderer.forward_on_batch(b)
    print(time.time()-time1)

rendered_img = rgb_out['color_out'].reshape(b['images'][0].shape)
plt.figure()
plt.imshow(rendered_img)
plt.figure()
plt.imshow(b["images"][0].detach().cpu())

val_dict = model.val_on_scene(scene, savedir_images=os.path.join(args.savedir, "images"), all_frames=True)


def get_render_cam_trafo(translation_change, euler_angles, scene, reference_id):
    print("Manipulating Reference Frame.")

    ind_list = list(np.arange(len(scene)))

    # Copy reference frame
    ref_frame_idx = scene.frames_cameras[reference_id][0]

    new_frame = scene.frames[ref_frame_idx]
    ref_new_frame_copy = copy.deepcopy(new_frame)
    cam_idx = 1
    # print([(node_idx, cam_node.name) for (node_idx, cam_node) in scene.nodes['camera'].items()])
    camera_ed = new_frame.get_edge_by_child_idx([cam_idx])[0][0]
    cam_ed_idx = new_frame.get_edge_idx_by_child_idx([cam_idx])[0][0]
    # print(camera_ed.get_transformation_c2p().get_matrix().detach().cpu().numpy()[0].T)
    rot_skew_angle = copy.deepcopy(camera_ed.rotation)
    cam2wo_rot = so3_exponential_map(rot_skew_angle)
    cam2wo_translat = copy.deepcopy(camera_ed.translation)
    # print(cam2wo_rot)

    # Create new camera edge
    DTYPE = camera_ed.translation.dtype
    DEVICE = camera_ed.translation.device

    # Get changed rotation
    euler_angels_torch = torch.tensor(euler_angles, dtype=DTYPE, device=DEVICE)
    rot_change = euler_angles_to_matrix(euler_angels_torch, 'ZYX')
    new_rot = torch.matmul(cam2wo_rot, rot_change)
    assert torch.norm(torch.matmul(new_rot[0], new_rot[0].T) - torch.eye(3)) < 1e-5

    # Get changed camera
    translation_change_camera = torch.tensor(translation_change, dtype=DTYPE, device=DEVICE)
    translation_change_world = torch.matmul(new_rot[0], translation_change_camera)

    translation_world = cam2wo_translat + translation_change_world
    camera_ed.translation = translation_world
    print("Reference Translation: {}; New Translation {}".format(
        cam2wo_translat.detach().cpu().numpy(),
        scene.frames[reference_id].edges[cam_ed_idx].translation.detach().cpu().numpy()))
    camera_ed.rotation = so3_log_map(new_rot)
    print("Reference Rotation: {}; New Rotation {}".format(
        rot_skew_angle.detach().cpu().numpy(),
        scene.frames[reference_id].edges[cam_ed_idx].rotation.detach().cpu().numpy()))

    # Output new frame
    translation_world = translation_world.detach().cpu().numpy()
    rotation_world = new_rot.detach().cpu().numpy()

    # Reverse Scene Graph Manipulation
    scene.frames[ref_frame_idx] = copy.deepcopy(ref_new_frame_copy)

    return translation_world, rotation_world


def get_rotation_steps(euler_start, euler_end, n_steps):
    # Rotation Interpolation with slerp
    mat_start = euler_angles_to_matrix(torch.tensor(euler_start), "ZYX")
    quat_start = matrix_to_quaternion(mat_start)

    mat_end = euler_angles_to_matrix(torch.tensor(euler_end), "ZYX")
    quat_end = matrix_to_quaternion(mat_end)

    new_euler_ls = []
    steps = torch.linspace(0, 1, n_steps)

    cosHalfTheta = torch.sum(quat_start * quat_end)
    halfTheta = torch.acos(cosHalfTheta)
    sinHalfTheta = torch.sqrt(1.0 - cosHalfTheta * cosHalfTheta)

    if halfTheta != 0:
        ratioA = torch.sin((1 - steps) * halfTheta) / sinHalfTheta
        ratioB = torch.sin(steps * halfTheta) / sinHalfTheta
        mid_quat = quat_start * ratioA[:, None] + quat_end * ratioB[:, None]
        for i in range(n_steps):
            new_quat = mid_quat[i, None]
            new_euler = matrix_to_euler_angles(quaternion_to_matrix(new_quat), "ZYX")
            new_euler_ls.append(new_euler.numpy())
    else:
        for i in range(n_steps):
            new_euler_ls.append(euler_start)

    return new_euler_ls


def render_frame_from_cam_trafo(translation_change, euler_angles, scene, reference_id):
    print("Manipulating Reference Frame.")

    ind_list = list(np.arange(len(scene)))

    # Copy reference frame
    ref_frame_idx = scene.frames_cameras[reference_id][0]

    new_frame = scene.frames[ref_frame_idx]
    ref_new_frame_copy = copy.deepcopy(new_frame)
    cam_idx = 1
    # print([(node_idx, cam_node.name) for (node_idx, cam_node) in scene.nodes['camera'].items()])
    camera_ed = new_frame.get_edge_by_child_idx([cam_idx])[0][0]
    cam_ed_idx = new_frame.get_edge_idx_by_child_idx([cam_idx])[0][0]
    # print(camera_ed.get_transformation_c2p().get_matrix().detach().cpu().numpy()[0].T)
    rot_skew_angle = copy.deepcopy(camera_ed.rotation)
    cam2wo_rot = so3_exponential_map(rot_skew_angle)
    cam2wo_translat = copy.deepcopy(camera_ed.translation)
    # print(cam2wo_rot)

    # Create new camera edge
    DTYPE = camera_ed.translation.dtype
    DEVICE = camera_ed.translation.device

    # Get changed rotation
    euler_angels_torch = torch.tensor(euler_angles, dtype=DTYPE, device=DEVICE)
    rot_change = euler_angles_to_matrix(euler_angels_torch, 'ZYX')
    new_rot = torch.matmul(cam2wo_rot, rot_change)
    assert torch.norm(torch.matmul(new_rot[0], new_rot[0].T) - torch.eye(3)) < 1e-5

    # Get changed camera
    translation_change_camera = torch.tensor(translation_change, dtype=DTYPE, device=DEVICE)
    translation_change_world = torch.matmul(new_rot[0], translation_change_camera)

    translation_world = cam2wo_translat + translation_change_world
    camera_ed.translation = translation_world
    print("Reference Translation: {}; New Translation {}".format(
        cam2wo_translat.detach().cpu().numpy(),
        scene.frames[reference_id].edges[cam_ed_idx].translation.detach().cpu().numpy()))
    camera_ed.rotation = so3_log_map(new_rot)
    print("Reference Rotation: {}; New Rotation {}".format(
        rot_skew_angle.detach().cpu().numpy(),
        scene.frames[reference_id].edges[cam_ed_idx].rotation.detach().cpu().numpy()))

    # Get best point cloud
    li_node_id = [key for key, li in scene.nodes['lidar'].items()][0]
    distance = np.linalg.norm(cam2wo_translat.detach().cpu().numpy() - scene.frames[reference_id].edges[
        cam_ed_idx].translation.detach().cpu().numpy())
    closest_frame_idx = reference_id
    for i in range(len(scene) - reference_id - 1):

        check_frame_id = i + reference_id + 1
        check_cam_ed = scene.frames[check_frame_id].get_edge_by_child_idx([1])[0][0]
        distance_3D = check_cam_ed.translation - scene.frames[reference_id].edges[cam_ed_idx].translation
        new_dist = torch.norm(distance_3D)
        # print(new_dist)
        if new_dist < distance:
            distance = new_dist
            closest_frame_idx = check_frame_id

    print("Old PT Cloud Path: {}; New PT Cloud Path: {}".format(scene.frames[reference_id].point_cloud_pth[li_node_id],
                                                                scene.frames[closest_frame_idx].point_cloud_pth[
                                                                    li_node_id]))

    scene.frames[reference_id].point_cloud_pth[li_node_id] = scene.frames[closest_frame_idx].point_cloud_pth[li_node_id]

    # Preprocess new frame
    print("Extracting rays and klosest points.")
    bi = ind_list[reference_id]
    batch = ut.collate_fn_dict_of_lists([scene.__getitem__(bi, intersections_only=False, random_rays=False, )])
    print("Done.")
    print("Rendering New Frame.")
    rgb_out = model.renderer.forward_on_batch(batch)
    ref_shape = batch['images'][0].shape
    print("Done.")
    translation_world = translation_world.detach().cpu().numpy()
    rotation_world = new_rot.detach().cpu().numpy()

    # Reverse Scene Graph Manipulation
    scene.frames[ref_frame_idx] = copy.deepcopy(ref_new_frame_copy)

    return rgb_out, ref_shape, translation_world, rotation_world


midpoint = False
if translation_mid is None:
    d_translation = translation_end - translation_start
    traj_start = translation_start
else:
    d_translation = translation_mid - translation_start
    traj_start = translation_start
    midpoint = True
# print(d_translation)
if euler_mid is None:
    new_euler = get_rotation_steps(euler_start, euler_end, n_steps)
else:
    new_euler = get_rotation_steps(euler_start, euler_mid, n_steps)
    midpoint = True

if render_reference:
    print("Reference Remdering")
    with torch.no_grad():
        ind_list = list(np.arange(len(scene)))
        bi = ind_list[reference_id]
        b = ut.collate_fn_dict_of_lists([scene.__getitem__(bi, intersections_only=False, random_rays=False, )])
        print(b)
        rgb_out = model.renderer.forward_on_batch(b)
        plt.figure()
        rendered_img = rgb_out['color_out'].reshape(b['images'][0].shape)
        plt.imshow(rendered_img)

imgs = []
cam_position = []
cam_orientation = []
n_points = 1 + 1 * midpoint
print(n_points)
for p in range(n_points):

    for step, euler_angles in enumerate(new_euler):
        if n_steps > 1:
            step_sz = d_translation * (step / (n_steps - 1))
        else:
            step_sz = np.zeros_like(translation_start)

        translation_change = traj_start + step_sz
        print("Translational Change from Reference (m): {}.".format(translation_change))

        print("Rotational Change from Reference (deg): {}.".format(np.rad2deg(euler_angles)))

        if not pose_only:
            with torch.no_grad():
                rgb_out, ref_shape, new_translation, new_rotation = \
                    render_frame_from_cam_trafo(translation_change, euler_angles, scene, reference_id)

            plt.figure()
            rendered_img = rgb_out['color_out'].reshape(ref_shape)
            imgs.append(rendered_img)
            plt.imshow(rendered_img)
        else:
            new_translation, new_rotation = get_render_cam_trafo(translation_change, euler_angles, scene, reference_id)

        cam_position.append(new_translation)
        cam_orientation.append(new_rotation)

    if midpoint:
        new_euler = get_rotation_steps(euler_mid, euler_end, n_steps)[1:]
        d_translation = translation_end - translation_mid
        traj_start = translation_mid + (d_translation * (1 / (n_steps - 1)))

cam_position = np.concatenate(cam_position)
cam_orientation = np.concatenate(cam_orientation)

from PIL import Image

if not os.path.exists(os.path.join(args.savedir, 'images/interpolation')):
    os.mkdir(os.path.join(args.savedir, 'images/interpolation'))

for k, img in enumerate(imgs):
    img = np.round(img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    path = img_name + "{}.png".format(str(k).zfill(3))
    img.save(path)

print(path)

all_cam_transla = []
cam_idx = 1
for fr in scene.frames.values():
    cam_ed = fr.get_edge_by_child_idx([cam_idx])[0][0]
    all_cam_transla.append(cam_ed.translation.detach().numpy())
all_cam_transla = np.concatenate(all_cam_transla)[reference_id - 2:reference_id + 15]

ax = plt.figure()
plt.scatter(all_cam_transla[:, 0], all_cam_transla[:, 1])
plt.axis('equal')

# plt.scatter(cam_position[:,0], cam_position[:, 1], c="orange")

for t, r in zip(cam_position[:], cam_orientation[:]):
    plt.scatter(t[0], t[1], c="orange")
    plt.arrow(t[0], t[1], r[0, 2] * .5, r[1, 2] * .5, color="orange")
    plt.axis('equal')

plt.savefig(img_name + 'plt.png')
print(img_name)
# for fr in scene.frames.items:
#     fr.edges

# Export Render Pose Information
segmemt_pth = ''
for s in scene.dataset.images[0].split('/')[1:-2]:
    segmemt_pth += '/' + s

scene_descript = scene.frames_cameras[reference_id][2]
first_fr = scene_descript['first_frame']
last_fr = scene_descript['last_frame']
render_pose_file_n = "render_poses_{}_{}.npy".format(str(first_fr).zfill(4), str(last_fr).zfill(4))

# Get camera pose matrix for NeRF
cam_pose = np.concatenate([cam_orientation, cam_position[..., None]], axis=-1)
cam_pose = np.concatenate([cam_pose, np.repeat(np.array([[[0., 0., 0., 1.]]]), len(cam_pose), axis=0)], axis=1)
cam_pose_openGL = cam_pose.dot(np.array([[-1., 0., 0., 0., ], [0., 1., 0., 0., ], [0., 0., -1., 0., ], [0., 0., 0., 1., ], ]))

c_id = scene.frames_cameras[reference_id][1]
H = scene.nodes['camera'][c_id].H / exp_dict['scale']
W = scene.nodes['camera'][c_id].W / exp_dict['scale']
f = scene.nodes['camera'][c_id].intrinsics.f_x.detach().cpu().numpy() / exp_dict['scale']

hwf = np.repeat(np.array([[H, W, f, reference_id]]), len(cam_pose), axis=0)[..., None]
cam_pose_openGL = np.concatenate([cam_pose_openGL, hwf], axis=2)

cam_pose_openGL.shape

np.save(os.path.join(segmemt_pth, exp_name + '_' + render_pose_file_n,), cam_pose_openGL)
'''os.path.join(segmemt_pth, exp_name + '_' + render_pose_file_n)
'''


