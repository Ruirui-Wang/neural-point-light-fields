import copy
import csv

import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from src.datasets import rosbag, CameraIntrinsicHandler

transform_matrix_z = np.array([
            [-1., 0., 0., 0.],
            [0., -1., 0., 0.],
            [0., 0., 1., 1.2],
            [0., 0., 0., 1.]
        ])
timestamps = []
frame_ids = []
positions = []
orientations = []
poses = []
pointcloud_timestamps = []
with open('./data/pose_data.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        timestamp_sec = float(row[0])
        timestamp_nsec = float(row[1])
        frame_id = row[2]
        x = float(row[3])
        y = float(row[4])
        z = float(row[5])
        qx = float(row[6])
        qy = float(row[7])
        qz = float(row[8])
        qw = float(row[9])
        timestamps.append(timestamp_sec + timestamp_nsec * 1e-9)
        frame_ids.append(frame_id)
        positions.append((x, y, z))
        orientations.append((qx, qy, qz, qw))
with open('./data/timestamp_data.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        index = int(row[0])
        timestamp = int(row[1])
        pointcloud_timestamps.append(timestamp)

for i in range(len(pointcloud_timestamps)):
    pose = np.eye(4)
    t1 = None
    t2 = None
    for j in range(len(positions)):
        if int(timestamps[j] * 1e9) - int(pointcloud_timestamps[i]) >= 0:
            t1 = j - 1
            t2 = j
            break
    if t1 == None and t2 == None:
        break
    else:
        print(t1, t2)
        alpha = (int(pointcloud_timestamps[i]) - timestamps[t1] * 1e9) / (timestamps[t2] * 1e9 - timestamps[t1] * 1e9)
        desired_position = (
            positions[t1][0] + alpha * (positions[t2][0] - positions[t1][0]),
            positions[t1][1] + alpha * (positions[t2][1] - positions[t1][1]),
            positions[t1][2] + alpha * (positions[t2][2] - positions[t1][2])
        )

        desired_orientation = (
            orientations[t1][0] + alpha * (orientations[t2][0] - orientations[t1][0]),
            orientations[t1][1] + alpha * (orientations[t2][1] - orientations[t1][1]),
            orientations[t1][2] + alpha * (orientations[t2][2] - orientations[t1][2]),
            orientations[t1][3] + alpha * (orientations[t2][3] - orientations[t1][3])
        )
        pose[:3, 3] = desired_position
        pose[:3, :3] = np.dot(transform_matrix_z[:3, :3], rosbag.quaternion_to_rotation_matrix(*desired_orientation))
        poses.append(pose)



rotation_matrix = np.array([[-0.2, 0.98, -0.021],
       [0.086, -0.0036, -1],
       [-0.97, -0.21, -0.083]])


translation_vector = np.array([-0.12, -0.047, 0.051])

extrinsics_lidar_to_cam = np.eye(4)
extrinsics_lidar_to_cam[:3, :3] = rotation_matrix
extrinsics_lidar_to_cam[:3, 3] = translation_vector

extrinsics_cam_to_lidar = np.linalg.inv(extrinsics_lidar_to_cam)








def compute_lidar_poses(veh_poses, extrinsics_cam_to_lidar):
    lidar_poses = []
    for veh_pose in veh_poses:
        lidar_pose = veh_pose @ extrinsics_cam_to_lidar
        lidar_poses.append(lidar_pose)
    return np.stack(lidar_poses)

def compute_camera_poses(lidar_poses, extrinsics_lidar_to_cam):
    cam_poses = []
    for lidar_pose in lidar_poses:
        cam_pose = lidar_pose @ extrinsics_lidar_to_cam
        cam_poses.append(cam_pose)
    return np.stack(cam_poses)

lidar_poses = compute_lidar_poses(poses, extrinsics_cam_to_lidar)
cam_poses = compute_camera_poses(lidar_poses, extrinsics_lidar_to_cam)

print("Lidar poses in world coordinates:\n", lidar_poses[100])
print("Camera poses in world coordinates:\n", cam_poses[100])
print("Vehicle poses in world coordinates:\n", poses[100])


def plot_poses(poses, color='blue', label='Pose'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for pose in poses:
        x, y, z = pose[:3, 3]
        ax.scatter(x, y, z, c=color, label=label)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()

'''plot_poses(poses, color='blue', label='Camera Poses')

plot_poses(lidar_poses, color='red', label='Lidar Poses')

plot_poses(cam_poses, color='green', label='Vehicle Poses')'''


import plotly.graph_objs as go

def plot_poses_all(poses1, poses2, poses3):
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=[pose[0, 3] for pose in poses1],
        y=[pose[1, 3] for pose in poses1],
        z=[pose[2, 3] for pose in poses1],
        mode='markers',
        marker=dict(size=4, color='green'),
        name='Camera Poses'
    ))

    fig.add_trace(go.Scatter3d(
        x=[pose[0, 3] for pose in poses2],
        y=[pose[1, 3] for pose in poses2],
        z=[pose[2, 3] for pose in poses2],
        mode='markers',
        marker=dict(size=2, color='blue'),
        name='Lidar Poses'
    ))

    fig.add_trace(go.Scatter3d(
        x=[pose[0, 3] for pose in poses3],
        y=[pose[1, 3] for pose in poses3],
        z=[pose[2, 3] for pose in poses3],
        mode='markers',
        marker=dict(size=2, color='red'),
        name='Vehicle Poses'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        legend=dict(x=0, y=1)
    )

    fig.show()

'''plot_poses_all(cam_poses, lidar_poses, poses)
'''
def project_and_save(points, image, M1, M2):
    resolution = image.shape

    coords = points[:, 0:3]
    ones = np.ones(len(coords)).reshape(-1, 1)
    coords = np.concatenate([coords, ones], axis=1)
    transform = copy.deepcopy(M1 @ M2).reshape(4, 4)
    coords = coords @ transform.T
    coords = coords[np.where(coords[:, 2] > 0)]

    coords[:, 0] /= coords[:, 2]
    coords[:, 1] /= coords[:, 2]

    coords = coords[np.where(coords[:, 2] > 1e-1)]
    coords = coords[np.where(coords[:, 2] < 2e1)]
    coords = coords[np.where(coords[:, 0] > 0)]
    coords = coords[np.where(coords[:, 0] < resolution[1])]
    coords = coords[np.where(coords[:, 1] > 0)]
    coords = coords[np.where(coords[:, 1] < resolution[0])]

    return coords, coords[:, 2]


def show_projected_image(image, coords=None, depth=None):
    canvas = image.copy()
    dmax = max(depth)
    dmin = min(depth)

    if coords is not None:
        for index in range(coords.shape[0]):
            p = (int(coords[index, 0]), int(coords[index, 1]))
            c = (depth[index] - dmin) / (dmax - dmin) * 255
            cv2.circle(canvas, p, 2, color=[255 - c, c, c], thickness=1)

    plt.imshow(canvas)
    plt.show()

img = cv2.imread('./data/images/basler01/000080.png')
pointclouds = np.loadtxt('./data/lidar/basler01/000080.txt')
intrinsics = CameraIntrinsicHandler('./dataset/camera_calibration/calibration_basler01.txt').get_parameters()

intrinsics_4x4 = np.eye(4)
intrinsics_4x4[:3, :3] = intrinsics.camera_matrix

print(extrinsics_lidar_to_cam)

coords, depth = project_and_save(pointclouds, img, intrinsics_4x4 ,extrinsics_lidar_to_cam)
show_projected_image(cv2.flip(cv2.flip(img, 0),1), coords=coords,depth = depth)
show_projected_image(img, coords=coords,depth = depth)






