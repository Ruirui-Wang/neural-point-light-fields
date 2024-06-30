import csv
import os
import pickle
import numpy as np
from collections import defaultdict
from copy import deepcopy

from src.datasets.camera_intrinsics_handler import CameraIntrinsicHandler
from src.datasets.lidar_image_projector import extrinsics
from src.datasets.utils import invert_transformation,roty_matrix, rotz_matrix
import matplotlib.pyplot as plt
from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles
import torch


class Rosbag:
    def __init__(self, datadir, scene_dict):
        self.num_cameras = 1
        self.num_lasers = 1
        self.lidar_poses = []
        self.poses = []
        image_directory = './data/images/basler01'
        image_paths = []
        for filename in os.listdir(image_directory):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(image_directory, filename)
                image_paths.append(image_path)
        sorted_image_paths = sorted(image_paths)
        self.images = sorted_image_paths


        pointcloud_directory = './data/lidar/basler01'
        point_cloud_paths = []
        for filename in os.listdir(pointcloud_directory):
            if filename.endswith('.txt') :
                point_cloud_path = os.path.join(pointcloud_directory, filename)
                point_cloud_paths.append(point_cloud_path)
        sorted_point_cloud_paths = sorted(point_cloud_paths)
        self.point_cloud_pth = sorted_point_cloud_paths

        self.num_cam_frames = len(self.images)
        assert np.round(len(self.images)/self.num_cameras)==len(self.images)/self.num_cameras
        self.num_frames = len(self.images)//self.num_cameras

        intrinsics = CameraIntrinsicHandler('./dataset/camera_calibration/calibration_basler01.txt').get_parameters()

        self.focal = intrinsics.camera_matrix[0][0]
        self.H = intrinsics.height
        self.W = intrinsics.width

        # all needed attributes
        self.scene_id = scene_dict['scene_id']
        self.type = scene_dict['type']
        self.box_scale = scene_dict['box_scale']

        n_frames = len(self.images)

        # scene data
        self.near_plane = scene_dict['near_plane']
        self.far_plane = scene_dict['far_plane']

        # Get camera and vehicle poses:
        # Camera calibrations
        transform_matrix_z = np.array([
            [-1., 0., 0., 0.],
            [0., -1., 0., 0.],
            [0., 0., 1., 1.2],
            [0., 0., 0., 1.]
        ])
        veh2cam = extrinsics
        cam2veh_data = np.stack(veh2cam)
        timestamps = []
        frame_ids = []
        positions = []
        orientations = []
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
                    t1 = j-1
                    t2 = j
                    break
            if t1 == None and t2 == None:
                self.images = self.images[:i]
                self.point_cloud_pth = self.point_cloud_pth[:i]
                self.num_frames = i
                self.num_cam_frames = i
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
                pose[:3, :3] = np.dot(transform_matrix_z[:3, :3], quaternion_to_rotation_matrix(*desired_orientation))
                self.lidar_poses.append(pose)
                self.poses.append(pose@cam2veh_data)
                self.veh_pose = np.stack(self.poses)[:n_frames]

        self.poses = np.stack(self.poses)
        self.lidar_poses = np.stack(self.lidar_poses)
        print('Loaded Rosbag')

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    rotation_matrix = np.array([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy]
    ])

    return rotation_matrix
