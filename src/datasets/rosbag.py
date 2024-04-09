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
        image_directory = './data/images/basler00'
        image_paths = []
        for filename in os.listdir(image_directory):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(image_directory, filename)
                image_paths.append(image_path)
        self.images = image_paths

        pointcloud_directory = './data/lidar/basler00'
        point_cloud_paths = []
        for filename in os.listdir(pointcloud_directory):
            if filename.endswith('.bin') :
                point_cloud_path = os.path.join(pointcloud_directory, filename)
                point_cloud_paths.append(point_cloud_path)
        self.point_cloud_pth = point_cloud_paths

        self.num_cam_frames = len(self.images)
        assert np.round(len(self.images)/self.num_cameras)==len(self.images)/self.num_cameras
        self.num_frames = len(self.images)//self.num_cameras

        intrinsics = CameraIntrinsicHandler('./dataset/camera_calibration/calibration_basler00.txt').get_parameters()

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
        veh2cam = extrinsics
        cam2veh_data = np.stack(veh2cam)
        # TODO: POSES
        cam_poses = np.zeros((n_frames, self.num_cameras, 4, 4))
        cam_poses_world = np.zeros((n_frames, self.num_cameras, 4, 4))
        laser_poses = np.zeros((n_frames, self.num_cameras, 4, 4))
        lidar_poses_world = np.zeros((n_frames, self.num_cameras, 4, 4))
        veh2world = np.zeros((n_frames, self.num_cameras, 4, 4))
        self.poses = cam_poses.reshape(n_frames * self.num_cameras, 4, 4)

        self.poses_world = cam_poses_world.reshape(n_frames * self.num_cameras, 4, 4)

        self.lidar_poses = laser_poses.reshape(n_frames, 4, 4)
        self.lidar_poses_world = lidar_poses_world.reshape(n_frames, 4, 4)

        self.veh_pose = np.stack(veh2world)[:n_frames]

        print('Loaded Rosbag')
