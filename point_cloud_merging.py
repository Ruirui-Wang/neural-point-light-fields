import copy
import csv
import open3d as o3d
from matplotlib import pyplot as plt
import os
import numpy as np

timestamps = []
frame_ids = []
positions = []
orientations = []
point_clouds = []
file_names = []
transform_matrix = np.array([[1., 0., 0., 0.],
                             [0., 1., 0., 0.],
                             [0., 0., 1., 1.2],
                             [0., 0., 0., 1.]])
transform_matrix_z = np.array([
    [-1., 0., 0., 0.],
    [0., -1., 0., 0.],
    [0., 0., 1., 1.2],
    [0., 0., 0., 1.]
])



def read_calibration_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    K = []
    D = []
    extract_camera_matrix = False
    extract_distortion = False

    for line in lines:
        if line.strip() == 'camera matrix':
            extract_camera_matrix = True
        elif line.strip() == 'distortion':
            extract_distortion = True
        elif extract_camera_matrix == True:
            K.append([float(x) for x in line.strip().split()])
            if len(K) >= 3:
                extract_camera_matrix = False
        elif extract_distortion == True:
            D = ([float(x) for x in line.strip().split()])
            extract_distortion = False
    return K, D

K, D = read_calibration_file("ost.txt")

M2 = np.array([[0.151696,0.988253,0.0119597,-0.105466],
[-0.00284016,0.0126386,-0.999966,-0.139951],
[-0.988267,0.151624,0.00612336,0.32747],
[0,0,0,1]], dtype=np.float32)
M1 = np.eye(4)
M1[:3, :3] = K


def colorize_point_cloud(points,M1, M2):
    resolution = [600,800]
    coords = points[:, :3]
    ones = np.ones(len(coords)).reshape(-1, 1)
    coords = np.concatenate([coords, ones], axis=1)
    original = coords
    transform = copy.deepcopy(M1 @ M2).reshape(4, 4)
    coords = coords @ transform.T
    coords = np.hstack((coords, np.arange(len(coords)).reshape(-1, 1)))
    coords = coords[np.where(coords[:, 2] > 0)]

    coords[:, 0] /= coords[:, 2]
    coords[:, 1] /= coords[:, 2]

    coords = coords[np.where(coords[:, 2] > 1e0)]
    coords = coords[np.where(coords[:, 2] < 1e2)]

    valid_indices = []
    for coord in coords:
        x, y, z = coord[:3]
        index = int(coord[4])
        u = int(x)
        v = int(y)
        if 0 <= u < resolution[1] and 0 <= v < resolution[0]:
            valid_indices.append(index)
    filtered_coords = original[valid_indices]

    return filtered_coords



def read_point_cloud_folder(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            file_path = os.path.join(folder_path, file_name)
            point_cloud = np.loadtxt(file_path)
            point_cloud = point_cloud[np.where(point_cloud[:, 2] >= -0.9)]
            point_cloud = point_cloud[np.where(point_cloud[:, 2] <= 1.8)]
            point_clouds.append(point_cloud)
            file_names.append(file_name[:-4])
    return point_clouds, file_names

folder_path = 'data/pointcloud'
point_clouds, file_names = read_point_cloud_folder(folder_path)


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    rotation_matrix = np.array([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy]
    ])

    return rotation_matrix


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

point_cloud = o3d.geometry.PointCloud()
trajectory = o3d.geometry.LineSet()

for i in range(len(point_clouds)):
    pose = np.eye(4)
    mark = None
    t1 = None
    t2 = None
    for j in range(len(positions)):
        if abs(timestamps[j]*1e2-int(file_names[i])//10**7) <=100:
            print(timestamps[j] * 1e9 - int(file_names[i]))
            if timestamps[j] * 1e9 - int(file_names[i])>=0:
                t1 = mark
                t2 = j
                break
            mark = j
    if t1 == None and t2 == None:
        break
    alpha = (int(file_names[i]) - timestamps[t1]*1e9) / (timestamps[t2]*1e9 - timestamps[t1]*1e9)
    desired_position = (
        positions[t1][0] + alpha * (positions[t2][0] - positions[t1][0]),
        positions[t1][1] + alpha * (positions[t2][1] - positions[t1][1]),
        positions[t1][2] + alpha * (positions[t2][2] - positions[t1][2])
    )
    from scipy.spatial.transform import Rotation

    r1 = Rotation.from_quat([orientations[t1][0], orientations[t1][1], orientations[t1][2], orientations[t1][3]])
    r2 = Rotation.from_quat([orientations[t2][0], orientations[t2][1], orientations[t2][2], orientations[t2][3]])
    print(orientations[t1],orientations[t2])
    q1 = orientations[t1]
    q2 = orientations[t2]
    desired_orientation = (
        orientations[t1][0] + alpha * (orientations[t2][0] - orientations[t1][0]),
        orientations[t1][1] + alpha * (orientations[t2][1] - orientations[t1][1]),
        orientations[t1][2] + alpha * (orientations[t2][2] - orientations[t1][2]),
        orientations[t1][3] + alpha * (orientations[t2][3] - orientations[t1][3])
    )
    pose[:3, 3] = desired_position
    pose[:3, :3] = np.dot(transform_matrix_z[:3,:3],quaternion_to_rotation_matrix(*desired_orientation))

    print(pose)
    transformed_point_cloud = np.dot(pose[:3, :3],point_clouds[i][:, 0:3].T ) + pose[:3, 3][:, np.newaxis]

    if i %1 == 0:
        point_cloud.points.extend(transformed_point_cloud.T)

for i in range(len(positions)):
    trajectory.points.append(positions[i])
    if i < len(positions) - 1 and i>0:
        trajectory.lines.append([i, i + 1])

scene = o3d.visualization.Visualizer()
scene.create_window()

scene.add_geometry(point_cloud)
'''scene.add_geometry(trajectory)
'''
render_option = scene.get_render_option()
render_option.point_size = 1

scene.run()
scene.destroy_window()

quaternions = orientations
quaternions /= np.linalg.norm(quaternions, axis=1)[:, np.newaxis]  # Normalize quaternions

fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
components = ['x', 'y', 'z', 'w']
# Plot each component of the quaternion against the index
for i in range(4):
    axes[i].plot(range(len(quaternions)), quaternions[:, i])
    axes[i].set_ylabel(f'Component {components[i]}')
axes[3].set_xlabel('Index')
plt.show()



euler_angles = np.zeros((len(orientations), 3))
for i, quaternion in enumerate(orientations):
    r = Rotation.from_quat(quaternion)
    euler_angles[i] = r.as_euler('xyz')

fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
components = ['x', 'y', 'z']
# Plot each component of the Euler angles against the index
for i in range(3):
    axes[i].plot(range(len(euler_angles)), euler_angles[:, i])
    axes[i].set_ylabel(f'Euler Angle {components[i]}')
axes[2].set_xlabel('Index')
plt.show()





