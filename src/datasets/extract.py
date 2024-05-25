import csv
import struct
from datetime import datetime

import cv2
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr

import numpy as np
import open3d as o3d

ros_type_to_struct_type = {
    1: 'b',   # INT8
    2: 'B',   # UINT8
    3: 'h',   # INT16
    4: 'H',   # UINT16
    5: 'i',   # INT32
    6: 'I',   # UINT32
    7: 'f',   # FLOAT32
    8: 'd'    # FLOAT64
}

def extract_point_cloud_data(msg):
    point_step = msg.point_step
    row_step = msg.row_step
    fields = msg.fields

    # Extract point cloud data from msg.data
    raw_data = msg.data.astype(np.uint8)
    num_points = len(raw_data)//point_step
    point_cloud_data = np.zeros((num_points, len(fields)))

    for i in range(num_points):
        point_start = i * point_step
        for j, field in enumerate(fields):
            data_type_str = ros_type_to_struct_type[field.datatype]
            data_type_size = struct.calcsize(data_type_str)
            field_data = raw_data[point_start + field.offset: point_start + field.offset + field.count * data_type_size]
            values = struct.unpack(f"{field.count}{data_type_str}", field_data)
            point_cloud_data[i, j:j+field.count] = values

    return point_cloud_data

def extract_image_data(msg):
    cv_image = np.ndarray(shape=(msg.height, msg.width, 3), dtype=np.uint8, buffer=msg.data)
    return cv_image
'''    cv2.imwrite(f"image_{timestamp}.jpg", cv_image)
'''




import os

synchronized_msg = []
len_sync_msg = 0
os.makedirs("../../data", exist_ok=False)
os.makedirs("../../data/images", exist_ok=True)
os.makedirs("../../data/images/basler00", exist_ok=True)
os.makedirs("../../data/images/basler01", exist_ok=True)
os.makedirs("../../data/lidar", exist_ok=True)
os.makedirs("../../data/lidar/basler00", exist_ok=True)
os.makedirs("../../data/lidar/basler01", exist_ok=True)
with open('../../data/pose_data.csv', 'w') as csvfile:
    pass
with open('../../data/timestamp_data.csv', 'w') as csvfile:
    pass

with Reader('../../dataset') as reader:
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == '/herbie/observations/velodyne_points':
            previous_pointcloud_msg = deserialize_cdr(rawdata, connection.msgtype)
            previous_pointcloud_timestamp = timestamp
        elif connection.topic == '/basler00/pylon_ros2_camera_node/image_raw':
            image_msg = deserialize_cdr(rawdata, connection.msgtype)
            image = extract_image_data(image_msg)
            closest_timestamp = previous_pointcloud_timestamp if previous_pointcloud_timestamp else float('inf')
            # Check the next pointcloud message
            for next_connection, next_timestamp, next_rawdata in reader.messages():
                if next_connection.topic == '/herbie/observations/velodyne_points':
                    if abs(closest_timestamp - timestamp) > abs(next_timestamp - timestamp):
                        pointcloud_msg = deserialize_cdr(rawdata, connection.msgtype)
                        pointcloud = extract_point_cloud_data(pointcloud_msg)
                        synchronized_msg.append({
                            'image_timestamp': timestamp,
                            'image': image,
                            'pointcloud_timestamp': next_timestamp,
                            'pointcloud': pointcloud,
                            'camera': '00'
                        })
                    else:
                        previous_pointcloud = extract_point_cloud_data(previous_pointcloud_msg)
                        synchronized_msg.append({
                            'image_timestamp': timestamp,
                            'image': image,
                            'pointcloud_timestamp': closest_timestamp,
                            'pointcloud': previous_pointcloud,
                            'camera': '00'
                        })
                    image_filename = os.path.join('../../data/images/basler00', f"{synchronized_msg[-1]['image_timestamp']}.jpg")
                    cv2.imwrite(image_filename, synchronized_msg[-1]['image'])
                    file_path = os.path.join('../../data/lidar/basler00', f"{synchronized_msg[-1]['pointcloud_timestamp']}.txt")
                    np.savetxt(file_path, synchronized_msg[-1]['pointcloud'])
                    break
        elif connection.topic == '/basler01/pylon_ros2_camera_node/image_raw':
            image_msg = deserialize_cdr(rawdata, connection.msgtype)
            image = extract_image_data(image_msg)
            closest_timestamp = previous_pointcloud_timestamp if previous_pointcloud_timestamp else float('inf')
            # Check the next pointcloud message
            for next_connection, next_timestamp, next_rawdata in reader.messages():
                if next_connection.topic == '/herbie/observations/velodyne_points':
                    if abs(closest_timestamp - timestamp) > abs(next_timestamp - timestamp):
                        if abs(next_timestamp - timestamp) < 5e6: # 5ms
                            print(abs(next_timestamp - timestamp))
                            pointcloud_msg = deserialize_cdr(rawdata, connection.msgtype)
                            pointcloud = extract_point_cloud_data(pointcloud_msg)
                            synchronized_msg.append({
                                'image_timestamp': timestamp,
                                'image': image,
                                'pointcloud_timestamp': next_timestamp,
                                'pointcloud': pointcloud,
                                'camera': '01'
                            })
                    else:
                        if abs(closest_timestamp - timestamp) < 5e6: # 5ms
                            print(abs(closest_timestamp - timestamp))
                            previous_pointcloud = extract_point_cloud_data(previous_pointcloud_msg)
                            synchronized_msg.append({
                                'image_timestamp': timestamp,
                                'image': image,
                                'pointcloud_timestamp': closest_timestamp,
                                'pointcloud': previous_pointcloud,
                                'camera': '01'
                            })
                    if len(synchronized_msg) > 0 and len_sync_msg!= len(synchronized_msg):
                        len_sync_msg = len(synchronized_msg)
                        image_filename = os.path.join('../../data/images/basler01', f"{len_sync_msg-1:06}.png")
                        cv2.imwrite(image_filename, synchronized_msg[-1]['image'])
                        print("image",image_filename)
                        file_path = os.path.join('../../data/lidar/basler01', f"{len_sync_msg-1:06}.txt")
                        np.savetxt(file_path, synchronized_msg[-1]['pointcloud'])
                        print("pointcloud", file_path)
                        with open('../../data/timestamp_data.csv', 'a') as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(
                                [f"{len_sync_msg-1:06}", synchronized_msg[-1]['pointcloud_timestamp']])
                    break
        elif connection.topic == '/herbie/observations/pose':
            msg = deserialize_cdr(rawdata, connection.msgtype)
            stamp = msg.header.stamp
            frame_id = msg.header.frame_id
            position = msg.pose.position
            orientation = msg.pose.orientation
            with open('../../data/pose_data.csv', 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([stamp.sec, stamp.nanosec, frame_id, position.x, position.y, position.z, orientation.x,
                                 orientation.y, orientation.z, orientation.w])












