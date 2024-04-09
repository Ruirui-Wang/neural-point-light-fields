from dataclasses import dataclass
from rosbags.rosbag2 import Reader
from src.datasets.point_cloud_helpers import *
from src.datasets.image_processor import *
from src.datasets.point_cloud_processor import *
from src.datasets.ros_bag_processor import *

@dataclass
class SynchronizedMessage:
    image: np.ndarray
    pointcloud: np.ndarray
    image_timestamp: int
    pointcloud_timestamp: int
    camera: str

class BagFileProcessor:
    def __init__(self, bag_file_path):
        self.bag_file_path = bag_file_path
        self.image_processor = ImageProcessor()
        self.pointcloud_processor = PointCloudProcessor()
        self.image_data = {}
        self.pointcloud_data = {}
        self.camera_topics = {
            '/basler00/pylon_ros2_camera_node/image_raw': 'basler00',
            '/basler01/pylon_ros2_camera_node/image_raw': 'basler01'
        }
        self.limit = 1000 # FIXME: This is a temporary limit for development purposes

    def generate_synchronized_messages(self):
        self._process_bag_file()
        return self._synchronize_messages()

    def _process_bag_file(self):
        with Reader(self.bag_file_path) as reader:
            for connection, timestamp, rawdata in reader.messages():
                self.limit -= 1
                msgtype = connection.msgtype
                topic = connection.topic
                if msgtype.endswith('Image'):
                    image, topic = self.image_processor.process_image(rawdata, msgtype, topic)
                    self.image_data[timestamp] = (image, topic)
                elif msgtype.endswith('PointCloud2'):
                    points = self.pointcloud_processor.process_pointcloud(rawdata, msgtype)
                    self.pointcloud_data[timestamp] = points
                if self.limit == 0:
                    break

    def _find_closest_pointcloud_timestamp(self, image_timestamp):
        return min(self.pointcloud_data.keys(), key=lambda x: abs(x - image_timestamp), default=None)

    def _synchronize_messages(self):
        synchronized_messages = []
        for image_timestamp, (image, topic) in self.image_data.items():
            pointcloud_timestamp = self._find_closest_pointcloud_timestamp(image_timestamp)
            if pointcloud_timestamp:
                pointcloud = self.pointcloud_data[pointcloud_timestamp]
                synchronized_messages.append(SynchronizedMessage(
                    image=image,
                    pointcloud=pointcloud,
                    image_timestamp=image_timestamp,
                    pointcloud_timestamp=pointcloud_timestamp,
                    camera=self.camera_topics[topic]
                ))
        return synchronized_messages