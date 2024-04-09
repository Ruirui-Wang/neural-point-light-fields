from rosbags.serde import deserialize_cdr
from nerf.point_cloud_helpers import *

class PointCloudProcessor:
    @staticmethod
    def process_pointcloud(rawdata, msgtype):
        pointcloud_message = deserialize_cdr(rawdata, msgtype)
        points = read_points_list(cloud=pointcloud_message, skip_nans=True)
        return points