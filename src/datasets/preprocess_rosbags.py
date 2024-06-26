from src.datasets.camera_intrinsics_handler import CameraIntrinsicHandler
from src.datasets.ros_bag_processor import BagFileProcessor
from src.datasets.kitti_dataset_creator import KITTIDatasetCreator

bag_file_path = './dataset'

bag_file_processor = BagFileProcessor(bag_file_path)
synchronized_messages = bag_file_processor.generate_synchronized_messages()

camera_handler_basler00 = CameraIntrinsicHandler('dataset/camera_calibration/calibration_basler00.txt')
camera_handler_basler01 = CameraIntrinsicHandler('dataset/camera_calibration/calibration_basler01.txt')

camera_intrinsics = {
    'basler00': camera_handler_basler00.get_parameters(),
    'basler01': camera_handler_basler01.get_parameters()
}

output_path = "./data"
creator = KITTIDatasetCreator(output_path, camera_intrinsics)
creator.create_dataset(synchronized_messages)

