import cv2
import numpy as np
from pathlib import Path

class KITTIDatasetCreator:
    def __init__(self, output_dir, camera_intrinsics):
        self.output_dir = Path(output_dir)
        self.camera_intrinsics = camera_intrinsics

        # Directories are per camera for images and calibration data
        self.cameras_dirs = {camera_id: {'images': self.output_dir / "images" / camera_id,
                                         'lidar': self.output_dir / "lidar" / camera_id,  # Assuming you want separate lidar data per camera
                                         'calib': self.output_dir / "calib" / camera_id}
                             for camera_id in camera_intrinsics.keys()}

        for dirs in self.cameras_dirs.values():
            for dir in dirs.values():
                dir.mkdir(parents=True, exist_ok=True)
        
        self._extrinsic_matrix = np.array([
            [0.151696, 0.988253, 0.0119597, -0.105466],
            [-0.00284016, 0.0126386, -0.999966, -0.139951],
            [-0.988267, 0.151624, 0.00612336, 0.32747],
            [0, 0, 0, 1]
        ])

    def create_dataset(self, synchronized_messages):
        for i, sync_msg in enumerate(synchronized_messages):
            self._save_image(sync_msg.image, sync_msg.camera, i)
            self._save_lidar_data(sync_msg.pointcloud, sync_msg.camera, i)  # Now includes camera_id
            self._save_calibration(sync_msg.camera, i)

    def _save_image(self, image, camera_id, index):
        filepath = self.cameras_dirs[camera_id]['images'] / f"{index:06}.png"
        cv2.imwrite(str(filepath), image)
        print(f"Saved image to {filepath}")

    def _save_lidar_data(self, pointcloud, camera_id, index):
        filepath = self.cameras_dirs[camera_id]['lidar'] / f"{index:06}.bin"
        pointcloud_np = np.array(pointcloud, dtype=np.float32)
        pointcloud_np.tofile(str(filepath))
        print(f"Saved LIDAR data to {filepath}")

    def _save_calibration(self, camera_id, index):
        params = self.camera_intrinsics[camera_id]
        filepath = self.cameras_dirs[camera_id]['calib'] / f"{camera_id}_{index:06}.txt"
        with open(filepath, 'w') as f:
            f.write(f"P0: {np.array2string(params.camera_matrix, separator=' ')[1:-1]}\n")
            f.write(f"Tr_velo_to_cam: {np.array2string(self._extrinsic_matrix, separator=' ')[1:-1]}\n")
        print(f"Saved calibration data to {filepath}")
