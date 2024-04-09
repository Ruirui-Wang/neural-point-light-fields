import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

extrinsics = np.array([
    [0.151696, 0.988253, 0.0119597, -0.105466],
    [-0.00284016, 0.0126386, -0.999966, -0.139951],
    [-0.988267, 0.151624, 0.00612336, 0.32747],
    [0, 0, 0, 1]
])


class LidarImageProjector:
    def __init__(self, camera_parameters):
        self._camera_parameters = camera_parameters
        self._extrinsic_matrix = extrinsics

    def visualize_lidar_on_images(self, synchronized_messages):
        for sync_msg in synchronized_messages:
            self._project_and_visualize(sync_msg)

    def _project_and_visualize(self, sync_msg):
        cam_params = self._camera_parameters[sync_msg.camera]
        image, depths = self._project_points_to_image(sync_msg.image, sync_msg.pointcloud, cam_params)
        self._display_image_with_depth_colorbar(image, depths, sync_msg.image_timestamp, sync_msg.pointcloud_timestamp)

    def _project_points_to_image(self, image, points, cam_params):
        camera_matrix = cam_params.camera_matrix
        distortion_coeffs = cam_params.distortion

        rotation_matrix = self._extrinsic_matrix[:3, :3]
        translation_vec = self._extrinsic_matrix[:3, 3]
        rotation_vec, _ = cv2.Rodrigues(rotation_matrix)

        points_array = np.array([(point.x, point.y, point.z) for point in points], dtype=np.float32)
        valid_depth_indices = (points_array[:, 0] <= -0.1) & (points_array[:, 0] >= -20)
        filtered_points_array = points_array[valid_depth_indices]

        points_2d, _ = cv2.projectPoints(filtered_points_array, rotation_vec, translation_vec, camera_matrix,
                                         distortion_coeffs)
        depths = filtered_points_array[:, 2]

        return self._draw_points_on_image(image, points_2d, depths), depths

    def _draw_points_on_image(self, image, points_2d, depths):
        image_flipped = cv2.flip(image, -1)
        min_depth, max_depth = np.min(depths), np.max(depths)
        normalized_depths = (depths - min_depth) / (max_depth - min_depth)
        colormap = cm.get_cmap('jet')
        colors = colormap(normalized_depths)[:, :3]

        for point, color in zip(points_2d, colors):
            if not np.any(np.isnan(point)) and not np.any(np.isinf(point)):
                point_int = tuple(point.ravel().astype(int))
                if 0 <= point_int[0] < image_flipped.shape[1] and 0 <= point_int[1] < image_flipped.shape[0]:
                    cv_color = tuple(int(c * 255) for c in color[::-1])
                    cv2.circle(image_flipped, point_int, 1, cv_color, thickness=1)

        return image_flipped

    def _display_image_with_depth_colorbar(self, image, depths, image_timestamp, pointcloud_timestamp, cmap='jet'):
        fig, ax = plt.subplots(figsize=(10, 7.5))
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        ax.set_title(f"Image Timestamp: {image_timestamp} | Pointcloud Timestamp: {pointcloud_timestamp}")

        norm = Normalize(vmin=np.min(depths), vmax=np.max(depths))
        smappable = ScalarMappable(norm=norm, cmap=cmap)
        smappable.set_array([])

        cbar = fig.colorbar(smappable, ax=ax, orientation='vertical', shrink=0.5, aspect=20, pad=0.02)
        cbar.set_label('Depth (m)')
        plt.show()
