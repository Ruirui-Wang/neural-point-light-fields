from dataclasses import dataclass
import numpy as np

@dataclass
class CameraParameters:
    width: int
    height: int
    camera_matrix: np.ndarray
    distortion: np.ndarray
    rectification: np.ndarray
    projection: np.ndarray

class CameraIntrinsicHandler:
    def __init__(self, filepath):
        self._filepath = filepath
        self._parameters = self._parse_file()

    def _parse_file(self):
        params = {
            'width': None,
            'height': None,
            'camera_matrix': None,
            'distortion': None,
            'rectification': None,
            'projection': None
        }

        def to_float_list(line):
            return np.array(list(map(float, line.strip().split())))

        with open(self._filepath, 'r') as file:
            lines = file.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line == 'width':
                    params['width'] = int(lines[i + 1].strip())
                    i += 2
                elif line == 'height':
                    params['height'] = int(lines[i + 1].strip())
                    i += 2
                elif line == 'camera matrix':
                    params['camera_matrix'] = np.array([
                        to_float_list(lines[i + 1]),
                        to_float_list(lines[i + 2]),
                        to_float_list(lines[i + 3])
                    ])
                    i += 4
                elif line == 'distortion':
                    params['distortion'] = to_float_list(lines[i + 1])
                    i += 2
                elif line == 'rectification':
                    params['rectification'] = np.array([
                        to_float_list(lines[i + 1]),
                        to_float_list(lines[i + 2]),
                        to_float_list(lines[i + 3])
                    ])
                    i += 4
                elif line == 'projection':
                    params['projection'] = np.array([
                        to_float_list(lines[i + 1]),
                        to_float_list(lines[i + 2]),
                        to_float_list(lines[i + 3])
                    ])
                    i += 4
                else:
                    i += 1

        return CameraParameters(
            width=params['width'],
            height=params['height'],
            camera_matrix=params['camera_matrix'],
            distortion=params['distortion'],
            rectification=params['rectification'],
            projection=params['projection']
        )

    def get_parameters(self):
        return self._parameters