from rosbags.serde import deserialize_cdr
import numpy as np

class ImageProcessor:
    def __init__(self):
        pass

    def process_image(self, rawdata, msgtype, topic):
        image_message = deserialize_cdr(rawdata, msgtype)
        encoding = 'bgr8'
        cv_image = np.frombuffer(image_message.data, dtype=np.uint8).reshape(image_message.height, image_message.width, -1)
        return cv_image, topic
