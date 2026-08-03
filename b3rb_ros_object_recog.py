# Copyright 2024-2026 NXP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np
import os

# We wrap the import in a try-except so the node still starts (and logs a
# clear reason why it's not detecting anything) if ultralytics isn't
# installed yet, rather than crashing on launch.
# Install with: pip install ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ------------------ TUNABLE PARAMETERS ------------------

# Path to your trained weights file. Ultralytics training runs save this at
# runs/detect/train/weights/best.pt by default - copy that file next to this
# script (or update this path) after training.
MODEL_WEIGHTS_FILENAME = 'best.pt'

# Minimum confidence to trust a detection enough to act on it. Too low and
# you'll get spurious/flickery sign calls; too high and you'll miss real
# signs at a distance or bad angle. Starting heuristic - tune against your
# own validation run.
CONFIDENCE_THRESHOLD = 0.60

# CLASS NAME MAPPING: your trained model's class names come from your
# dataset's data.yaml (whatever you named the classes when labeling in
# Roboflow) - they will NOT automatically be "LEFT"/"RIGHT"/"STRAIGHT".
# line_follower.py's sign_board_callback checks for "LEFT"/"RIGHT"/"STRAIGHT"
# (as substrings, case-insensitive) in whatever string this node publishes.
# Fill this in with your actual class names once you know them - print
# self.model.names after loading (see the log line in __init__ below) to see
# exactly what your model calls each class.
# In b3rb_ros_object_recog_2.py

CLASS_NAME_MAP = {
    'A': 'A', 'B': 'B', 'C': 'C',
    'X': 'X', 'Y': 'Y', 'Z': 'Z',
    'Left': 'LEFT', 'Right': 'RIGHT', 'Straight': 'STRAIGHT'
}

class ObjectRecognizer(Node):
    """
    ROS 2 Node that processes raw camera images to recognize traffic sign
    boards using a YOLOv8 detector, and publishes the detected sign type on
    the `/sign_board_detection` topic (which line_follower.py already
    subscribes to via sign_board_callback).
    """
    def __init__(self):
        super().__init__('object_recognizer')

        # Subscription for camera images.
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10)

        # Publisher for sign board detection results.
        self.publisher_sign = self.create_publisher(
            String,
            '/sign_board_detection',
            10)

        # Attempt to load the trained YOLOv8 model located next to this file.
        self.model = None
        if YOLO is not None:
            dir_path = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(dir_path, MODEL_WEIGHTS_FILENAME)
            if os.path.exists(model_path):
                try:
                    self.model = YOLO(model_path)
                    self.get_logger().info(f"Loaded YOLO model from {model_path}")
                    # This prints exactly what your model calls each class -
                    # use it to fill in CLASS_NAME_MAP above correctly.
                    self.get_logger().info(f"Model class names: {self.model.names}")
                except Exception as e:
                    self.get_logger().error(f"Failed to load YOLO model: {e}")
            else:
                self.get_logger().warn(f"Model file not found at {model_path}")
        else:
            self.get_logger().warn(
                "ultralytics is not installed (pip install ultralytics). "
                "Object recognizer will not detect anything until it is."
            )

        self.get_logger().info("Object Recognizer Node started. Waiting for images...")

    def camera_image_callback(self, message):
        """Processes incoming camera frames to classify traffic signs."""
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return

        sign_detected = self.classify_sign(image)

        if sign_detected is not None:
            msg = String()
            msg.data = sign_detected
            self.publisher_sign.publish(msg)
            self.get_logger().info(f"Detected Sign Board: {sign_detected}")



    def classify_sign(self, image):
        """
        Detects multiple classes, separates targets (letters) from directions,
        and pairs them based on physical proximity in the bounding boxes.
        Publishes a formatted string like: "A:LEFT, B:STRAIGHT"
        """
        if self.model is None:
            return None

        results = self.model(image, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes
        letters = []
        directions = []

        # 1. Gather all confident detections and their center coordinates
        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            if conf < CONFIDENCE_THRESHOLD:
                continue
                
            class_idx = int(boxes.cls[i])
            class_name = self.model.names[class_idx]
            mapped = CLASS_NAME_MAP.get(class_name)
            
            if not mapped:
                continue

            # Calculate the center (x, y) of the bounding box
            x1, y1, x2, y2 = boxes.xyxy[i]
            cx, cy = float((x1 + x2) / 2), float((y1 + y2) / 2)

            if mapped in ['LEFT', 'RIGHT', 'STRAIGHT']:
                directions.append((mapped, cx, cy))
            else:
                letters.append((mapped, cx, cy))

        if not letters or not directions:
            return None  # We need at least one letter and one direction to make a pair

        # 2. Pair each letter with the arrow physically closest to it
        pairs = []
        for l_txt, lx, ly in letters:
            # Find the direction box with the shortest distance to this letter
            closest_dir = min(directions, key=lambda d: (d[1]-lx)**2 + (d[2]-ly)**2)
            pairs.append(f"{l_txt}:{closest_dir[0]}")

        # Returns a string like "A:LEFT,B:STRAIGHT"
        return ",".join(pairs) if pairs else None


def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()