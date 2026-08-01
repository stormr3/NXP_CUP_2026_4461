
# Copyright 2024-2026 NXP
# Copyright 2016 Open Source Robotics Foundation, Inc.
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
import time
import math
from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import String
from synapse_msgs.msg import EdgeVectors, ServerCommunication

QOS_PROFILE_DEFAULT = 10
PI = math.pi

# Control bounds
SPEED_MIN = 0.0
SPEED_MAX = 1.0
TURN_MIN = -1.0
TURN_MAX = 1.0

# CONFIGURATION:
# The buggy is driven in manual mode by publishing standard controller Joy messages to /cerebri/in/joy.
# The layout is: msg.axes = [0.0, speed, 0.0, turn]
# - speed: positive for forward, negative for reverse. Range: [-1.0, 1.0]
# - turn: positive for left steer, negative for right steer. Range: [-1.0, 1.0]
# msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1] (Keep buttons set to this pattern for manual override mode)

# NOTE on vector_1 / vector_2: each is geometry_msgs/Point[2] - a FIXED-size
# array of 2 points, always present regardless of vector_count. There's no
# documented near/far ordering for the two points, so every calculation below
# uses the AVERAGE of both points' x as "where this line sits horizontally" -
# this is exactly what your original working code did, and it doesn't depend
# on any assumption about point ordering.

# ------------------ TUNABLE PARAMETERS ------------------
# These don't touch the control bounds above - they're gains/speeds/thresholds
# for the two driving modes below.

# Which mode runs on boot: "LANE_FOLLOW" (double solid lines) or
# "LINE_FOLLOW" (single committed line, for intersections). Change this
# constant and restart the node to switch which one you're testing.
DEFAULT_DRIVE_MODE = "LANE_FOLLOW"

# Only used when DEFAULT_DRIVE_MODE == "LINE_FOLLOW": which line to commit to.
DEFAULT_FOLLOW_SIDE = "LEFT"   # "LEFT" or "RIGHT"

# Same Kp and offset ratio as your original working lane follower - unchanged.
STEER_KP = 0.004
LANE_GAP_OFFSET_RATIO = 0.4

# If the two detected boundaries' midpoints are farther apart than this
# fraction of image width, they're too far apart to be a real parallel lane
# pair - almost certainly one of them is a foreign edge (e.g. the near curb
# of a cross-street revealed mid-turn), not your actual right/left boundary.
# Heuristic starting point, not measured against your sim - log the real
# track_width value during a tight-turn run and adjust if this fires too
# early/late.
TRACK_WIDTH_DIVERGENCE_RATIO = 1

# LANE_FOLLOW speeds
LANE_SPEED_TWO_LINES = 0.70    # confident: both boundaries agree on a normal-width track
LANE_SPEED_ONE_LINE = 0.35     # only one boundary in play (either genuinely one visible, or divergence fallback)
LANE_SPEED_LOST_SHORT = 0.30   # briefly no boundary at all: hold last turn, ease off speed
LANE_SPEED_LOST_LONG = 0.15    # lost for a while: crawl instead of driving blind at speed
LANE_LOST_GRACE_FRAMES = 5     # frames before dropping from LOST_SHORT to LOST_LONG speed
LANE_APEX_BLIND_GRACE_FRAMES = 10  # frames to keep sustaining the apex turn after the line vanishes mid-apex, before giving up and falling back to generic lost-line handling

# LINE_FOLLOW speeds/behavior (single committed line, e.g. through an intersection)
LINE_GAP_PX = 40.0             # if committed line's midpoint crosses within this many px of center, we're driving over it
LINE_TURN_HOLD = 0.45          # steer magnitude to hold a turn arc toward the committed line
LINE_TURN_EASE = 0.20          # reduced steer magnitude once we've drifted on top of the line
LINE_SPEED_TRACK = 0.70        # speed while the committed line is visible and tracked
LINE_SPEED_BLIND = 0.35        # speed while the committed line has vanished (e.g. turn apex) and we're sweeping to reacquire it

# Obstacle avoidance tuning
OBSTACLE_DISTANCE_THRESHOLD = 0.65    # meters; trigger avoidance below this range
FRONT_SECTOR_START_FRAC = 7 / 18     # start of the front sector, as a fraction of the full 360 scan
FRONT_SECTOR_END_FRAC = 11 / 18      # end of the front sector, as a fraction of the full 360 scan
AVOID_TURN = 0.3
AVOID_SPEED = 0.3

class LineFollower(Node):
    """
    Core controller Node for the B3RB buggy.
    Two selectable driving modes (see drive_mode / follow_side):
      - LANE_FOLLOW: your original centering/single-line logic, extended with
        a fallback for when the two detected boundaries diverge too far to be
        a real lane pair (tight-turn cross-edge case).
      - LINE_FOLLOW: commits to ONE line and holds a gap to it; if it vanishes
        (turn apex), holds a turn arc toward it until reacquired.
    """
    def __init__(self):
        super().__init__('line_follower')

        # ------------------ Subscriptions ------------------

        # 1. Lane Edge Vectors (from edge_vectors_publisher)
        self.subscription_vectors = self.create_subscription(
            EdgeVectors,
            '/edge_vectors',
            self.edge_vectors_callback,
            QOS_PROFILE_DEFAULT)

        # 2. LIDAR Obstacle Scanner
        self.subscription_lidar = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            QOS_PROFILE_DEFAULT)

        # 3. Server Communication Feedback Loop
        self.subscription_server = self.create_subscription(
            ServerCommunication,
            '/ServerCommunication',
            self.server_communication_callback,
            QOS_PROFILE_DEFAULT)

        # 4. QR Code Detections (from qr_detector)
        self.subscription_qr = self.create_subscription(
            String,
            '/qr_detection',
            self.qr_detection_callback,
            QOS_PROFILE_DEFAULT)

        # 5. Sign Board Detections (from object_recognizer)
        self.subscription_signs = self.create_subscription(
            String,
            '/sign_board_detection',
            self.sign_board_callback,
            QOS_PROFILE_DEFAULT)

        # ------------------ Publishers ------------------

        # Publisher to drive/steer the buggy
        self.publisher_joy = self.create_publisher(
            Joy,
            '/cerebri/in/joy',
            QOS_PROFILE_DEFAULT)

        # Publisher to send messages to the Server
        self.publisher_server = self.create_publisher(
            ServerCommunication,
            '/ServerCommunication',
            QOS_PROFILE_DEFAULT)

        # ------------------ State Variables & Timer ------------------

        # Default controls: drive straight slowly
        self.target_speed = 0.15
        self.target_turn = 0.0

        # State variables (You can add your own state flags / state machines here)
        self.obstacle_in_front = False
        self.patient_id = None
        self.hospital_id = None
        self.current_destination = None
        self.mission_completed = False
        # Racing line / late apex entry tracking
        self.horizontal_line_frames = 0
        self.apex_active = False   # True if the line vanished WHILE it was in apex/horizontal orientation
        self.apex_turn = 0.0       # the turn command to sustain while apex_active
        self.apex_blind_frames = 0

        # ===== SERVER COMMUNICATION STATE (added) =====
        # Translate between the server's single-letter codes and building names.
        self.sign_to_building = {
            'A': 'PATIENT_1', 'B': 'PATIENT_2', 'C': 'PATIENT_3',
            'X': 'HOSPITAL_1', 'Y': 'HOSPITAL_2', 'Z': 'HOSPITAL_3',
        }
        self.building_to_sign = {v: k for k, v in self.sign_to_building.items()}
        self.server_uid = 0          # rolling message id, 0-255
        self.last_sent_qr = None     # avoid re-registering the same building repeatedly
        self.awaiting_hospital = False
        # ===== END SERVER STATE =====

        # ---- Driving mode ----
        # HOW TO SWITCH MODES FOR TESTING: change DEFAULT_DRIVE_MODE (and
        # DEFAULT_FOLLOW_SIDE if testing LINE_FOLLOW) at the top of this file
        # and restart the node. Later, sign_board_callback can set
        # self.drive_mode / self.follow_side directly instead.
        self.drive_mode = DEFAULT_DRIVE_MODE      # "LANE_FOLLOW" or "LINE_FOLLOW"
        self.follow_side = DEFAULT_FOLLOW_SIDE    # "LEFT" or "RIGHT" (LINE_FOLLOW only)

        # LANE_FOLLOW bookkeeping
        self.lane_lost_frame_count = 0
        self.last_target_x = None  # last commanded target_x, used to keep the
                                    # divergence fallback picking the SAME
                                    # boundary frame-to-frame instead of
                                    # potentially flip-flopping between the two

        # LINE_FOLLOW bookkeeping
        self.line_blind_frame_count = 0

        # Timer to publish drive commands at 10Hz
        self.control_timer = self.create_timer(0.1, self.publish_drive_commands)

        self.get_logger().info(
            f"Line Follower controller initialized. drive_mode={self.drive_mode}"
            + (f" follow_side={self.follow_side}" if self.drive_mode == "LINE_FOLLOW" else "")
        )

    def publish_drive_commands(self):
        """Timer callback that periodically publishes the current speed and steer command."""
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]  # Manual override button configuration
        msg.axes = [0.0, self.target_speed, 0.0, self.target_turn]
        self.publisher_joy.publish(msg)

    def rover_move_manual_mode(self, speed, turn):
        """Helper to immediately set control speed and steering angle."""
        self.target_speed = float(max(min(speed, SPEED_MAX), -SPEED_MAX))
        self.target_turn = float(max(min(turn, TURN_MAX), -TURN_MAX))

    # ------------------ Mode: LANE_FOLLOW ------------------

    def _handle_lane_follow(self, message, width, half_width):
        count = message.vector_count

        if count == 2:
            # Same midpoint math as your original working code.
            v1_mid_x = (message.vector_1[0].x + message.vector_1[1].x) / 2.0
            v2_mid_x = (message.vector_2[0].x + message.vector_2[1].x) / 2.0
            track_width = abs(v1_mid_x - v2_mid_x)

            if track_width <= (width * TRACK_WIDTH_DIVERGENCE_RATIO):
                # Normal case: genuine parallel lane pair -> center between them.
                target_x = (v1_mid_x + v2_mid_x) / 2.0
                self.target_speed = LANE_SPEED_TWO_LINES
            else:
                # Divergence too large to be a real lane pair. Treat this as a
                # single-line situation: pick whichever of the two midpoints
                # is closer to where we were just tracking (so we don't
                # suddenly jump to whichever one happens to be first), and
                # ignore the other one entirely.
                if self.last_target_x is None:
                    chosen_mid_x = min(v1_mid_x, v2_mid_x, key=lambda x: abs(x - half_width))
                else:
                    chosen_mid_x = min(v1_mid_x, v2_mid_x, key=lambda x: abs(x - self.last_target_x))

                offset = width * LANE_GAP_OFFSET_RATIO
                if chosen_mid_x < half_width:
                    target_x = chosen_mid_x + offset
                else:
                    target_x = chosen_mid_x - offset
                self.target_speed = LANE_SPEED_ONE_LINE

            error = half_width - target_x
            self.target_turn = max(-1.0, min(1.0, STEER_KP * error))
            self.last_target_x = target_x
            self.lane_lost_frame_count = 0

        elif count == 1:
            p0 = message.vector_1[0]
            p1 = message.vector_1[1]

            dx = abs(p0.x - p1.x)
            dy = abs(p0.y - p1.y)

            # Detect horizontal/perpendicular line across camera view
            if dx > (dy * 3):
                self.horizontal_line_frames += 1
                line_center_x = (p0.x + p1.x) / 2.0

                if line_center_x >= half_width:
                    # PHASE 2: Hit the Apex! (Hard Left Turn once track opens up)
                    self.target_turn = (0.10+dx/dy*0.05) 
                    self.target_speed = 0.70
                else:
                    # PHASE 2: Hit the Apex! (Hard Right Turn)
                    self.target_turn = -(0.10+dx/dy*0.05) 
                    self.target_speed = 0.70

                # Remember this as the committed apex turn - if the line
                # vanishes entirely on the very next frame (very common right
                # at the apex), we want to keep applying THIS turn rather
                # than falling into the generic "no idea, hold whatever was
                # last set and slow down" fallback below.
                self.apex_active = True
                self.apex_turn = self.target_turn
                self.apex_blind_frames = 0
            else:
                # Reset counter when line is back to normal vertical orientation
                self.horizontal_line_frames = 0
                self.apex_active = False  # back to normal tracking, not mid-apex anymore

                # Normal single vertical line tracking
                v1_mid_x = (p0.x + p1.x) / 2.0
                offset = width * LANE_GAP_OFFSET_RATIO

                if v1_mid_x < half_width:
                    target_x = v1_mid_x + offset
                else:
                    target_x = v1_mid_x - offset

                error = half_width - target_x
                self.target_turn = max(-1.0, min(1.0, (STEER_KP) * error))
                self.target_speed = LANE_SPEED_ONE_LINE

            self.last_target_x = (p0.x + p1.x) / 2.0
            self.lane_lost_frame_count = 0

        else:
            # No boundary visible at all.
            if self.apex_active and self.apex_blind_frames < LANE_APEX_BLIND_GRACE_FRAMES:
                # We were mid-apex (line horizontal) and it just vanished -
                # this is the exact case you asked about. Keep sustaining
                # THAT turn at a real driving speed (not a crawl) for a
                # short grace period, since we expect to sweep back onto the
                # line shortly, rather than treating this as a generic
                # "totally lost" event.
                self.apex_blind_frames += 1
                self.target_turn = self.apex_turn
                self.target_speed = LANE_SPEED_ONE_LINE
                self.lane_lost_frame_count = 0
            else:
                # Either we weren't mid-apex, or the apex sweep has gone on
                # longer than expected and something else is wrong - fall
                # back to holding the last steering command and easing off
                # speed the longer this persists.
                self.apex_active = False
                self.lane_lost_frame_count += 1
                self.target_speed = (LANE_SPEED_LOST_SHORT
                         if self.lane_lost_frame_count <= LANE_LOST_GRACE_FRAMES
                         else LANE_SPEED_LOST_LONG)
            # last_target_x intentionally left unchanged - preserves
            # continuity for when a boundary reappears.

    # ------------------ Mode: LINE_FOLLOW ------------------

    def _handle_line_follow(self, message, width, half_width):
        count = message.vector_count
        sign = 1.0 if self.follow_side == "LEFT" else -1.0

        candidates = []
        if count >= 1:
            candidates.append(message.vector_1)
        if count >= 2:
            candidates.append(message.vector_2)

        # Among detected boundaries, find the one on the side we've committed
        # to. Deliberately NOT "whichever is closer" - once committed at a
        # split, stick to that side even if the other briefly appears too, so
        # we don't flip-flop mid-intersection.
        target_vec = None
        for v in candidates:
            mid_x = (v[0].x + v[1].x) / 2.0
            if self.follow_side == "LEFT" and mid_x < half_width:
                target_vec = v
                break
            if self.follow_side == "RIGHT" and mid_x >= half_width:
                target_vec = v
                break

        if target_vec is not None:
            p0, p1 = target_vec[0], target_vec[1]
            dx = abs(p0.x - p1.x)
            dy = abs(p0.y - p1.y)
            mid_x = (p0.x + p1.x) / 2.0

            if dx > dy * 3:
                # Same apex signal as LANE_FOLLOW's count==1 case: the
                # committed line itself has gone near-horizontal, meaning
                # we're mid-apex. Hold a firm turn toward our committed side.
                self.target_turn = sign * min(1.0, 0.1 + dx / dy * 0.1)
                self.target_speed = LINE_SPEED_TRACK
            else:
                # Normal case: proportional gap-hold, identical in kind to
                # the count==1 lane-follow logic - this is what actually
                # fixes the "always circling" bug, since it goes to ~0 turn
                # when we're already at the target gap instead of always
                # commanding a fixed turn magnitude.
                offset = width * LANE_GAP_OFFSET_RATIO
                if mid_x < half_width:
                    target_x = mid_x + offset
                else:
                    target_x = mid_x - offset
                error = half_width - target_x
                self.target_turn = max(-1.0, min(1.0, STEER_KP * error))
                self.target_speed = LINE_SPEED_TRACK

            self.line_blind_frame_count = 0
        else:
            # Committed line isn't visible this frame (e.g. turn apex). Keep
            # sweeping in the committed direction at reduced speed until it
            # reappears, rather than going straight/blind.
            self.line_blind_frame_count += 1
            self.target_turn = sign * LINE_TURN_HOLD
            self.target_speed = LINE_SPEED_BLIND

    # ------------------ Callback Implementations ------------------

    def edge_vectors_callback(self, message):
        if self.obstacle_in_front:
            # LIDAR has priority while dodging; don't fight the avoidance maneuver.
            return

        """
        Receives lane boundaries from the camera vector extractor and
        dispatches to whichever mode is active (see drive_mode / follow_side).
        """
        width = float(message.image_width)
        if width <= 0:
            return  # guard against a boot/glitch frame with no valid image_width
        half_width = width / 2.0

        if self.drive_mode == "LANE_FOLLOW":
            self._handle_lane_follow(message, width, half_width)
        elif self.drive_mode == "LINE_FOLLOW":
            self._handle_line_follow(message, width, half_width)

    def lidar_callback(self, message):
        """
        Receives LIDAR range measurements and steers around obstacles that
        appear in the front sector, deferring to lane-following otherwise.
        """
        num_readings = len(message.ranges)
        if num_readings == 0:
            return

        front_start = int(num_readings * FRONT_SECTOR_START_FRAC)
        front_end = int(num_readings * FRONT_SECTOR_END_FRAC)
        front_sector = message.ranges[front_start:front_end]

        def valid(ranges):
            return [r for r in ranges if message.range_min <= r <= message.range_max]

        front_valid = valid(front_sector)
        min_front_dist = min(front_valid) if front_valid else float('inf')

        self.obstacle_in_front = min_front_dist < OBSTACLE_DISTANCE_THRESHOLD
        if not self.obstacle_in_front:
            return

        mid = len(front_sector) // 2
        right_valid = valid(front_sector[:mid])
        left_valid = valid(front_sector[mid:])

        left_clearance = min(left_valid) if left_valid else float('inf')
        right_clearance = min(right_valid) if right_valid else float('inf')

        turn = AVOID_TURN if left_clearance >= right_clearance else -AVOID_TURN
        self.rover_move_manual_mode(AVOID_SPEED, turn)

    def server_communication_callback(self, message):
        """
        Receives coordination commands from the server.

        Protocol:
        - Ignore anything not addressed to the buggy (dest != 1).
        - ack == 1 means the server is just confirming receipt of one of our
          messages; log it and stop.
        - Otherwise the server is sending a target letter in `msg`. Store the
          building as our current destination and ACK it back with the SAME
          uid, or the server retries 5x then declares "Connection Lost".
        """
        if message.dest != 1:
            return

        if message.ack == 1:
            self.get_logger().info(f"Server ACKed uid={message.uid}")
            return

        target_letter = message.msg.strip()
        building = self.sign_to_building.get(target_letter)
        if building is not None:
            self.current_destination = building
            self.awaiting_hospital = False
            self.get_logger().info(f"New destination: {building} (letter '{target_letter}')")
            self.send_server_ack(message.uid)

    def send_server_update(self, text_msg):
        """Sends a data message to the server with a proper rolling uid."""
        server_msg = ServerCommunication()
        server_msg.src = 1       # Source component: Buggy-1
        server_msg.dest = 2      # Destination component: Server-2
        server_msg.uid = self.server_uid
        server_msg.ack = 0
        server_msg.msg = text_msg
        self.publisher_server.publish(server_msg)
        self.server_uid = (self.server_uid + 1) % 256

    def send_server_ack(self, uid):
        """Sends an ack=1 back to the server with its uid to confirm receipt."""
        server_msg = ServerCommunication()
        server_msg.src = 1       # Source component: Buggy-1
        server_msg.dest = 2      # Destination component: Server-2
        server_msg.uid = uid
        server_msg.ack = 1
        server_msg.msg = ""
        self.publisher_server.publish(server_msg)
        self.get_logger().info(f"Sent ACK for uid={uid}")

    def qr_detection_callback(self, message):
        """
        Receives QR codes scanned from the buildings. When a recognized
        building QR is read AND LIDAR reports we're close to it (obstacle_in_front
        used as an in-zone proxy), register the building with the server.
        """
        data = message.data.strip()
        self.get_logger().info(f"Heard QR code: {data}")

        building = None
        for name in self.building_to_sign:
            if name in data:
                building = name
                break

        if building is None or building == self.last_sent_qr:
            return

        # Zone proxy: only register when LIDAR says we're up against the building.
        if not self.obstacle_in_front:
            return

        letter = self.building_to_sign[building]
        self.send_server_update(letter)
        self.last_sent_qr = building
        self.awaiting_hospital = True
        self.get_logger().info(f"Registered {building} (sent '{letter}') to server.")

    def sign_board_callback(self, message):
        """
        Receives traffic sign boards.

        GUIDELINE (Sign Board Routing):
        - Use the detected signs to choose the quickest route at intersections.
        - Not wired up yet - once this parses real signs, it can set
          self.drive_mode = "LINE_FOLLOW" and self.follow_side = "LEFT"/"RIGHT"
          directly instead of the DEFAULT_* constants at the top of the file.
        """
        self.get_logger().info(f"Heard Sign Board: {message.data}")
        pass


def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
