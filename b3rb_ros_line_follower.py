# Copyright 2024-2026 NXP
# Copyright 2016 Open Source Robotics Foundation, Inc
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

# NOTE on vector_1 / vector_2: each is geometry_msgs/Point[2] - a FIXED-size
# array of 2 points, always present regardless of vector_count. 

# ------------------ TUNABLE PARAMETERS ------------------
DEFAULT_DRIVE_MODE = "LANE_FOLLOW"
DEFAULT_FOLLOW_SIDE = "RIGHT"   # "LEFT" or "RIGHT"

STEER_KP = 0.004
LANE_GAP_OFFSET_RATIO = 0.4
TRACK_WIDTH_DIVERGENCE_RATIO = 1

# LANE_FOLLOW speeds
LANE_SPEED_TWO_LINES = 1.0 
LANE_SPEED_ONE_LINE = 1.0
LANE_SPEED_LOST_SHORT = 0.50
LANE_SPEED_LOST_LONG = 0.3 
LANE_LOST_GRACE_FRAMES = 5
LANE_APEX_BLIND_GRACE_FRAMES = 10
LANE_SHARP_SPEED = 0.4

# LINE_FOLLOW speeds/behavior
LINE_TURN_HOLD = 0.45 
LINE_SPEED_TRACK = 0.90 
LINE_SPEED_BLIND = 0.35 

STRAIGHT_SPEED = 0.8
STRAIGHT_STEER_KP = 0.004 
STRAIGHT_LOST_GRACE_FRAMES = 8 
STRAIGHT_LOST_SPEED = 0.4

# Obstacle avoidance tuning
OBSTACLE_DISTANCE_THRESHOLD = 0.65 
FRONT_SECTOR_START_FRAC = 7 / 18
FRONT_SECTOR_END_FRAC = 11 / 18 
AVOID_TURN = 0.3
AVOID_SPEED = 0.7

BLIND_APPROACH_DURATION = 2.4
AVOID_RECOVERY_FRAMES = 6 

# =============================================================================
# ===== COLLISION RECOVERY: PARAMETERS =====
# =============================================================================
RECOVERY_ENABLED = True 

STUCK_MIN_CMD_SPEED = 0.15 
STUCK_FRONT_RANGE = 0.30 
STUCK_FRONT_SECTOR_HALF_DEG = 30.0 
STUCK_RANGE_JITTER = 0.03 
STUCK_CONFIRM_FRAMES = 8 

RECOVERY_REVERSE_SPEED = -0.30 
RECOVERY_COUNTER_STEER = 0.60 

RECOVERY_REVERSE_BASE_TIME = 1.0 
RECOVERY_REVERSE_TIME_STEP = 0.6 
RECOVERY_PAUSE_TIME = 0.4 
RECOVERY_CLEAR_RANGE = 0.60 

RECOVERY_MAX_ATTEMPTS = 3 
RECOVERY_COOLDOWN = 2.0 
RECOVERY_ATTEMPT_RESET = 15.0 

RECOVERY_STATES_ACTIVE = ('REVERSE', 'PAUSE')

# =============================================================================
# ===== PARKING: PARAMETERS =====
# =============================================================================
PARK_SIDE = "LEFT" 

PARK_VEHICLE_WIDTH = 0.22
PARK_SAFETY_MARGIN = 0.06

PARK_SEARCH_SPEED_CAP = 0.45 
PARK_ENTRY_SPEED = 0.28 
PARK_CREEP_SPEED = 0.22 
PARK_TURN_FULL = 1.0 

PARK_CONE_RANGE_MAX = 1.60 
PARK_GAP_MIN_WIDTH = PARK_VEHICLE_WIDTH + 2.0 * PARK_SAFETY_MARGIN 
PARK_GAP_MAX_WIDTH = 1.20 
PARK_BAY_MIN_DEPTH_GAIN = 0.25 
PARK_GAP_MIN_BEAMS = 3 

PARK_SEARCH_SECTOR_CENTER_DEG = 90.0 
PARK_SEARCH_SECTOR_HALF_DEG = 55.0
PARK_ENTRY_SECTOR_CENTER_DEG = 50.0 
PARK_ENTRY_SECTOR_HALF_DEG = 70.0
PARK_FRONT_SECTOR_HALF_DEG = 18.0 
PARK_CENTER_SECTOR_HALF_DEG = 25.0 

PARK_COMMIT_LEAD_X = 0.45
PARK_BAY_MIN_X = 0.02 

PARK_ENTRY_ALIGNED_DEG = 18.0 

PARK_STOP_FRONT_RANGE = 0.35 
PARK_CENTER_DEADBAND = 0.10 
PARK_CENTER_KP = 1.2 
PARK_CENTER_TURN_CLAMP = 0.45 

PARK_START_DELAY = 1.0 
PARK_SEARCH_TIMEOUT = 30.0 
PARK_ENTRY_TIMEOUT = 5.0 
PARK_CREEP_TIMEOUT = 7.0 
PARK_SETTLE_DURATION = 1.5 

PARK_STATES_CAMERA_OFF = ('HOLD', 'ENTRY', 'CREEP', 'SETTLE', 'PARKED', 'ABORT')

def wrap_pi(a):
    return math.atan2(math.sin(a), math.cos(a))

def ang_diff(a, b):
    return wrap_pi(a - b)

def polar_to_xy(r, theta):
    return (r * math.cos(theta), r * math.sin(theta))

# =============================================================================
#  ===== K-TURN: CONSTANTS =====
# =============================================================================
KTURN_ENABLED = True

KTURN_CACHE_VALID_SEC = 20.0
KTURN_TRIGGER_MODE = "DIFFERENT_ARM"
KTURN_ON_UNKNOWN_ROUTE = False

KTURN_P1_REVERSE_SEC = 1.40
KTURN_P2_FORWARD_SEC = 1.18
KTURN_P3_REVERSE_SEC = 0.80
KTURN_P4_COAST_SEC = 0.60

KTURN_REVERSE_SPEED = -0.70
KTURN_FORWARD_SPEED = 0.60
KTURN_COAST_SPEED = 0.35
KTURN_STEER_FULL = 1.0
KTURN_DIRECTION = "RIGHT"

KTURN_REAR_SECTOR_HALF_DEG = 35.0
KTURN_FRONT_SECTOR_HALF_DEG = 30.0
KTURN_REAR_MIN_RANGE = 0.35
KTURN_FRONT_MIN_RANGE = 0.35

KTURN_TOTAL_TIMEOUT_SEC = 8.0
KTURN_STATES_ACTIVE = ('P1_REVERSE', 'P2_FORWARD', 'P3_REVERSE', 'P4_COAST')

class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')

        # ------------------ Subscriptions ------------------
        self.subscription_vectors = self.create_subscription(
            EdgeVectors,
            '/edge_vectors',
            self.edge_vectors_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_straight_vectors = self.create_subscription(
            EdgeVectors,
            '/straight_board_vectors',
            self.straight_vectors_callback,
            QOS_PROFILE_DEFAULT)
        self.latest_straight_vectors = None

        self.publisher_drive_mode = self.create_publisher(
            String,
            '/drive_mode',
            QOS_PROFILE_DEFAULT)

        self.drive_mode = DEFAULT_DRIVE_MODE 
        self.follow_side = DEFAULT_FOLLOW_SIDE 

        mode_msg = String()
        mode_msg.data = self.drive_mode 
        self.publisher_drive_mode.publish(mode_msg)

        self.subscription_lidar = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_server = self.create_subscription(
            ServerCommunication,
            '/ServerCommunication',
            self.server_communication_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_qr = self.create_subscription(
            String,
            '/qr_detection',
            self.qr_detection_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_signs = self.create_subscription(
            String,
            '/sign_board_detection',
            self.sign_board_callback,
            QOS_PROFILE_DEFAULT)

        # ------------------ Publishers ------------------
        self.publisher_joy = self.create_publisher(
            Joy,
            '/cerebri/in/joy',
            QOS_PROFILE_DEFAULT)

        self.publisher_server = self.create_publisher(
            ServerCommunication,
            '/ServerCommunication',
            QOS_PROFILE_DEFAULT)

        # ------------------ State Variables ------------------
        self.target_speed = 0.15
        self.target_turn = 0.0

        self.obstacle_in_front = False
        self.recovery_frames_remaining = 0
        self.last_avoid_turn = 0.0
        self.frames_avoided = 0
        self.patient_id = None
        self.hospital_id = None
        self.current_destination = None
        self.mission_completed = False
        
        self.horizontal_line_frames = 0
        self.apex_active = False
        self.apex_turn = 0.0
        self.apex_blind_frames = 0
        self.straight_turn_direction = None
        self.straight_lost_frames = 0
        self.straight_board_seen = False
        self.revert_lane_frames = 0
        self.revert_armed = False
        self.pending_intersection_direction = None

        # Server Comm State
        self.sign_to_building = {
            'A': 'PATIENT_1', 'B': 'PATIENT_2', 'C': 'PATIENT_3',
            'X': 'HOSPITAL_1', 'Y': 'HOSPITAL_2', 'Z': 'HOSPITAL_3',
        }
        self.building_to_sign = {v: k for k, v in self.sign_to_building.items()}
        self.server_uid = 0
        self.last_sent_qr = None
        self.awaiting_hospital = False
        self.current_destination = 'PATIENT_1'
        self.previous_destination = 'PATIENT_1'
        self.patients_delivered = 0
        self.waiting_for_ack = False
        self.server_retries = 0
        self.last_msg_send_time = 0.0

        # QR Approach State
        self.qr_last_seen_time = None
        self.qr_approach_active = False
        self.stopped_for_patient = False
        self.pending_letter = None
        self.pending_building = None

        # Parking State
        self.park_state = 'IDLE'
        self.park_state_entry_time = time.time()
        self.park_side_sign = 1.0 if PARK_SIDE == "LEFT" else -1.0
        self.park_locked_bay = None
        self.park_locked_bay_bearing = None
        self.park_latest_scan = None

        # Collision Recovery State
        self.recovery_state = 'IDLE'
        self.recovery_state_entry_time = time.time()
        self.recovery_latest_scan = None
        self.recovery_stuck_frames = 0
        self.recovery_range_history = []
        self.recovery_attempts = 0
        self.recovery_last_end_time = 0.0
        self.recovery_counter_turn = 0.0
        self.recovery_park_state_at_trip = None
        self.recovery_given_up = False

        # K-TURN: State Variables
        self.kturn_state = 'IDLE'
        self.kturn_state_entry_time = time.time()
        self.kturn_start_time = 0.0
        self.kturn_sign = 1.0 if KTURN_DIRECTION == "RIGHT" else -1.0
        self.sign_cache = {}
        self.sign_cache_time = 0.0
        self.sign_cache_direction_taken = None
        self.kturn_latest_scan = None

        # Lane Bookkeeping
        self.lane_lost_frame_count = 0
        self.last_target_x = None
        self.line_blind_frame_count = 0

        self.control_timer = self.create_timer(0.1, self.publish_drive_commands)

        self.get_logger().info(
            f"Line Follower controller initialized. drive_mode={self.drive_mode}"
            + (f" follow_side={self.follow_side}" if self.drive_mode == "LINE_FOLLOW" else "")
        )

    def publish_drive_commands(self):
        """Timer callback that periodically publishes the current speed and steer command."""
        self.check_qr_approach()
        self.check_server_retries()

        # Execute priorities properly
        self.check_parking_tick()
        self.check_kturn_tick()         # <-- K-Turn runs here (outranks parking)
        self.check_recovery_tick()      # <-- Recovery outranks everything

        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1] 
        msg.axes = [0.0, self.target_speed, 0.0, self.target_turn]
        self.publisher_joy.publish(msg)

    def rover_move_manual_mode(self, speed, turn):
        """Helper to immediately set control speed and steering angle."""
        self.target_speed = float(max(min(speed, SPEED_MAX), -SPEED_MAX))
        self.target_turn = float(max(min(turn, TURN_MAX), -TURN_MAX))

    # ------------------ QR Approach / Stop-in-Zone Logic ------------------

    def check_server_retries(self):
        if self.waiting_for_ack:
            current_time = time.time()
            if current_time - self.last_msg_send_time >= 1.0:
                if self.server_retries < 5:
                    self.send_server_update(self.pending_letter, uid=self.server_uid)
                    self.last_msg_send_time = current_time
                    self.server_retries += 1
                    self.get_logger().info(
                        f"No ACK yet. Retry {self.server_retries}/5 for '{self.pending_letter}' (UID {self.server_uid})..."
                    )
                else:
                    self.get_logger().warn("Max server retries reached, no ACK received. Idling.")
                    self.waiting_for_ack = False
                    self.server_uid = (self.server_uid + 1) % 256

    def check_qr_approach(self):
        if self.park_state != 'IDLE':
            return

        if not self.qr_approach_active or self.stopped_for_patient:
            return  

        elapsed = time.time() - self.qr_last_seen_time

        if elapsed >= BLIND_APPROACH_DURATION:
            self.target_speed = 0.0
            self.target_turn = 0.0  
            self.stopped_for_patient = True
            
            self.waiting_for_ack = True
            self.server_retries = 1
            self.last_msg_send_time = time.time()
            
            self.send_server_update(self.pending_letter, uid=self.server_uid)
            
            self.last_sent_qr = self.pending_building
            self.awaiting_hospital = True
            self.get_logger().info(
                f"Stopped at {self.pending_building}, sent '{self.pending_letter}' with UID {self.server_uid}."
            )

    # ------------------ Mode: LANE_FOLLOW ------------------

    def _handle_lane_follow(self, message, width, half_width):
        count = message.vector_count

        if count == 2:
            v1_mid_x = (message.vector_1[0].x + message.vector_1[1].x) / 2.0
            v2_mid_x = (message.vector_2[0].x + message.vector_2[1].x) / 2.0
            track_width = abs(v1_mid_x - v2_mid_x)

            if track_width <= (width * TRACK_WIDTH_DIVERGENCE_RATIO):
                target_x = (v1_mid_x + v2_mid_x) / 2.0
                self.target_speed = LANE_SPEED_TWO_LINES
            else:
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

            if dx > (dy * 3):
                self.horizontal_line_frames += 1
                line_center_x = (p0.x + p1.x) / 2.0

                sharpness = min(dx / max(dy, 0.001), 10.0)
                apex_speed = max(0.30, 0.70 - (sharpness * 0.04))

                if line_center_x >= half_width:
                    self.target_turn = (0.4+0.006*dx/dy)
                    self.target_speed = LANE_SHARP_SPEED
                else:
                    self.target_turn = -(0.4+0.006*dx/dy)
                    self.target_speed = LANE_SHARP_SPEED

                self.apex_active = True
                self.apex_turn = self.target_turn
                self.apex_blind_frames = 0
            else:
                self.horizontal_line_frames = 0
                self.apex_active = False 

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
            if self.apex_active and self.apex_blind_frames < LANE_APEX_BLIND_GRACE_FRAMES:
                self.apex_blind_frames += 1
                self.target_turn = self.apex_turn
                self.target_speed = LANE_SPEED_ONE_LINE
                self.lane_lost_frame_count = 0
            else:
                self.apex_active = False
                self.lane_lost_frame_count += 1
                self.target_speed = (LANE_SPEED_LOST_SHORT
                         if self.lane_lost_frame_count <= LANE_LOST_GRACE_FRAMES
                         else LANE_SPEED_LOST_LONG)

    # ------------------ Mode: LINE_FOLLOW ------------------

    def _handle_line_follow(self, message, width, half_width):
        count = message.vector_count
        sign = 1.0 if self.follow_side == "LEFT" else -1.0

        candidates = []
        if count >= 1:
            candidates.append(message.vector_1)
        if count >= 2:
            candidates.append(message.vector_2)

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
                self.target_turn = sign * min(1.0, 0.1 + dx / dy * 0.1)
                self.target_speed = LINE_SPEED_TRACK
            else:
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
            self.line_blind_frame_count += 1
            self.target_turn = sign * LINE_TURN_HOLD
            self.target_speed = LINE_SPEED_BLIND

    def straight_vectors_callback(self, message):
        self.latest_straight_vectors = message

    def _handle_straight(self, message, width, half_width):
        if message.vector_count > 0:
            self.straight_lost_frames = 0
            self._handle_lane_follow(message, width, half_width)
            return

        board_msg = self.latest_straight_vectors

        if board_msg is None or board_msg.vector_count == 0:
            self.straight_lost_frames += 1

            if not self.straight_board_seen:
                self.target_turn = 0.0
                self.target_speed = STRAIGHT_SPEED
            else:
                self.target_speed = (STRAIGHT_SPEED
                                      if self.straight_lost_frames < STRAIGHT_LOST_GRACE_FRAMES
                                      else STRAIGHT_LOST_SPEED)
            return

        self.straight_lost_frames = 0
        self.straight_board_seen = True

        p0, p1 = board_msg.vector_1[0], board_msg.vector_1[1]
        board_mid_x = (p0.x + p1.x) / 2.0
        error = half_width - board_mid_x
        self.target_turn = max(-1.0, min(1.0, STRAIGHT_STEER_KP * error))
        self.target_speed = STRAIGHT_SPEED

    def _check_revert_to_lane_follow(self, message, width):
        if self.drive_mode in ["LINE_FOLLOW", "STRAIGHT"]:
            if not self.revert_armed:
                if message.vector_count < 2:
                    self.revert_armed = True
                    self.get_logger().info("Entered intersection (vector_count < 2). Reversion logic ARMED.")
                return 

            if message.vector_count == 2:
                v1_mid_x = (message.vector_1[0].x + message.vector_1[1].x) / 2.0
                v2_mid_x = (message.vector_2[0].x + message.vector_2[1].x) / 2.0
                track_width = abs(v1_mid_x - v2_mid_x)

                if track_width <= (width * TRACK_WIDTH_DIVERGENCE_RATIO):
                    self.revert_lane_frames += 1
                    if self.revert_lane_frames >= 30: 
                        self.drive_mode = "LANE_FOLLOW"
                        self.revert_lane_frames = 0
                        self.revert_armed = False

                        mode_msg = String()
                        mode_msg.data = self.drive_mode
                        self.publisher_drive_mode.publish(mode_msg)

                        self.get_logger().info("Reacquired double lane boundaries post-turn. Reverted drive_mode to LANE_FOLLOW.")
            else:
                self.revert_lane_frames = 0
        
    # ------------------ Callback Implementations ------------------

    def edge_vectors_callback(self, message):
        # K-TURN GUARD
        if self.kturn_state in KTURN_STATES_ACTIVE:
            return

        if self.recovery_state in RECOVERY_STATES_ACTIVE:
            return

        if self.park_state in PARK_STATES_CAMERA_OFF:
            return

        if self.obstacle_in_front or self.stopped_for_patient:
            return

        width = float(message.image_width)
        if width <= 0:
            return
        half_width = width / 2.0

        if self.pending_intersection_direction and self.drive_mode == "LANE_FOLLOW":
            should_switch = False

            if self.pending_intersection_direction == "STRAIGHT":
                if message.vector_count == 0:
                    should_switch = True
            else:
                if message.vector_count < 2:
                    should_switch = True

            if should_switch:
                self.get_logger().info(
                    f"Entering intersection (vector_count={message.vector_count}). "
                    f"Activating mode '{self.pending_intersection_direction}'."
                )
                self._switch_drive_mode(self.pending_intersection_direction)
                self.pending_intersection_direction = None

        self._check_revert_to_lane_follow(message, width)

        if self.drive_mode == "LANE_FOLLOW":
            self._handle_lane_follow(message, width, half_width)
        elif self.drive_mode == "LINE_FOLLOW":
            self._handle_line_follow(message, width, half_width)
        elif self.drive_mode == "STRAIGHT":
            self._handle_straight(message, width, half_width)

    def lidar_callback(self, message):
        self.recovery_latest_scan = message

        if self.recovery_state in RECOVERY_STATES_ACTIVE:
            return          
        
        self._recovery_update_detector(message)
        
        if self.recovery_state in RECOVERY_STATES_ACTIVE:
            return          

        # --- KTURN SCAN CACHE ---
        if self.kturn_state in KTURN_STATES_ACTIVE:
            self.kturn_latest_scan = message
            return
        self.kturn_latest_scan = message
        # ------------------------

        if self.park_state != 'IDLE':
            self.park_latest_scan = message

            if self.park_state == 'SEARCH':
                self._park_do_search(message)
                if self.park_state != 'SEARCH':
                    return          
            elif self.park_state == 'ENTRY':
                self._park_do_entry(message)
                return
            elif self.park_state == 'CREEP':
                self._park_do_creep(message)
                return
            else:
                return              

        if self.stopped_for_patient:
            return

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
        
        if min_front_dist < OBSTACLE_DISTANCE_THRESHOLD:
            self.obstacle_in_front = True
            self.recovery_frames_remaining = AVOID_RECOVERY_FRAMES
            self.frames_avoided += 1
        
            mid = len(front_sector) // 2
            right_valid = valid(front_sector[:mid])
            left_valid = valid(front_sector[mid:])
            left_clearance = min(left_valid) if left_valid else float('inf')
            right_clearance = min(right_valid) if right_valid else float('inf')
        
            turn = AVOID_TURN if left_clearance >= right_clearance else -AVOID_TURN
            self.last_avoid_turn = turn
            self.target_turn = turn
            self.target_speed = AVOID_SPEED
        
        elif self.recovery_frames_remaining > 0 and self.frames_avoided > 5:
            self.get_logger().info(f"Frames avoided ={self.frames_avoided}")
            mid = len(front_sector) // 2
            right_valid = valid(front_sector[:mid])
            left_valid = valid(front_sector[mid:])
            left_clearance = min(left_valid) if left_valid else float('inf')
            right_clearance = min(right_valid) if right_valid else float('inf')
        
            return_dir = -self.last_avoid_turn   
            return_blocked = (
                (return_dir > 0 and min(left_valid)  < OBSTACLE_DISTANCE_THRESHOLD if left_valid  else False) or
                (return_dir < 0 and min(right_valid) < OBSTACLE_DISTANCE_THRESHOLD if right_valid else False)
            )
        
            self.obstacle_in_front = True
            self.recovery_frames_remaining -= 1
        
            if return_blocked:
                self.target_turn = 0.0
                self.target_speed = AVOID_SPEED * 0.8
            else:
                scale = min(self.frames_avoided / AVOID_RECOVERY_FRAMES, 1.0)
                recovery_turn = -self.last_avoid_turn * scale * 2
                self.target_turn = recovery_turn
                self.target_speed = AVOID_SPEED * 0.8        
        else:
            self.obstacle_in_front = False
            self.frames_avoided = 0              

    def server_communication_callback(self, message):
        if message.dest != 1:
            return

        if message.ack == 1:
            if message.uid == self.server_uid:
                self.get_logger().info(f"Server ACKed our message UID={message.uid}")
                self.waiting_for_ack = False
                self.server_uid = (self.server_uid + 1) % 256
            return

        raw_msg = message.msg.strip().upper()

        if raw_msg == "OK":
            self.patients_delivered += 1
            self.mission_completed = True
            self.target_speed = 0.0
            self.target_turn = 0.0
            self.stopped_for_patient = True
            
            self.get_logger().info(
                f"Received 'OK' from server! Mission Completed. "
                f"Total Patients Delivered: {self.patients_delivered}/3. Buggy stopped."
            )
            self.send_server_ack(message.uid)
            self.start_parking()

        elif raw_msg == "INVALID":
            self.get_logger().warn(
                f"Server returned 'INVALID'. Reverting destination from '{self.current_destination}' "
                f"to '{self.previous_destination}' and resuming movement."
            )
            self.current_destination = self.previous_destination
            self.stopped_for_patient = False
            self.qr_approach_active = False
            self.pending_letter = None
            self.pending_building = None
            
            self.send_server_ack(message.uid)

        else:
            building = self.sign_to_building.get(raw_msg)
            if building is not None:
                self.previous_destination = self.current_destination
                self.current_destination = building
                self.awaiting_hospital = False

                if "PATIENT" in building and "HOSPITAL" in self.previous_destination:
                    self.patients_delivered += 1
                    self.get_logger().info(f"Patient delivery confirmed! Count: {self.patients_delivered}/3")

                self.stopped_for_patient = False
                self.qr_approach_active = False
                self.pending_letter = None
                self.pending_building = None

                self.get_logger().info(f"New destination received: {building} (letter '{raw_msg}')")
                
                self.send_server_ack(message.uid)
                
                # --- START K-TURN LOGIC ---
                target_letter = self.building_to_sign.get(building)
                if target_letter:
                    self.start_kturn(target_letter)

    def send_server_update(self, text_msg, uid):
        server_msg = ServerCommunication()
        server_msg.src = 1       
        server_msg.dest = 2      
        server_msg.uid = uid
        server_msg.ack = 0
        server_msg.msg = text_msg
        self.publisher_server.publish(server_msg)

    def send_server_ack(self, uid_to_ack):
        server_msg = ServerCommunication()
        server_msg.src = 1       
        server_msg.dest = 2      
        server_msg.uid = uid_to_ack 
        server_msg.ack = 1
        server_msg.msg = ""
        self.publisher_server.publish(server_msg)

        self.server_uid = (self.server_uid + 1) % 256
        self.get_logger().info(f"Sent ACK for UID={uid_to_ack}. Advanced internal server_uid to {self.server_uid}.")

    def qr_detection_callback(self, message):
        if self.park_state != 'IDLE':
            return

        data = message.data.strip()

        if self.stopped_for_patient:
            return 

        building = None
        for name in self.building_to_sign:
            if name in data:
                building = name
                break

        if building is None or building == self.last_sent_qr:
            return

        if building != self.current_destination:
            self.get_logger().info(f"Saw {building}, but heading to {self.current_destination}")
            return

        self.qr_last_seen_time = time.time()
        self.qr_approach_active = True
        self.pending_letter = self.building_to_sign[building]
        self.pending_building = building
        self.get_logger().info(f"Approaching target: {building}.")

    def sign_board_callback(self, message):
        if self.park_state != 'IDLE':
            return

        if self.stopped_for_patient or self.obstacle_in_front:
            return

        raw_data = message.data.strip().upper()

        cleaned = raw_data
        for symbol in ["(", ")", "[", "]", "{", "}"]:
            cleaned = cleaned.replace(symbol, "")

        entries = cleaned.replace(',', ' ').split()

        sign_dict = {}
        for entry in entries:
            if ':' in entry:
                key, val = entry.split(':', 1)
                sign_dict[key.strip()] = val.strip()

        if len(sign_dict) < 6:
            return

        # Cache FULL sign map for K-Turn logic
        self._kturn_cache_sign(sign_dict)

        target_building = self.current_destination 
        target_letter = self.building_to_sign.get(target_building, '') 

        chosen_direction = sign_dict.get(target_letter) or sign_dict.get(target_building)

        if not chosen_direction:
            tokens = cleaned.replace(':', ' ').replace(',', ' ').split()
            valid_directions = ["LEFT", "RIGHT", "STRAIGHT"]
            for i in range(len(tokens) - 1):
                if tokens[i] in [target_building, target_letter] and tokens[i + 1] in valid_directions:
                    chosen_direction = tokens[i + 1]
                    break

        if chosen_direction:
            self.pending_intersection_direction = chosen_direction
            self.get_logger().info(
                f"Sign Board parsed target '{target_building}' ({target_letter}) -> "
                f"Buffered direction '{chosen_direction}' for upcoming intersection."
            )
        else:
            self.get_logger().warn(
                f"Sign board heard ({raw_data}), but no matching direction found for target '{target_building}'"
            )

    def _switch_drive_mode(self, direction):
        """Switches drive_mode and publishes the change to the vision node."""
        
        self._kturn_record_direction_taken(direction)

        if direction in ["LEFT", "RIGHT"]:
            self.drive_mode = "LINE_FOLLOW"
            self.follow_side = direction
        elif direction == "STRAIGHT":
            self.drive_mode = "STRAIGHT"
            self.straight_turn_direction = None
            self.straight_board_seen = False  
            self.straight_lost_frames = 0
            self.target_turn = 0.0              

        self.revert_lane_frames = 0
        self.revert_armed = False

        mode_msg = String()
        mode_msg.data = self.drive_mode
        self.publisher_drive_mode.publish(mode_msg)

        self.get_logger().info(
            f"Sign route selected direction '{direction}'. Switched drive_mode to '{self.drive_mode}' "
            f"(follow_side='{self.follow_side if self.drive_mode == 'LINE_FOLLOW' else 'N/A'}'). Reversion locked out."
        )

    # =========================================================================
    # ===== K-TURN: METHODS =====
    # =========================================================================

    def _kturn_cache_sign(self, sign_dict):
        if not sign_dict:
            return
        self.sign_cache = dict(sign_dict)
        self.sign_cache_time = time.time()
        self.get_logger().info(
            f"Sign cache updated ({len(self.sign_cache)} routes), "
            f"valid for {KTURN_CACHE_VALID_SEC:.0f} s: {self.sign_cache}")

    def _kturn_record_direction_taken(self, direction):
        self.sign_cache_direction_taken = direction
        self.get_logger().info(
            f"K-turn bookkeeping: arm taken at last junction = '{direction}'")

    def _kturn_cache_is_fresh(self):
        if not self.sign_cache:
            return False
        return (time.time() - self.sign_cache_time) <= KTURN_CACHE_VALID_SEC

    def _kturn_should_turn(self, new_destination_letter):
        if not KTURN_ENABLED:
            return False, "K-turn disabled"

        if self.kturn_state != 'IDLE':
            return False, "K-turn already running"

        if not self._kturn_cache_is_fresh():
            age = (time.time() - self.sign_cache_time) if self.sign_cache else -1
            return False, (f"sign cache stale/empty (age={age:.1f}s > "
                           f"{KTURN_CACHE_VALID_SEC:.0f}s) - driving forward")

        if self.sign_cache_direction_taken is None:
            return False, "no record of which arm we took - driving forward"

        dir_next = self.sign_cache.get(new_destination_letter)
        if dir_next is None:
            if KTURN_ON_UNKNOWN_ROUTE:
                return True, (f"'{new_destination_letter}' not in cached sign - "
                              f"turning on the unknown-route policy")
            return False, (f"'{new_destination_letter}' not in cached sign - "
                           f"driving forward (conservative default)")

        dir_taken = self.sign_cache_direction_taken

        same_arm = (dir_next == dir_taken)

        if KTURN_TRIGGER_MODE == "DIFFERENT_ARM":
            should = not same_arm
        else:
            should = same_arm

        reason = (f"took '{dir_taken}', '{new_destination_letter}' is on "
                  f"'{dir_next}' (same_arm={same_arm}, "
                  f"mode={KTURN_TRIGGER_MODE}) -> "
                  f"{'K-TURN' if should else 'drive forward'}")
        return should, reason

    def start_kturn(self, new_destination_letter):
        should, reason = self._kturn_should_turn(new_destination_letter)
        self.get_logger().info(f"K-turn decision: {reason}")

        if not should:
            return False

        self.get_logger().warn(
            f"=== DESTINATION '{new_destination_letter}' IS BEHIND US - "
            f"STARTING K-TURN ({KTURN_DIRECTION}) ===")

        self.stopped_for_patient = False
        self.qr_approach_active = False
        self.pending_intersection_direction = None

        self.kturn_start_time = time.time()
        self._kturn_transition('P1_REVERSE')
        return True

    def _kturn_transition(self, new_state):
        self.get_logger().info(f"K-TURN FSM: {self.kturn_state} -> {new_state}")
        self.kturn_state = new_state
        self.kturn_state_entry_time = time.time()

    def _kturn_time_in_state(self):
        return time.time() - self.kturn_state_entry_time

    def _kturn_rear_range(self):
        scan = self.kturn_latest_scan
        if scan is None:
            return math.inf
        beams = self._park_beams_in_sector(
            scan, math.pi, math.radians(KTURN_REAR_SECTOR_HALF_DEG))
        finite = [r for (_, r) in beams if math.isfinite(r)]
        return min(finite) if finite else math.inf

    def _kturn_front_range(self):
        scan = self.kturn_latest_scan
        if scan is None:
            return math.inf
        beams = self._park_beams_in_sector(
            scan, 0.0, math.radians(KTURN_FRONT_SECTOR_HALF_DEG))
        finite = [r for (_, r) in beams if math.isfinite(r)]
        return min(finite) if finite else math.inf

    def check_kturn_tick(self):
        if self.kturn_state == 'IDLE':
            return             

        if (time.time() - self.kturn_start_time) > KTURN_TOTAL_TIMEOUT_SEC:
            self.get_logger().warn(
                f"K-turn exceeded {KTURN_TOTAL_TIMEOUT_SEC:.0f} s total - "
                f"aborting and handing control back.")
            self._kturn_finish()
            return

        s = self.kturn_sign

        if self.kturn_state == 'P1_REVERSE':
            rear = self._kturn_rear_range()

            if rear < KTURN_REAR_MIN_RANGE:
                self.get_logger().warn(
                    f"K-turn P1 aborted early - rear obstacle at {rear:.2f} m. "
                    f"Advancing to forward phase.")
                self._kturn_transition('P2_FORWARD')
                return

            if self._kturn_time_in_state() >= KTURN_P1_REVERSE_SEC:
                self._kturn_transition('P2_FORWARD')
                return

            self.target_speed = KTURN_REVERSE_SPEED
            self.target_turn = KTURN_STEER_FULL * s

        elif self.kturn_state == 'P2_FORWARD':
            front = self._kturn_front_range()

            if front < KTURN_FRONT_MIN_RANGE:
                self.get_logger().warn(
                    f"K-turn P2 aborted early - front obstacle at {front:.2f} m. "
                    f"Advancing to second reverse.")
                self._kturn_transition('P3_REVERSE')
                return

            if self._kturn_time_in_state() >= KTURN_P2_FORWARD_SEC:
                self._kturn_transition('P3_REVERSE')
                return

            self.target_speed = KTURN_FORWARD_SPEED
            self.target_turn = -KTURN_STEER_FULL * s

        elif self.kturn_state == 'P3_REVERSE':
            rear = self._kturn_rear_range()

            if rear < KTURN_REAR_MIN_RANGE:
                self.get_logger().warn(
                    f"K-turn P3 aborted early - rear obstacle at {rear:.2f} m. "
                    f"Advancing to coast-out.")
                self._kturn_transition('P4_COAST')
                return

            if self._kturn_time_in_state() >= KTURN_P3_REVERSE_SEC:
                self._kturn_transition('P4_COAST')
                return

            self.target_speed = KTURN_REVERSE_SPEED * 0.93
            self.target_turn = KTURN_STEER_FULL * s

        elif self.kturn_state == 'P4_COAST':
            if self._kturn_time_in_state() >= KTURN_P4_COAST_SEC:
                self._kturn_finish()
                return

            self.target_speed = KTURN_COAST_SPEED
            self.target_turn = -KTURN_STEER_FULL * s

    def _kturn_finish(self):
        self.get_logger().info("K-turn complete - wiping stale lane state.")

        self.apex_active = False
        self.apex_turn = 0.0
        self.apex_blind_frames = 0
        self.last_target_x = None
        self.lane_lost_frame_count = 0
        self.line_blind_frame_count = 0
        self.horizontal_line_frames = 0

        self.drive_mode = "LANE_FOLLOW"
        self.revert_armed = False
        self.revert_lane_frames = 0
        self.pending_intersection_direction = None
        self.straight_board_seen = False
        self.straight_lost_frames = 0

        mode_msg = String()
        mode_msg.data = self.drive_mode
        self.publisher_drive_mode.publish(mode_msg)

        self.obstacle_in_front = False
        self.frames_avoided = 0
        self.recovery_frames_remaining = 0

        self.sign_cache = {}
        self.sign_cache_time = 0.0
        self.sign_cache_direction_taken = None

        self.target_speed = 0.0
        self.target_turn = 0.0
        self._kturn_transition('IDLE')

    # =========================================================================
    # ===== PARKING: METHODS =====
    # =========================================================================
    
    def start_parking(self):
        if self.park_state != 'IDLE':
            self.get_logger().warn(
                f"start_parking() ignored - already in park_state '{self.park_state}'.")
            return
        self.get_logger().info("=== ALL DELIVERIES COMPLETE - BEGINNING PARKING ===")
        self._park_transition('HOLD')

    def _park_transition(self, new_state):
        self.get_logger().info(f"PARK FSM: {self.park_state} -> {new_state}")
        self.park_state = new_state
        self.park_state_entry_time = time.time()

    def _park_time_in_state(self):
        return time.time() - self.park_state_entry_time

    def check_parking_tick(self):
        if self.park_state == 'IDLE':
            return      

        if self.park_state == 'HOLD':
            self.target_speed = 0.0
            self.target_turn = 0.0
            if self._park_time_in_state() >= PARK_START_DELAY:
                self.stopped_for_patient = False
                self.qr_approach_active = False
                self.waiting_for_ack = False
                self._park_transition('SEARCH')

        elif self.park_state == 'SEARCH':
            if self.target_speed > PARK_SEARCH_SPEED_CAP:
                self.target_speed = PARK_SEARCH_SPEED_CAP

            if self._park_time_in_state() > PARK_SEARCH_TIMEOUT:
                self.get_logger().warn(
                    "PARK SEARCH timed out - no valid bay found. Halting safely.")
                self._park_transition('ABORT')

        elif self.park_state == 'SETTLE':
            self.target_speed = 0.0
            self.target_turn = 0.0
            if self._park_time_in_state() >= PARK_SETTLE_DURATION:
                self._park_report_final()
                self._park_transition('PARKED')

        elif self.park_state in ('PARKED', 'ABORT'):
            self.target_speed = 0.0
            self.target_turn = 0.0

    def _park_beams_in_sector(self, scan, center_rad, half_span_rad):
        out = []
        for i, r in enumerate(scan.ranges):
            theta = wrap_pi(scan.angle_min + i * scan.angle_increment)
            if abs(ang_diff(theta, center_rad)) > half_span_rad:
                continue
            if (not math.isfinite(r)) or r < scan.range_min or r > scan.range_max:
                r = math.inf
            out.append((theta, r))
        out.sort(key=lambda t: t[0])
        return out

    def _park_min_range(self, scan, center_deg, half_span_deg):
        beams = self._park_beams_in_sector(
            scan, math.radians(center_deg), math.radians(half_span_deg))
        finite = [r for (_, r) in beams if math.isfinite(r)]
        return min(finite) if finite else math.inf

    def _park_find_bays(self, beams):
        n = len(beams)
        if n < 3:
            return []

        is_structure = [math.isfinite(r) and r < PARK_CONE_RANGE_MAX
                        for (_, r) in beams]

        bays = []
        i = 0
        while i < n:
            if is_structure[i]:
                i += 1
                continue

            run_start = i
            while i < n and not is_structure[i]:
                i += 1
            run_end = i - 1

            left_tooth_idx = run_start - 1
            right_tooth_idx = run_end + 1
            if left_tooth_idx < 0 or right_tooth_idx >= n:
                continue        

            if (run_end - run_start + 1) < PARK_GAP_MIN_BEAMS:
                continue

            th_a, r_a = beams[left_tooth_idx]
            th_b, r_b = beams[right_tooth_idx]
            xa, ya = polar_to_xy(r_a, th_a)
            xb, yb = polar_to_xy(r_b, th_b)
            width = math.hypot(xb - xa, yb - ya)

            if width < PARK_GAP_MIN_WIDTH or width > PARK_GAP_MAX_WIDTH:
                continue

            run_ranges = [r for (_, r) in beams[run_start:run_end + 1]]
            finite_run = [r for r in run_ranges if math.isfinite(r)]
            if finite_run:
                run_depth = sorted(finite_run)[len(finite_run) // 2]   
            else:
                run_depth = math.inf        
            tooth_depth = min(r_a, r_b)
            if run_depth < tooth_depth + PARK_BAY_MIN_DEPTH_GAIN:
                continue

            cx = 0.5 * (xa + xb)
            cy = 0.5 * (ya + yb)
            bays.append({
                'x': cx,                        
                'y': cy,                        
                'bearing': math.atan2(cy, cx),  
                'range': math.hypot(cx, cy),
                'width': width,
            })

        return bays

    def _park_select_bay(self, bays):
        ahead = [b for b in bays if b['x'] > PARK_BAY_MIN_X]
        if not ahead:
            return None

        if self.park_locked_bay_bearing is None:
            return min(ahead, key=lambda b: b['x'])

        return min(bays, key=lambda b: abs(ang_diff(
            b['bearing'], self.park_locked_bay_bearing)))

    def _park_do_search(self, scan):
        center = math.radians(PARK_SEARCH_SECTOR_CENTER_DEG) * self.park_side_sign
        beams = self._park_beams_in_sector(
            scan, center, math.radians(PARK_SEARCH_SECTOR_HALF_DEG))
        bay = self._park_select_bay(self._park_find_bays(beams))

        if bay is None:
            return          

        self.park_locked_bay = bay
        self.park_locked_bay_bearing = bay['bearing']

        self.get_logger().info(
            f"Bay candidate: x={bay['x']:.2f} m, y={bay['y']:.2f} m, "
            f"w={bay['width']:.2f} m, brg={math.degrees(bay['bearing']):.0f} deg",
            throttle_duration_sec=0.5)

        if bay['x'] <= PARK_COMMIT_LEAD_X:
            self.get_logger().info(
                f"COMMITTING to bay: x={bay['x']:.2f} m "
                f"(lead={PARK_COMMIT_LEAD_X:.2f} m), width={bay['width']:.2f} m")
            self.apex_active = False
            self._park_transition('ENTRY')

    def _park_do_entry(self, scan):
        center = math.radians(PARK_ENTRY_SECTOR_CENTER_DEG) * self.park_side_sign
        beams = self._park_beams_in_sector(
            scan, center, math.radians(PARK_ENTRY_SECTOR_HALF_DEG))
        bay = self._park_select_bay(self._park_find_bays(beams))

        if bay is not None:
            self.park_locked_bay = bay
            self.park_locked_bay_bearing = bay['bearing']

            if abs(bay['bearing']) < math.radians(PARK_ENTRY_ALIGNED_DEG):
                self.get_logger().info(
                    f"Arc complete - bay now at "
                    f"{math.degrees(bay['bearing']):.0f} deg. Straightening.")
                self._park_transition('CREEP')
                return

        if self._park_time_in_state() > PARK_ENTRY_TIMEOUT:
            self.get_logger().warn(
                "PARK ENTRY timed out - assuming arc complete, handing to CREEP.")
            self._park_transition('CREEP')
            return

        self.rover_move_manual_mode(PARK_ENTRY_SPEED,
                                    PARK_TURN_FULL * self.park_side_sign)

    def _park_do_creep(self, scan):
        front_r = self._park_min_range(scan, 0.0, PARK_FRONT_SECTOR_HALF_DEG)
        left_r = self._park_min_range(scan, 90.0, PARK_CENTER_SECTOR_HALF_DEG)
        right_r = self._park_min_range(scan, -90.0, PARK_CENTER_SECTOR_HALF_DEG)

        if front_r < PARK_STOP_FRONT_RANGE:
            self.get_logger().info(
                f"Depth reached: front={front_r:.2f} m, "
                f"L={left_r:.2f} R={right_r:.2f}. Settling.")
            self._park_transition('SETTLE')
            return

        if self._park_time_in_state() > PARK_CREEP_TIMEOUT:
            self.get_logger().warn(
                f"PARK CREEP timed out at front={front_r:.2f} m. Stopping here.")
            self._park_transition('SETTLE')
            return

        if math.isfinite(left_r) and math.isfinite(right_r):
            error = left_r - right_r
            if abs(error) < PARK_CENTER_DEADBAND:
                turn = 0.0                      
            else:
                turn = PARK_CENTER_KP * error
                turn = max(-PARK_CENTER_TURN_CLAMP,
                           min(PARK_CENTER_TURN_CLAMP, turn))
        else:
            turn = 0.0

        self.rover_move_manual_mode(PARK_CREEP_SPEED, turn)

    def _park_report_final(self):
        scan = self.park_latest_scan
        if scan is None:
            self.get_logger().info("PARKED (no scan available for final report).")
            return
        front_r = self._park_min_range(scan, 0.0, PARK_FRONT_SECTOR_HALF_DEG)
        left_r = self._park_min_range(scan, 90.0, PARK_CENTER_SECTOR_HALF_DEG)
        right_r = self._park_min_range(scan, -90.0, PARK_CENTER_SECTOR_HALF_DEG)
        asym = abs(left_r - right_r) if (math.isfinite(left_r)
                                         and math.isfinite(right_r)) else float('nan')
        self.get_logger().info(
            f"=== PARKED === front={front_r:.2f} m  left={left_r:.2f} m  "
            f"right={right_r:.2f} m  asymmetry={asym:.2f} m")

    def park_dump_side_profile(self):
        if self.park_latest_scan is None:
            self.get_logger().warn("No scan received yet.")
            return
        center = math.radians(PARK_SEARCH_SECTOR_CENTER_DEG) * self.park_side_sign
        beams = self._park_beams_in_sector(
            self.park_latest_scan, center,
            math.radians(PARK_SEARCH_SECTOR_HALF_DEG))
        comb = ''.join('#' if (math.isfinite(r) and r < PARK_CONE_RANGE_MAX)
                       else '.' for (_, r) in beams)
        self.get_logger().info(f"side comb: [{comb}]")
        for b in self._park_find_bays(beams):
            self.get_logger().info(
                f"   bay  x={b['x']:+.2f}  y={b['y']:+.2f}  "
                f"w={b['width']:.2f}  brg={math.degrees(b['bearing']):+.0f}")

    # =========================================================================
    # ===== COLLISION RECOVERY: METHODS =====
    # =========================================================================

    def _recovery_front_range(self, scan):
        beams = self._park_beams_in_sector(
            scan, 0.0, math.radians(STUCK_FRONT_SECTOR_HALF_DEG))
        finite = [r for (_, r) in beams if math.isfinite(r)]
        return min(finite) if finite else math.inf

    def _recovery_clear_detector(self):
        self.recovery_stuck_frames = 0
        self.recovery_range_history = []

    def _recovery_update_detector(self, scan):
        if not RECOVERY_ENABLED:
            return

        if (self.recovery_attempts > 0
                and (time.time() - self.recovery_last_end_time) > RECOVERY_ATTEMPT_RESET):
            self.recovery_attempts = 0
            self.recovery_given_up = False

        if self.recovery_given_up:
            return

        if (time.time() - self.recovery_last_end_time) < RECOVERY_COOLDOWN:
            self._recovery_clear_detector()
            return

        if self.target_speed < STUCK_MIN_CMD_SPEED:
            self._recovery_clear_detector()
            return

        if self.stopped_for_patient:
            self._recovery_clear_detector()
            return

        if self.park_state in ('HOLD', 'CREEP', 'SETTLE', 'PARKED', 'ABORT'):
            self._recovery_clear_detector()
            return

        front_r = self._recovery_front_range(scan)
        if (not math.isfinite(front_r)) or front_r > STUCK_FRONT_RANGE:
            self._recovery_clear_detector()
            return

        self.recovery_range_history.append(front_r)
        if len(self.recovery_range_history) > STUCK_CONFIRM_FRAMES:
            self.recovery_range_history.pop(0)

        self.recovery_stuck_frames += 1

        if self.recovery_stuck_frames >= STUCK_CONFIRM_FRAMES:
            h = self.recovery_range_history
            if len(h) >= STUCK_CONFIRM_FRAMES and (max(h) - min(h)) < STUCK_RANGE_JITTER:
                self._recovery_start(front_r)
            else:
                self.recovery_stuck_frames = STUCK_CONFIRM_FRAMES

    def _recovery_start(self, front_r):
        self.recovery_attempts += 1
        self.recovery_park_state_at_trip = self.park_state

        pre_turn = self.target_turn
        if abs(pre_turn) < 0.05:
            self.recovery_counter_turn = (RECOVERY_COUNTER_STEER
                                          if (self.recovery_attempts % 2 == 1)
                                          else -RECOVERY_COUNTER_STEER)
        else:
            self.recovery_counter_turn = -math.copysign(
                RECOVERY_COUNTER_STEER, pre_turn)

        self.get_logger().warn(
            f"STUCK DETECTED: front={front_r:.2f} m held flat for "
            f"{self.recovery_stuck_frames} frames while commanding "
            f"speed={self.target_speed:.2f}. Recovery attempt "
            f"{self.recovery_attempts}/{RECOVERY_MAX_ATTEMPTS}, "
            f"counter-steer={self.recovery_counter_turn:+.2f}")

        self._recovery_clear_detector()
        self._recovery_transition('REVERSE')

    def _recovery_transition(self, new_state):
        self.get_logger().info(
            f"RECOVERY FSM: {self.recovery_state} -> {new_state}")
        self.recovery_state = new_state
        self.recovery_state_entry_time = time.time()

    def _recovery_time_in_state(self):
        return time.time() - self.recovery_state_entry_time

    def check_recovery_tick(self):
        if self.recovery_state == 'IDLE':
            return          

        if self.recovery_state == 'REVERSE':
            duration = (RECOVERY_REVERSE_BASE_TIME
                        + (self.recovery_attempts - 1) * RECOVERY_REVERSE_TIME_STEP)

            self.target_speed = RECOVERY_REVERSE_SPEED
            self.target_turn = self.recovery_counter_turn

            if self._recovery_time_in_state() >= duration:
                self._recovery_transition('PAUSE')

        elif self.recovery_state == 'PAUSE':
            self.target_speed = 0.0
            self.target_turn = 0.0

            if self._recovery_time_in_state() < RECOVERY_PAUSE_TIME:
                return

            scan = self.recovery_latest_scan
            front_r = (self._recovery_front_range(scan)
                       if scan is not None else math.inf)

            if front_r >= RECOVERY_CLEAR_RANGE:
                self.get_logger().info(
                    f"Recovery successful - front now {front_r:.2f} m. "
                    f"Handing control back.")
                self._recovery_finish()

            elif self.recovery_attempts < RECOVERY_MAX_ATTEMPTS:
                self.recovery_attempts += 1
                self.recovery_counter_turn = -self.recovery_counter_turn
                self.get_logger().warn(
                    f"Still blocked at {front_r:.2f} m. Escalating to attempt "
                    f"{self.recovery_attempts}/{RECOVERY_MAX_ATTEMPTS}, "
                    f"counter-steer={self.recovery_counter_turn:+.2f}")
                self._recovery_transition('REVERSE')

            else:
                self.get_logger().error(
                    f"Recovery exhausted after {RECOVERY_MAX_ATTEMPTS} attempts "
                    f"(front={front_r:.2f} m). Giving up, resuming normal drive.")
                self.recovery_given_up = True
                self._recovery_finish()

    def _recovery_finish(self):
        self.recovery_last_end_time = time.time()
        self._recovery_clear_detector()

        self.obstacle_in_front = False
        self.frames_avoided = 0
        self.recovery_frames_remaining = 0

        self.apex_active = False
        self.apex_blind_frames = 0

        if self.recovery_park_state_at_trip == 'ENTRY':
            self.get_logger().info(
                "Collision occurred during park ENTRY - dropping bay lock and "
                "returning to SEARCH to re-acquire.")
            self.park_locked_bay = None
            self.park_locked_bay_bearing = None
            self._park_transition('SEARCH')

        self.recovery_park_state_at_trip = None
        self._recovery_transition('IDLE')

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
