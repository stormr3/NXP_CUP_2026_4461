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

# ------------------ TUNABLE PARAMETERS ------------------
DEFAULT_DRIVE_MODE = "LANE_FOLLOW"
DEFAULT_FOLLOW_SIDE = "RIGHT"   # "LEFT" or "RIGHT"

# Gain & Track parameters
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

# ===== PARKING MANEUVER TUNABLES =====
PARKING_TURN_DIRECTION = "LEFT"  # Switch to "RIGHT" if spot is on the right
PARKING_CONE_DIST = 0.75          # Distance to parking cone to trigger turn (meters)
PARKING_APPROACH_SPEED = 0.30    # Slow crawl speed toward cone
PARKING_TURN_SPEED = 0.45        # Forward speed during insertion turn
PARKING_TURN_STEER = 0.75        # Steering angle magnitude (0.0 to 1.0)
PARKING_TURN_DURATION = 1.8      # Duration (seconds) to hold turn into spot
# =====================================

class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')

        # Subscriptions
        self.subscription_vectors = self.create_subscription(
            EdgeVectors, '/edge_vectors', self.edge_vectors_callback, QOS_PROFILE_DEFAULT)

        self.subscription_straight_vectors = self.create_subscription(
            EdgeVectors, '/straight_board_vectors', self.straight_vectors_callback, QOS_PROFILE_DEFAULT)
        self.latest_straight_vectors = None  

        self.publisher_drive_mode = self.create_publisher(
            String, '/drive_mode', QOS_PROFILE_DEFAULT)

        self.drive_mode = DEFAULT_DRIVE_MODE      
        self.follow_side = DEFAULT_FOLLOW_SIDE    

        mode_msg = String()
        mode_msg.data = self.drive_mode  
        self.publisher_drive_mode.publish(mode_msg)

        self.subscription_lidar = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, QOS_PROFILE_DEFAULT)

        self.subscription_server = self.create_subscription(
            ServerCommunication, '/ServerCommunication', self.server_communication_callback, QOS_PROFILE_DEFAULT)

        self.subscription_qr = self.create_subscription(
            String, '/qr_detection', self.qr_detection_callback, QOS_PROFILE_DEFAULT)

        self.subscription_signs = self.create_subscription(
            String, '/sign_board_detection', self.sign_board_callback, QOS_PROFILE_DEFAULT)

        # Publishers
        self.publisher_joy = self.create_publisher(
            Joy, '/cerebri/in/joy', QOS_PROFILE_DEFAULT)

        self.publisher_server = self.create_publisher(
            ServerCommunication, '/ServerCommunication', QOS_PROFILE_DEFAULT)

        # Controls & States
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

        # PARKING STATE MACHINE
        self.parking_mode = False
        self.parking_state = "IDLE"  # "IDLE", "SEARCH_CONE", "TURNING", "FINISHED"
        self.parking_start_time = 0.0

        # Line Tracking Bookkeeping
        self.lane_lost_frame_count = 0
        self.last_target_x = None  
        self.line_blind_frame_count = 0

        # Timer
        self.control_timer = self.create_timer(0.1, self.publish_drive_commands)

        self.get_logger().info(
            f"Line Follower controller initialized. drive_mode={self.drive_mode}"
        )

    def publish_drive_commands(self):
        """Timer callback publishing speed and steer commands at 10Hz."""
        self.check_qr_approach()
        self.check_server_retries()

        # Handle Parking Motion Execution
        if self.parking_mode:
            if self.parking_state == "SEARCH_CONE":
                # Crawl straight/lane-follow until cone is spotted
                self.target_speed = PARKING_APPROACH_SPEED
            elif self.parking_state == "TURNING":
                elapsed = time.time() - self.parking_start_time
                if elapsed < PARKING_TURN_DURATION:
                    sign = 1.0 if PARKING_TURN_DIRECTION == "LEFT" else -1.0
                    self.target_speed = PARKING_TURN_SPEED
                    self.target_turn = sign * PARKING_TURN_STEER
                else:
                    self.get_logger().info("Parking maneuver complete! Buggy parked.")
                    self.waiting_for_ack = True
                    self.server_retries = 1
                    self.last_msg_send_time = time.time()
                    self.send_server_update("PARKED", uid=self.server_uid)
                    self.get_logger().info(f"Sent 'PARKED' to server with UID {self.server_uid}.")
                    self.parking_state = "FINISHED"
                    self.stopped_for_patient = True
                    self.target_speed = 0.0
                    self.target_turn = 0.0
            elif self.parking_state == "FINISHED":
                self.target_speed = 0.0
                self.target_turn = 0.0

        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1] 
        msg.axes = [0.0, self.target_speed, 0.0, self.target_turn]
        self.publisher_joy.publish(msg)

    def rover_move_manual_mode(self, speed, turn):
        self.target_speed = float(max(min(speed, SPEED_MAX), -SPEED_MAX))
        self.target_turn = float(max(min(turn, TURN_MAX), -TURN_MAX))

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
        if not self.qr_approach_active or self.stopped_for_patient or self.parking_mode:
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

    # ------------------ Callbacks ------------------

    def edge_vectors_callback(self, message):
        if self.obstacle_in_front or self.stopped_for_patient or self.parking_state in ["TURNING", "FINISHED"]:
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

        # Dispatch based on current mode
        if self.drive_mode == "LANE_FOLLOW":
            self._handle_lane_follow(message, width, half_width)
        elif self.drive_mode == "LINE_FOLLOW":
            self._handle_line_follow(message, width, half_width)
        elif self.drive_mode == "STRAIGHT":
            self._handle_straight(message, width, half_width)

    def lidar_callback(self, message):
        if self.stopped_for_patient and not self.parking_mode:
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

        # PARKING CONE DETECTION
        if self.parking_mode:
            if self.parking_state == "SEARCH_CONE":
                if min_front_dist <= PARKING_CONE_DIST:
                    self.get_logger().info(
                        f"Parking Cone detected at {min_front_dist:.2f}m! Initiating {PARKING_TURN_DIRECTION} turn."
                    )
                    self.parking_state = "TURNING"
                    self.parking_start_time = time.time()
            return  # Skip regular obstacle avoidance during parking

        # REGULAR OBSTACLE AVOIDANCE
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
        
        elif self.recovery_frames_remaining > 0 and self.frames_avoided > 30:
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

        # MISSION COMPLETE -> PARKING INITIATED
        if raw_msg == "OK":
            self.patients_delivered += 1
            self.mission_completed = True
            
            self.get_logger().info(
                f"Received 'OK' from server! Total Patients Delivered: {self.patients_delivered}/3. "
                f"Initiating Parking Search Mode."
            )
            self.send_server_ack(message.uid)

            # Trigger Parking State Machine
            self.parking_mode = True
            self.parking_state = "SEARCH_CONE"
            self.stopped_for_patient = False

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
        data = message.data.strip()

        if self.stopped_for_patient or self.parking_mode:
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
        if self.stopped_for_patient or self.obstacle_in_front or self.parking_mode:
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