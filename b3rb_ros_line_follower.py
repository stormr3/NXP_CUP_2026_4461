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
DEFAULT_FOLLOW_SIDE = "RIGHT"   # "LEFT" or "RIGHT"

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
LANE_SPEED_TWO_LINES = 1.0    # confident: both boundaries agree on a normal-width track
LANE_SPEED_ONE_LINE = 1.0     # only one boundary in play (either genuinely one visible, or divergence fallback)
LANE_SPEED_LOST_SHORT = 0.50   # briefly no boundary at all: hold last turn, ease off speed
LANE_SPEED_LOST_LONG = 0.3    # lost for a while: crawl instead of driving blind at speed
LANE_LOST_GRACE_FRAMES = 5     # frames before dropping from LOST_SHORT to LOST_LONG speed
LANE_APEX_BLIND_GRACE_FRAMES = 10  # frames to keep sustaining the apex turn after the line vanishes mid-apex, before giving up and falling back to generic lost-line handling
LANE_SHARP_SPEED = 0.4

# LINE_FOLLOW speeds/behavior (single committed line, e.g. through an intersection)
LINE_TURN_HOLD = 0.45          # steer magnitude to hold a turn arc toward the committed line
LINE_SPEED_TRACK = 0.90        # speed while the committed line is visible and tracked
LINE_SPEED_BLIND = 0.35        # speed while the committed line has vanished (e.g. turn apex) and we're sweeping to reacquire it

STRAIGHT_SPEED = 0.8

# STRAIGHT mode (camera-based, green-signboard gate steering)
STRAIGHT_STEER_KP = 0.004          # start same as lane STEER_KP, tune from debug image
STRAIGHT_LOST_GRACE_FRAMES = 8     # frames with no board visible before easing off speed
STRAIGHT_LOST_SPEED = 0.4          # speed once lost longer than the grace period

# Obstacle avoidance tuning
OBSTACLE_DISTANCE_THRESHOLD = 0.65    # meters; trigger avoidance below this range
FRONT_SECTOR_START_FRAC = 7 / 18     # start of the front sector, as a fraction of the full 360 scan
FRONT_SECTOR_END_FRAC = 11 / 18      # end of the front sector, as a fraction of the full 360 scan
AVOID_TURN = 0.3
AVOID_SPEED = 0.7


# ===== QR APPROACH / STOP-IN-ZONE TIMING =====
# The QR is mounted high on the building; as the buggy gets close, the
# camera's fixed angle means the QR moves out of frame BEFORE the buggy has
# actually reached the ideal stopping point. Instead of relying on
# continuous QR visibility, we time how long to keep driving (normally,
# via lane-following) after the LAST sighting before assuming we've arrived
# and stopping. Tune this number against your real track distances.
BLIND_APPROACH_DURATION = 2.4   # seconds to keep driving after QR was last seen

AVOID_RECOVERY_FRAMES = 6  # ~1.2s at 10Hz; camera blocked after obstacle clears

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

        # 1b. Green signboard detections (STRAIGHT mode blind fallback only).
        # Kept on a separate topic/callback from /edge_vectors on purpose -
        # see notes in _handle_straight.
        self.subscription_straight_vectors = self.create_subscription(
            EdgeVectors,
            '/straight_board_vectors',
            self.straight_vectors_callback,
            QOS_PROFILE_DEFAULT)
        self.latest_straight_vectors = None  # most recent board detection msg

        self.publisher_drive_mode = self.create_publisher(
            String,
            '/drive_mode',
            QOS_PROFILE_DEFAULT)

        self.drive_mode = DEFAULT_DRIVE_MODE      # "LANE_FOLLOW" or "LINE_FOLLOW"
        self.follow_side = DEFAULT_FOLLOW_SIDE    # "LEFT" or "RIGHT" (LINE_FOLLOW only)

        # Publish current mode so vision node adjusts crop ROI automatically
        mode_msg = String()
        mode_msg.data = self.drive_mode  # e.g., "STRAIGHT" or "LANE_FOLLOW"
        self.publisher_drive_mode.publish(mode_msg)

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
        self.recovery_frames_remaining = 0   # ← ADD THIS LINE
        self.last_avoid_turn = 0.0         # ← ADD THIS LINE
        self.frames_avoided = 0          # ← ADD THIS LINE
        self.patient_id = None
        self.hospital_id = None
        self.current_destination = None
        self.mission_completed = False
        # Racing line / late apex entry tracking
        self.horizontal_line_frames = 0
        self.apex_active = False   # True if the line vanished WHILE it was in apex/horizontal orientation
        self.apex_turn = 0.0       # the turn command to sustain while apex_active
        self.apex_blind_frames = 0
        self.straight_turn_direction = None  # Will hold "LEFT" or "RIGHT"
        self.straight_lost_frames = 0        # consecutive frames with no green board seen
        self.straight_board_seen = False     # has a board been acquired yet this STRAIGHT session
        self.revert_lane_frames = 0
        self.revert_armed = False  # Lock out reversion until intersection is entered
        self.pending_intersection_direction = None

        # ===== SERVER COMMUNICATION STATE (added) =====
        self.sign_to_building = {
            'A': 'PATIENT_1', 'B': 'PATIENT_2', 'C': 'PATIENT_3',
            'X': 'HOSPITAL_1', 'Y': 'HOSPITAL_2', 'Z': 'HOSPITAL_3',
        }
        self.building_to_sign = {v: k for k, v in self.sign_to_building.items()}
        self.server_uid = 0
        self.last_sent_qr = None     
        self.awaiting_hospital = False
        
        self.current_destination = 'PATIENT_1' # Start by looking for Patient 1
        self.waiting_for_ack = False
        self.server_retries = 0
        self.last_msg_send_time = 0.0
        # ==============================================
        # ===== END SERVER STATE =====

        # ===== QR APPROACH / STOP-IN-ZONE STATE (added) =====
        self.qr_last_seen_time = None     # timestamp of most recent QR sighting
        self.qr_approach_active = False   # True once we've seen a QR and are tracking approach
        self.stopped_for_patient = False  # True once fully stopped and reported for this building
        self.pending_letter = None        # letter code to send once we stop
        self.pending_building = None      # building name matching pending_letter
        # ===== END QR APPROACH STATE =====

        # ---- Driving mode ----
        # HOW TO SWITCH MODES FOR TESTING: change DEFAULT_DRIVE_MODE (and
        # DEFAULT_FOLLOW_SIDE if testing LINE_FOLLOW) at the top of this file
        # and restart the node. Later, sign_board_callback can set
        # self.drive_mode / self.follow_side directly instead.
        

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
        self.check_qr_approach()
        self.check_server_retries() # <--- ADD THIS LINE

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
            """Handles 5x retries at 1-second intervals using self.server_uid."""
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
                        # Increment UID so the next attempt won't reuse this UID
                        self.server_uid = (self.server_uid + 1) % 256

    def check_qr_approach(self):
        if not self.qr_approach_active or self.stopped_for_patient:
            return  

        elapsed = time.time() - self.qr_last_seen_time

        if elapsed >= BLIND_APPROACH_DURATION:
            self.target_speed = 0.0
            self.target_turn = 0.0  
            self.stopped_for_patient = True
            
            # Setup server retry state
            self.waiting_for_ack = True
            self.server_retries = 1
            self.last_msg_send_time = time.time()
            
            # Send initial message using current server_uid (without incrementing yet)
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

                # ===== SHARP TURN SPEED CONTROL =====
                sharpness = min(dx / max(dy, 0.001), 10.0)
                apex_speed = max(0.30, 0.70 - (sharpness * 0.04))

                # ====================================

                if line_center_x >= half_width:
                    # PHASE 2: Hit the Apex! (Hard Left Turn once track opens up)
                    self.target_turn = (0.34+0.006*dx/dy)
                    self.target_speed = LANE_SHARP_SPEED     # ← dynamic, not hardcoded
                else:
                    # PHASE 2: Hit the Apex! (Hard Right Turn)
                    self.target_turn = -(0.34+0.006*dx/dy)
                    self.target_speed = LANE_SHARP_SPEED    # ← dynamic, not hardcoded

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


    def straight_vectors_callback(self, message):
        """
        Stores the latest green-signboard detection (published only during
        STRAIGHT mode by edge_vectors_publisher). Kept on its own topic and
        callback deliberately - NOT merged into /edge_vectors - because
        _check_revert_to_lane_follow reads /edge_vectors as real lane-line
        midpoints to decide when to hand control back to LANE_FOLLOW. Mixing
        board detections into that stream previously caused it to revert off
        coincidental board geometry instead of genuinely reacquired lines.
        """
        self.latest_straight_vectors = message

    def _handle_straight(self, message, width, half_width):
        """
        STRAIGHT mode: primarily just lane-follows off the painted lines like
        normal (a straight corridor usually still has visible lane lines,
        same as everywhere else on the track) via _handle_lane_follow. Only
        when we're fully blind to lines - message.vector_count == 0 on the
        normal lane-line /edge_vectors stream, e.g. the open gap right at the
        intersection where there's no paint - do we fall back to steering off
        the green signboards published separately on /straight_board_vectors.

        `message` here is the normal lane-line EdgeVectors message (same one
        LANE_FOLLOW/LINE_FOLLOW use), NOT the board detection message.
        """
        if message.vector_count > 0:
            # Lines visible - simple lane-follow centering, same as LANE_FOLLOW.
            self.straight_lost_frames = 0
            self._handle_lane_follow(message, width, half_width)
            return

        # Blind to lines - fall back to the most recent green-board detection.
        board_msg = self.latest_straight_vectors

        if board_msg is None or board_msg.vector_count == 0:
            self.straight_lost_frames += 1

            if not self.straight_board_seen:
                # Never acquired a board yet this session - don't trust
                # whatever target_turn happens to be sitting from before the
                # mode switch (or from lane-follow a moment ago). Go
                # dead-straight until we actually see something to steer on.
                self.target_turn = 0.0
                self.target_speed = STRAIGHT_SPEED
            else:
                # Previously tracking a board, briefly lost it - hold the
                # last commanded turn, easing off speed the longer it persists.
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
        """
        Automatically reverts drive_mode back to LANE_FOLLOW only AFTER
        the buggy has entered the intersection (vector_count < 2) and subsequently
        re-acquires two valid parallel lane boundaries on exit.
        """
        if self.drive_mode in ["LINE_FOLLOW", "STRAIGHT"]:
            # STEP 1: Arm the reversion logic only after double lanes disappear/open up
            if not self.revert_armed:
                if message.vector_count < 2:
                    self.revert_armed = True
                    self.get_logger().info("Entered intersection (vector_count < 2). Reversion logic ARMED.")
                return  # Do not attempt to revert while still on the approach road

            # STEP 2: Once armed, look for dual parallel lanes to signal turn completion
            if message.vector_count == 2:
                v1_mid_x = (message.vector_1[0].x + message.vector_1[1].x) / 2.0
                v2_mid_x = (message.vector_2[0].x + message.vector_2[1].x) / 2.0
                track_width = abs(v1_mid_x - v2_mid_x)

                # Confirm track width is valid (not diverging cross-edges)
                if track_width <= (width * TRACK_WIDTH_DIVERGENCE_RATIO):
                    self.revert_lane_frames += 1
                    if self.revert_lane_frames >= 60:  # require 3 consecutive stable frames
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
        if self.obstacle_in_front or self.stopped_for_patient:
            return

        width = float(message.image_width)
        if width <= 0:
            return
        half_width = width / 2.0

        # --- Check if we have reached an intersection while a direction is pending ---
        if self.pending_intersection_direction and self.drive_mode == "LANE_FOLLOW":
            should_switch = False

            if self.pending_intersection_direction == "STRAIGHT":
                # Wait until lines clear out completely (vector_count == 0)
                if message.vector_count == 0:
                    should_switch = True
            else:
                # LINE_FOLLOW (LEFT/RIGHT): switch as soon as double lines break (vector_count < 2)
                if message.vector_count < 2:
                    should_switch = True

            if should_switch:
                self.get_logger().info(
                    f"Entering intersection (vector_count={message.vector_count}). "
                    f"Activating mode '{self.pending_intersection_direction}'."
                )
                self._switch_drive_mode(self.pending_intersection_direction)
                self.pending_intersection_direction = None  # Clear buffered direction

        # Check if we can switch back to LANE_FOLLOW post-turn
        self._check_revert_to_lane_follow(message, width)

        # Dispatch based on current mode
        if self.drive_mode == "LANE_FOLLOW":
            self._handle_lane_follow(message, width, half_width)
        elif self.drive_mode == "LINE_FOLLOW":
            self._handle_line_follow(message, width, half_width)
        elif self.drive_mode == "STRAIGHT":
            self._handle_straight(message, width, half_width)
        

    def lidar_callback(self, message):
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
            # Actively dodging
            self.obstacle_in_front = True
            self.recovery_frames_remaining = AVOID_RECOVERY_FRAMES
            self.frames_avoided += 1                         # count how long we dodged
        
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
            # Recovery: check if return path is clear first
            mid = len(front_sector) // 2
            right_valid = valid(front_sector[:mid])
            left_valid = valid(front_sector[mid:])
            left_clearance = min(left_valid) if left_valid else float('inf')
            right_clearance = min(right_valid) if right_valid else float('inf')
        
            return_dir = -self.last_avoid_turn   # opposite of dodge
            return_blocked = (
                (return_dir > 0 and min(left_valid)  < OBSTACLE_DISTANCE_THRESHOLD if left_valid  else False) or
                (return_dir < 0 and min(right_valid) < OBSTACLE_DISTANCE_THRESHOLD if right_valid else False)
            )
        
            self.obstacle_in_front = True
            self.recovery_frames_remaining -= 1
        
            if return_blocked:
                # Next cone is in return path → go straight, don't jiggle
                self.target_turn = 0.0
                self.target_speed = AVOID_SPEED * 0.8
            else:
                # Variable recovery: proportional to how long we actually dodged
                scale = min(self.frames_avoided / AVOID_RECOVERY_FRAMES, 1.0)
                recovery_turn = -self.last_avoid_turn * scale * 2
                self.target_turn = recovery_turn
                self.target_speed = AVOID_SPEED * 0.8        
        else:
            # Fully clear → reset everything, hand back to camera
            self.obstacle_in_front = False
            self.frames_avoided = 0              # reset for next obstacle

    def server_communication_callback(self, message):
        # Ignore messages not intended for this node (dest == 1)
        if message.dest != 1:
            return

        # 1. Handle incoming ACK from server (acknowledging a message WE sent)
        if message.ack == 1:
            if message.uid == self.server_uid:
                self.get_logger().info(f"Server ACKed our message UID={message.uid}")
                self.waiting_for_ack = False
                
                # Increment UID on successful server ACK
                self.server_uid = (self.server_uid + 1) % 256
            return

        # 2. Handle incoming target command from server (e.g., 'X', 'Y', 'Z')
        target_letter = message.msg.strip()
        building = self.sign_to_building.get(target_letter)
        if building is not None:
            self.current_destination = building
            self.awaiting_hospital = False

            # Resume driving
            self.stopped_for_patient = False
            self.qr_approach_active = False
            self.pending_letter = None
            self.pending_building = None

            self.get_logger().info(f"New destination received: {building} (letter '{target_letter}')")
            
            # Send ACK back to server (this will increment self.server_uid inside send_server_ack)
            self.send_server_ack(message.uid)

    def send_server_update(self, text_msg, uid):
        """Sends a data message to the server with a specific UID."""
        server_msg = ServerCommunication()
        server_msg.src = 1       
        server_msg.dest = 2      
        server_msg.uid = uid
        server_msg.ack = 0
        server_msg.msg = text_msg

        self.publisher_server.publish(server_msg)

    def send_server_ack(self, uid_to_ack):
        """Sends an ACK message back to the server for a received command."""
        server_msg = ServerCommunication()
        server_msg.src = 1       
        server_msg.dest = 2      
        server_msg.uid = uid_to_ack  # Identifies which incoming message we are ACKing
        server_msg.ack = 1
        server_msg.msg = ""
        self.publisher_server.publish(server_msg)

        # Increment UID whenever we transmit an ACK
        self.server_uid = (self.server_uid + 1) % 256
        self.get_logger().info(f"Sent ACK for UID={uid_to_ack}. Advanced internal server_uid to {self.server_uid}.")

    def qr_detection_callback(self, message):
        """
        Receives QR codes scanned from the buildings.
        """
        data = message.data.strip()

        if self.stopped_for_patient:
            return  # already stopped/handling this building, ignore further reads

        building = None
        for name in self.building_to_sign:
            if name in data:
                building = name
                break

        # Ignore if it's not a recognized building OR if it's the one we just did
        if building is None or building == self.last_sent_qr:
            return

        # NEW: Ignore if the building is NOT our current destination
        if building != self.current_destination:
            self.get_logger().info(f"Saw {building}, but heading to {self.current_destination}")
            return

        self.qr_last_seen_time = time.time()
        self.qr_approach_active = True
        self.pending_letter = self.building_to_sign[building]
        self.pending_building = building
        self.get_logger().info(f"Approaching target: {building}.")

    def sign_board_callback(self, message):
        """
        Parses traffic sign board detections using a dictionary lookup.
        """
        if self.stopped_for_patient or self.obstacle_in_front:
            return

        raw_data = message.data.strip().upper()

        # 1. Remove brackets/parentheses
        cleaned = raw_data
        for symbol in ["(", ")", "[", "]", "{", "}"]:
            cleaned = cleaned.replace(symbol, "")

        # 2. Split into entries (handles comma or space separation)
        entries = cleaned.replace(',', ' ').split()

        # 3. Build dictionary from Key:Value pairs (e.g., {'A': 'LEFT', 'B': 'STRAIGHT'})
        sign_dict = {}
        for entry in entries:
            if ':' in entry:
                key, val = entry.split(':', 1)
                sign_dict[key.strip()] = val.strip()

        target_building = self.current_destination  # e.g., 'PATIENT_1'
        target_letter = self.building_to_sign.get(target_building, '')  # e.g., 'A'

        # 4. Dictionary Lookup for target
        chosen_direction = sign_dict.get(target_letter) or sign_dict.get(target_building)

        # Fallback: handles space-separated lists like ['A', 'LEFT', 'B', 'STRAIGHT']
        if not chosen_direction:
            tokens = cleaned.replace(':', ' ').replace(',', ' ').split()
            valid_directions = ["LEFT", "RIGHT", "STRAIGHT"]
            for i in range(len(tokens) - 1):
                if tokens[i] in [target_building, target_letter] and tokens[i + 1] in valid_directions:
                    chosen_direction = tokens[i + 1]
                    break

        # Execute mode switch if direction was found
        # Replace the mode execution block at the end of sign_board_callback with this:
        if chosen_direction:
            # Buffer the direction instead of switching immediately
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
        if direction in ["LEFT", "RIGHT"]:
            self.drive_mode = "LINE_FOLLOW"
            self.follow_side = direction
        elif direction == "STRAIGHT":
            self.drive_mode = "STRAIGHT"
            self.straight_turn_direction = None
            self.straight_board_seen = False   # fresh session - no stale-turn hold
            self.straight_lost_frames = 0
            self.target_turn = 0.0             # clear any residual turn from LANE_FOLLOW

        # Lock out reversion until we enter the intersection (vector_count < 2)
        self.revert_lane_frames = 0
        self.revert_armed = False

        # Publish mode change to vision node
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