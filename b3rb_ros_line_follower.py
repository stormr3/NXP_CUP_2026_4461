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

# ===== STRAIGHT MODE (drive through an intersection without turning) =====
# Third drive mode. Unlike LANE_FOLLOW, the apex/horizontal-line trigger is
# COMPLETELY SUPPRESSED here - at a crossing, the near-horizontal fillets
# where your lane boundaries curve away into the cross street would
# otherwise read as "sharp corner, turn now" and pull the buggy off a route
# that was supposed to go straight through.
STRAIGHT_SPEED = 0.55            # speed while crossing an intersection straight
STRAIGHT_CENTERING_KP = 0.004    # gentle centering correction if a usable vertical boundary is visible
STRAIGHT_MAX_TURN = 0.25         # hard cap on any correction while in STRAIGHT - never let it become a real turn
STRAIGHT_VERTICAL_RATIO = 3.0    # dx <= dy * ratio counts as "vertical enough" to correct against

# ===== SIGN BOARD ROUTING / MANEUVER STATE MACHINE =====
SIGN_SET_SIZE = 6                # only act once a COMPLETE set of 6 sign entries has arrived
SIGN_BUFFER_TIMEOUT = 3.0        # seconds; a partial (<6) buffer older than this is stale, discard it
MANEUVER_COMPLETE_FRAMES = 3     # consecutive frames of clean 2-vertical-boundary lane needed to declare the maneuver finished
MANEUVER_MIN_DURATION = 1.5      # seconds; ignore completion before this, so we don't "finish" instantly at the entry line
MANEUVER_TIMEOUT = 8.0           # seconds; hard backstop - revert to LANE_FOLLOW even if completion never detected

# Positional fallback: if the sign tokens carry no explicit direction word,
# assume the 6 entries are laid out left-block / centre-block / right-block.
#
# >>> VERIFY THIS AGAINST A REAL `ros2 topic echo /sign_board_detection` <<<
# This mapping is an assumption from the board layout in the screenshot, NOT
# measured. If the real board orders its 6 tiles differently, this is the ONE
# dict to change - nothing else in the routing logic depends on the layout.
SIGN_POSITION_TO_DIRECTION = {
    0: "LEFT",
    1: "LEFT",
    2: "STRAIGHT",
    3: "STRAIGHT",
    4: "RIGHT",
    5: "RIGHT",
}
# ===== END SIGN BOARD ROUTING =====

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
BLIND_APPROACH_DURATION = 4.2   # seconds to keep driving after QR was last seen

AVOID_RECOVERY_FRAMES = 6  # ~1.2s at 10Hz; camera blocked after obstacle clears

class LineFollower(Node):
    """
    Core controller Node for the B3RB buggy.
    Three selectable driving modes (see drive_mode / follow_side):
      - LANE_FOLLOW: your original centering/single-line logic, extended with
        a fallback for when the two detected boundaries diverge too far to be
        a real lane pair (tight-turn cross-edge case). Default cruising mode.
      - LINE_FOLLOW: commits to ONE line and holds a gap to it; if it vanishes
        (turn apex), holds a turn arc toward it until reacquired. Used for
        sign-directed LEFT/RIGHT turns at intersections.
      - STRAIGHT: crosses an intersection without turning, with the apex
        trigger fully suppressed. Used for sign-directed STRAIGHT routing.

    LINE_FOLLOW and STRAIGHT are entered by sign_board_callback when a
    complete 6-entry sign set names a maneuver for the current destination,
    and automatically revert to LANE_FOLLOW once the maneuver completes.
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

        # ===== SIGN BOARD / MANEUVER STATE (added) =====
        self.sign_buffer = []             # accumulating sign entries until we have SIGN_SET_SIZE
        self.sign_buffer_time = 0.0       # timestamp of last entry added, for staleness expiry
        self.maneuver_active = False      # True while executing a sign-directed LEFT/RIGHT/STRAIGHT
        self.maneuver_direction = None    # "LEFT" / "RIGHT" / "STRAIGHT" currently being executed
        self.maneuver_start_time = 0.0    # for MANEUVER_MIN_DURATION and MANEUVER_TIMEOUT
        self.maneuver_confirm_frames = 0  # consecutive clean-lane frames seen so far
        # ===== END SIGN BOARD / MANEUVER STATE =====

        # ---- Driving mode ----
        # Set at boot from the DEFAULT_* constants; from then on
        # sign_board_callback drives these via set_maneuver() / end_maneuver().
        self.drive_mode = DEFAULT_DRIVE_MODE      # "LANE_FOLLOW", "LINE_FOLLOW" or "STRAIGHT"
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
        self.check_qr_approach()
        self.check_server_retries() # <--- ADD THIS LINE
        self.check_maneuver_timeout()  # backstop: runs on the timer, so it still
                                       # fires even if EdgeVectors stops arriving

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

    # ------------------ Maneuver State Machine ------------------

    def set_maneuver(self, direction):
        """
        Enters a sign-directed maneuver. Single place that flips drive_mode /
        follow_side, so the mode can never be set inconsistently from
        scattered call sites.
        """
        if direction == "STRAIGHT":
            self.drive_mode = "STRAIGHT"
        elif direction in ("LEFT", "RIGHT"):
            self.drive_mode = "LINE_FOLLOW"
            self.follow_side = direction
        else:
            self.get_logger().warn(f"set_maneuver: unknown direction '{direction}', ignoring.")
            return

        self.maneuver_active = True
        self.maneuver_direction = direction
        self.maneuver_start_time = time.time()
        self.maneuver_confirm_frames = 0

        # Clear stale apex/lost state so nothing left over from the approach
        # leaks into the maneuver and fights it.
        self.apex_active = False
        self.apex_blind_frames = 0
        self.horizontal_line_frames = 0
        self.lane_lost_frame_count = 0
        self.line_blind_frame_count = 0

        self.get_logger().info(f"MANEUVER START: {direction} (drive_mode={self.drive_mode})")

    def end_maneuver(self, reason):
        """Reverts to normal lane following once the maneuver is done."""
        if not self.maneuver_active:
            return
        self.get_logger().info(f"MANEUVER END ({self.maneuver_direction}): {reason} -> LANE_FOLLOW")

        self.drive_mode = "LANE_FOLLOW"
        self.maneuver_active = False
        self.maneuver_direction = None
        self.maneuver_confirm_frames = 0

        # Clear apex/lost state again on exit, same reasoning as on entry.
        self.apex_active = False
        self.apex_blind_frames = 0
        self.horizontal_line_frames = 0
        self.lane_lost_frame_count = 0
        self.line_blind_frame_count = 0

    def check_maneuver_timeout(self):
        """
        Hard backstop, run on the 10Hz timer (NOT in edge_vectors_callback).
        If EdgeVectors stops arriving entirely mid-maneuver, the vision-based
        completion check below would never fire and we'd be stuck in the
        maneuver mode forever - this guarantees an exit.
        """
        if not self.maneuver_active:
            return
        if (time.time() - self.maneuver_start_time) > MANEUVER_TIMEOUT:
            self.end_maneuver("timeout")

    def check_maneuver_complete(self, message, width, half_width):
        """
        Vision-based completion: we consider the maneuver finished once we're
        back on a clean, normal lane - two boundaries, both roughly vertical,
        a sane distance apart - for MANEUVER_COMPLETE_FRAMES in a row.

        MANEUVER_MIN_DURATION guards the entry: right as we enter an
        intersection we may still briefly see a clean lane behind/around us,
        and completing instantly would abort the maneuver before it started.
        """
        if not self.maneuver_active:
            return

        if (time.time() - self.maneuver_start_time) < MANEUVER_MIN_DURATION:
            return  # too early to call it done

        if message.vector_count != 2:
            self.maneuver_confirm_frames = 0
            return

        v1_dx = abs(message.vector_1[0].x - message.vector_1[1].x)
        v1_dy = abs(message.vector_1[0].y - message.vector_1[1].y)
        v2_dx = abs(message.vector_2[0].x - message.vector_2[1].x)
        v2_dy = abs(message.vector_2[0].y - message.vector_2[1].y)

        both_vertical = (v1_dx <= v1_dy * STRAIGHT_VERTICAL_RATIO
                         and v2_dx <= v2_dy * STRAIGHT_VERTICAL_RATIO)

        v1_mid_x = (message.vector_1[0].x + message.vector_1[1].x) / 2.0
        v2_mid_x = (message.vector_2[0].x + message.vector_2[1].x) / 2.0
        sane_width = abs(v1_mid_x - v2_mid_x) <= (width * TRACK_WIDTH_DIVERGENCE_RATIO)

        if both_vertical and sane_width:
            self.maneuver_confirm_frames += 1
        else:
            self.maneuver_confirm_frames = 0

        if self.maneuver_confirm_frames >= MANEUVER_COMPLETE_FRAMES:
            self.end_maneuver("lane reacquired")

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

                # Your tuning (0.33 + 0.006*dx/dy) kept EXACTLY as-is. The only
                # change is dy is floored at 0.001 and the result is clamped to
                # the valid [-1, 1] control range: at a tight apex dy can reach
                # ~0, and this branch writes self.target_turn directly (it does
                # NOT go through rover_move_manual_mode's clamp), so an
                # unbounded value would be published raw into the Joy message.
                # In your normal operating range this changes nothing.
                apex_ratio = dx / max(dy, 0.001)
                apex_turn_magnitude = max(-1.0, min(1.0, 0.33 + 0.006 * apex_ratio))

                if line_center_x >= half_width:
                    # PHASE 2: Hit the Apex! (Hard Left Turn once track opens up)
                    self.target_turn = apex_turn_magnitude
                    self.target_speed = apex_speed     # ← dynamic, not hardcoded
                else:
                    # PHASE 2: Hit the Apex! (Hard Right Turn)
                    self.target_turn = -apex_turn_magnitude
                    self.target_speed = apex_speed     # ← dynamic, not hardcoded

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
                self.target_turn = sign * min(1.0, 0.1 + dx / max(dy, 0.001) * 0.1)
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

    # ------------------ Mode: STRAIGHT ------------------

    def _handle_straight(self, message, width, half_width):
        """
        Cross the intersection without turning.

        The whole point of this mode: the apex / horizontal-line trigger is
        NEVER consulted here. At a crossing, the rounded fillets where your
        lane boundaries curve away into the cross street are near-horizontal
        AND appear on both sides pointing opposite ways - so LANE_FOLLOW's
        apex rule would pull a hard turn in whichever direction it happened
        to detect that frame. In STRAIGHT we treat every horizontal edge as
        paint to drive over, not as a corner.

        Vertical boundaries are still used, but only for gentle centering,
        capped at STRAIGHT_MAX_TURN so a correction can never grow into a
        real turn.
        """
        count = message.vector_count

        usable_mids = []
        for i in range(count):
            v = message.vector_1 if i == 0 else message.vector_2
            dx = abs(v[0].x - v[1].x)
            dy = abs(v[0].y - v[1].y)
            if dx <= dy * STRAIGHT_VERTICAL_RATIO:      # vertical enough to trust
                usable_mids.append((v[0].x + v[1].x) / 2.0)

        if len(usable_mids) == 2:
            # Both boundaries visible and vertical - centre between them.
            target_x = (usable_mids[0] + usable_mids[1]) / 2.0
            error = half_width - target_x
            correction = STRAIGHT_CENTERING_KP * error
        elif len(usable_mids) == 1:
            # One vertical boundary - hold the normal offset from it.
            offset = width * LANE_GAP_OFFSET_RATIO
            mid_x = usable_mids[0]
            target_x = mid_x + offset if mid_x < half_width else mid_x - offset
            error = half_width - target_x
            correction = STRAIGHT_CENTERING_KP * error
        else:
            # Nothing vertical to steer by (all horizontal fillets, or blank
            # mid-crossing). This is the normal state in the middle of an
            # intersection - just go straight.
            correction = 0.0

        self.target_turn = max(-STRAIGHT_MAX_TURN, min(STRAIGHT_MAX_TURN, correction))
        self.target_speed = STRAIGHT_SPEED

    # ------------------ Sign Board Parsing ------------------

    def parse_sign_message(self, data):
        """
        Splits one incoming /sign_board_detection message into tokens.
        Tolerant of comma, semicolon, pipe or whitespace separation, so it
        works whether the recognizer emits "A,B,C,X,Y,Z" or "A B C X Y Z"
        in one message, or one letter per message.
        """
        cleaned = data.strip().upper()
        for sep in (',', ';', '|'):
            cleaned = cleaned.replace(sep, ' ')
        return [t for t in cleaned.split() if t]

    def direction_for_destination(self, board, destination):
        """
        Given a complete 6-entry sign board and the building we're currently
        heading to, return "LEFT" / "RIGHT" / "STRAIGHT", or None if this
        board doesn't mention our destination.

        Two resolution strategies, in order:
          1. EXPLICIT - the token itself names a direction (e.g. "A_LEFT",
             "A:STRAIGHT"). Preferred, since it doesn't rely on tile order.
          2. POSITIONAL - fall back to SIGN_POSITION_TO_DIRECTION using the
             token's index in the 6-entry set.
        """
        if destination is None:
            return None
        target_letter = self.building_to_sign.get(destination)
        if target_letter is None:
            return None

        for idx, token in enumerate(board):
            if target_letter not in token:
                continue

            # Strategy 1: explicit direction word embedded in the token.
            for d in ("STRAIGHT", "LEFT", "RIGHT"):
                if d in token:
                    return d

            # Strategy 2: positional fallback.
            return SIGN_POSITION_TO_DIRECTION.get(idx)

        return None  # our destination isn't on this board

    # ------------------ Callback Implementations ------------------

    def edge_vectors_callback(self, message):
        if self.obstacle_in_front:
            # LIDAR has priority while dodging; don't fight the avoidance maneuver.
            return

        if self.stopped_for_patient:
            # We've deliberately stopped in the patient/hospital zone; don't
            # let lane-following override that until we resume the mission.
            return

        """
        Receives lane boundaries from the camera vector extractor and
        dispatches to whichever mode is active (see drive_mode / follow_side).
        """
        width = float(message.image_width)
        if width <= 0:
            return  # guard against a boot/glitch frame with no valid image_width
        half_width = width / 2.0

        # If a sign-directed maneuver is running, check whether it's finished
        # BEFORE dispatching, so the moment it completes we drive this frame
        # in LANE_FOLLOW rather than one stale frame of the old mode.
        self.check_maneuver_complete(message, width, half_width)

        if self.drive_mode == "LANE_FOLLOW":
            self._handle_lane_follow(message, width, half_width)
        elif self.drive_mode == "LINE_FOLLOW":
            self._handle_line_follow(message, width, half_width)
        elif self.drive_mode == "STRAIGHT":
            self._handle_straight(message, width, half_width)
        else:
            # SAFETY GUARD: without this, an unrecognized drive_mode would
            # match no branch, nothing would write target_speed/target_turn,
            # and publish_drive_commands would keep republishing the LAST
            # command forever - i.e. if it happened mid-turn, the buggy
            # would drive in a circle indefinitely.
            self.get_logger().warn(
                f"Unknown drive_mode '{self.drive_mode}' - reverting to LANE_FOLLOW.")
            self.drive_mode = "LANE_FOLLOW"
            self._handle_lane_follow(message, width, half_width)

    def lidar_callback(self, message):
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

            # Destination changed - any buffered sign data was read against
            # the OLD destination, so it's meaningless now. Drop it.
            self.sign_buffer = []

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
        Receives traffic sign boards and routes the buggy at intersections.

        Flow:
          1. Accumulate incoming entries into self.sign_buffer.
          2. Do NOTHING until a COMPLETE set of SIGN_SET_SIZE (6) has arrived -
             a partial read is unreliable and could route us the wrong way.
          3. Look up self.current_destination on the completed board.
          4. Switch drive_mode / follow_side to the required maneuver.
          5. The maneuver auto-reverts to LANE_FOLLOW via
             check_maneuver_complete() / check_maneuver_timeout().
        """
        tokens = self.parse_sign_message(message.data)
        if not tokens:
            return

        now = time.time()

        # Expire a stale partial buffer, so leftover entries from a PREVIOUS
        # intersection can never combine with new ones to form a bogus "set".
        if self.sign_buffer and (now - self.sign_buffer_time) > SIGN_BUFFER_TIMEOUT:
            self.get_logger().info("Stale partial sign buffer discarded.")
            self.sign_buffer = []
        self.sign_buffer_time = now

        if len(tokens) >= SIGN_SET_SIZE:
            # Whole board arrived in one message.
            self.sign_buffer = tokens[:SIGN_SET_SIZE]
        else:
            # Entries trickling in one/few at a time - accumulate, preserving
            # arrival order (position matters for the positional fallback).
            for t in tokens:
                if t not in self.sign_buffer:
                    self.sign_buffer.append(t)

        if len(self.sign_buffer) < SIGN_SET_SIZE:
            self.get_logger().info(
                f"Sign board incomplete ({len(self.sign_buffer)}/{SIGN_SET_SIZE}), waiting.")
            return

        board = self.sign_buffer[:SIGN_SET_SIZE]
        self.get_logger().info(f"Complete sign board: {board}")
        self.sign_buffer = []   # consumed - next intersection starts fresh

        if self.maneuver_active:
            self.get_logger().info("Maneuver already in progress, ignoring this board.")
            return

        direction = self.direction_for_destination(board, self.current_destination)

        if direction is None:
            self.get_logger().info(
                f"Destination {self.current_destination} not found on this board - staying in LANE_FOLLOW.")
            return

        self.get_logger().info(
            f"Board routes {self.current_destination} -> {direction}")
        self.set_maneuver(direction)


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
