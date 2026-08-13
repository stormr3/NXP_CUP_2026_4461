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

# =============================================================================
#  MERGE SUMMARY - what changed vs. the pre-parking version
#  Every added/changed block is fenced with:  # ===== PARKING: ... =====
#  Nothing outside those fences was modified. Full list:
#
#   1. NEW module-level constants block  ("PARKING PARAMETERS")
#   2. NEW module-level helpers          (wrap_pi, ang_diff, polar_to_xy)
#   3. __init__                          + parking state variables
#   4. publish_drive_commands            + one call to check_parking_tick()
#   5. edge_vectors_callback             + one guard line at the top
#   6. lidar_callback                    + parking block at the top
#   7. server_communication_callback     'OK' branch now calls start_parking()
#   8. qr_detection_callback             + one guard line
#   9. sign_board_callback               + one guard line
#  10. NEW parking methods               (whole new section at the end of class)
#
#  All original driving logic (lane follow, line follow, straight, apex memory,
#  revert lock, obstacle avoidance, QR blind timing, server retries, sign
#  parsing) is byte-for-byte unchanged.
# =============================================================================

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


# =============================================================================
# ===== COLLISION RECOVERY: PARAMETERS (NEW BLOCK) =====
#
#  PURPOSE
#  -------
#  If the buggy wedges itself against a wall / cone / building - anywhere on
#  the track, or mid-park - back out, change the approach angle, and carry on.
#  This is a safety net under the WHOLE run, not a parking feature.
#
#  THE HARD PART IS DETECTION, NOT THE REVERSE
#  -------------------------------------------
#  Backing up is three lines. Knowing you are stuck is the real problem: there
#  is no contact sensor and no /collision topic in this stack, so "I have hit
#  something" must be INFERRED. And a false positive is expensive - reversing
#  mid-track when nothing is wrong ruins an otherwise clean run.
#
#  THE SIGNAL WE USE (LIDAR only - no new subscriptions, works today):
#      "I am commanding real forward speed,
#       something is very close in front,
#       and that distance is NOT changing."
#
#  All three must hold, for STUCK_CONFIRM_FRAMES in a row. The third clause is
#  what makes this safe: if you were merely approaching a wall, the range would
#  be shrinking every frame. A range pinned flat while you push forward means
#  you are physically against something.
#
#  FALSE-POSITIVE GATES (each kills one specific "not actually stuck" case):
#    - commanded speed below STUCK_MIN_CMD_SPEED  -> stopped on purpose
#    - stopped_for_patient                        -> parked at a QR building
#    - park_state in CREEP/SETTLE/PARKED/HOLD     -> crawling at a rail IS the
#                                                    goal during parking
#    - cooldown after a recovery                  -> don't re-fire on the same
#                                                    event we just handled
# =============================================================================

RECOVERY_ENABLED = True             # master switch - set False to disable entirely

# ---- Detection ---------------------------------------------------------------
STUCK_MIN_CMD_SPEED = 0.15          # must be commanding at least this to arm
STUCK_FRONT_RANGE = 0.30            # metres; this close = probably touching  [MEASURE]
STUCK_FRONT_SECTOR_HALF_DEG = 30.0  # front cone watched for the stuck test
STUCK_RANGE_JITTER = 0.03           # metres; range flatter than this = pinned
STUCK_CONFIRM_FRAMES = 8            # ~0.8 s at 10 Hz before we believe it

# ---- Recovery maneuver -------------------------------------------------------
RECOVERY_REVERSE_SPEED = -0.30      # negative = reverse (your Joy contract allows it)
RECOVERY_COUNTER_STEER = 0.60       # steer magnitude while reversing

# WHY COUNTER-STEER: if you hit something while turning left, reversing STRAIGHT
# puts you back on the same line and you hit it again. Reversing with opposite
# lock swings the nose away, so the next forward attempt has a different angle.
# Same thing you do in a car after misjudging a turn.

RECOVERY_REVERSE_BASE_TIME = 1.0    # seconds of reverse on the 1st attempt
RECOVERY_REVERSE_TIME_STEP = 0.6    # + this much per escalation
RECOVERY_PAUSE_TIME = 0.4           # seconds stopped before reassessing
RECOVERY_CLEAR_RANGE = 0.60         # front range that counts as "path is open again"

# ---- Loop protection ---------------------------------------------------------
# The classic failure: hit wall -> reverse -> drive -> hit same wall -> forever,
# burning the whole run. So attempts ESCALATE (longer reverse, flipped steer)
# and then STOP.
RECOVERY_MAX_ATTEMPTS = 3           # after this, give up and drive on
RECOVERY_COOLDOWN = 2.0             # seconds of detector blackout after recovery
RECOVERY_ATTEMPT_RESET = 15.0       # seconds of clean driving that resets the counter

# States in which the CAMERA must be ignored (recovery owns the wheel)
RECOVERY_STATES_ACTIVE = ('REVERSE', 'PAUSE')

# ===== END COLLISION RECOVERY PARAMETERS =====


# =============================================================================
# ===== PARKING: PARAMETERS (NEW BLOCK) =====
#
#  Runs only AFTER the server says 'OK' (all patients delivered).
#
#  WHY LIDAR AND NOT CAMERA:
#  The parking pad is a big WHITE surface on grey ground. edge_vectors will
#  happily lock onto its boundary and fight the maneuver. The cone/rail
#  dividers instead give LIDAR a distinctive "comb": near returns (cones)
#  separated by deep returns (open bays). That signature is lighting-
#  independent. The camera is not used for parking at all.
#
#  BEARING CONVENTION (derived from your own avoidance sector maths:
#  FRONT_SECTOR is 7/18..11/18 of the array and [:mid] is treated as RIGHT,
#  which is only true if index 0 == -180 deg):
#      index frac 0.00 -> -180 deg  (rear)
#      index frac 0.25 ->  -90 deg  (RIGHT)
#      index frac 0.50 ->    0 deg  (FRONT)
#      index frac 0.75 -> +/-90 deg (LEFT)
#  The parking code below uses angle_min + i*angle_increment instead of index
#  fractions, so it stays correct even if that assumption is ever wrong.
# =============================================================================

# ---- Which side of the track the parking pad sits on ------------------------
# Watch the buggy's approach in Gazebo: if the pad passes on the driver's LEFT,
# leave this as "LEFT". This one constant flips BOTH the search sector and the
# steering direction, so it is the only thing you change to mirror the maneuver.
PARK_SIDE = "LEFT"                  # "LEFT" or "RIGHT"          [MEASURE]

# ---- Vehicle envelope --------------------------------------------------------
PARK_VEHICLE_WIDTH = 0.22           # metres, widest point        [MEASURE]
PARK_SAFETY_MARGIN = 0.06           # metres clearance per side   [MEASURE]

# ---- Speeds ------------------------------------------------------------------
# Parking is a precision maneuver. Raising these is the fastest way to hit a cone.
PARK_SEARCH_SPEED_CAP = 0.45        # cap on lane-follow speed while hunting a bay
PARK_ENTRY_SPEED = 0.28             # speed during the turn-in arc
PARK_CREEP_SPEED = 0.22             # speed while creeping to final depth
PARK_TURN_FULL = 1.0                # full steering lock

# ---- Bay detection: the "comb" ----------------------------------------------
# A beam on the park side counts as STRUCTURE (cone / rail) below this range.
# Set it from ONE measurement: park manually beside the pad, `ros2 topic echo
# /scan`, read the range to the nearest cone, use ~1.5x that.
PARK_CONE_RANGE_MAX = 1.60          # metres                      [MEASURE]

# Minimum bay width, cone-to-cone, measured metrically in Cartesian space.
# NOT an angular threshold - angular width silently changes meaning with range.
PARK_GAP_MIN_WIDTH = PARK_VEHICLE_WIDTH + 2.0 * PARK_SAFETY_MARGIN   # 0.34 m

# Upper bound. THIS is what rejects "I have driven past the end of the pad and
# am now staring at empty field".
PARK_GAP_MAX_WIDTH = 1.20           # metres                      [MEASURE]

# A bay must be genuinely DEEPER than the cones bounding it, otherwise it is
# not an opening, just a seam between two objects at the same range.
PARK_BAY_MIN_DEPTH_GAIN = 0.25      # metres
PARK_GAP_MIN_BEAMS = 3              # reject single-beam sensor noise

# ---- Sectors -----------------------------------------------------------------
PARK_SEARCH_SECTOR_CENTER_DEG = 90.0    # abeam, on the park side
PARK_SEARCH_SECTOR_HALF_DEG = 55.0
PARK_ENTRY_SECTOR_CENTER_DEG = 50.0     # widened: the bay sweeps forward as we turn
PARK_ENTRY_SECTOR_HALF_DEG = 70.0
PARK_FRONT_SECTOR_HALF_DEG = 18.0       # for the "how deep am I" stop test
PARK_CENTER_SECTOR_HALF_DEG = 25.0      # around +/-90 deg, for in-bay centering

# ---- Commit geometry ---------------------------------------------------------
# Forward distance (vehicle-frame +x) at which we stop driving past the pad and
# begin the arc. An Ackermann vehicle MUST start turning before the bay is
# exactly abeam, or the rear inner wheel cuts the corner and clips the divider.
#     First estimate:  COMMIT_LEAD_X ~ R_min = wheelbase / tan(max_steer_angle)
# Measure both, compute R_min, start there, tune down until entry is clean.
PARK_COMMIT_LEAD_X = 0.45           # metres                      [MEASURE]
PARK_BAY_MIN_X = 0.02               # bays behind this are "already passed"

# ---- Entry arc completion ----------------------------------------------------
PARK_ENTRY_ALIGNED_DEG = 18.0       # |bay bearing| below this = pointed at the bay

# ---- Final stop test ---------------------------------------------------------
PARK_STOP_FRONT_RANGE = 0.35        # metres to the far rail      [MEASURE]
PARK_CENTER_DEADBAND = 0.10         # metres of tolerated L/R asymmetry
PARK_CENTER_KP = 1.2                # P-gain for in-bay centering
PARK_CENTER_TURN_CLAMP = 0.45       # keep the centering controller off full lock

# ---- Timeouts (every state MUST be able to give up) --------------------------
# A buggy frozen mid-maneuver scores worse than one that halts cleanly.
PARK_START_DELAY = 1.0              # pause at the last hospital before driving off
PARK_SEARCH_TIMEOUT = 30.0          # seconds hunting before giving up
PARK_ENTRY_TIMEOUT = 5.0            # seconds of arc before assuming it's done
PARK_CREEP_TIMEOUT = 7.0            # seconds creeping before stopping in place
PARK_SETTLE_DURATION = 1.5          # seconds of enforced zero before PARKED

# States in which the CAMERA must be ignored (the white pad reads as lane paint)
PARK_STATES_CAMERA_OFF = ('HOLD', 'ENTRY', 'CREEP', 'SETTLE', 'PARKED', 'ABORT')


# ===== PARKING: GEOMETRY HELPERS (NEW BLOCK) =====

def wrap_pi(a):
    """Wrap an angle into (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def ang_diff(a, b):
    """Smallest signed difference a - b, wrapped into (-pi, pi]."""
    return wrap_pi(a - b)


def polar_to_xy(r, theta):
    """Polar (range, bearing) -> Cartesian (x forward, y left) in vehicle frame."""
    return (r * math.cos(theta), r * math.sin(theta))

# ===== END PARKING PARAMETERS / HELPERS =====


class LineFollower(Node):
    """
    Core controller Node for the B3RB buggy.
    Two selectable driving modes (see drive_mode / follow_side):
      - LANE_FOLLOW: your original centering/single-line logic, extended with
        a fallback for when the two detected boundaries diverge too far to be
        a real lane pair (tight-turn cross-edge case).
      - LINE_FOLLOW: commits to ONE line and holds a gap to it; if it vanishes
        (turn apex), holds a turn arc toward it until reacquired.

    ===== PARKING (added) =====
    A second, independent FSM handles end-of-mission parking. It is dormant
    (park_state == 'IDLE') for the entire delivery run and costs nothing.
    See the PARKING section at the bottom of this class.
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
        self.previous_destination = 'PATIENT_1'  # Tracks previous node for rollback
        self.patients_delivered = 0             # Delivery counter
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

        # ===== PARKING: STATE VARIABLES (NEW BLOCK) =====
        # 'IDLE' for the whole delivery run - the parking FSM is completely
        # dormant and cannot affect anything until start_parking() is called.
        self.park_state = 'IDLE'
        self.park_state_entry_time = time.time()

        # +1 for LEFT, -1 for RIGHT. Does double duty: aims the search sector
        # AND signs the steering command, so PARK_SIDE flips both together.
        self.park_side_sign = 1.0 if PARK_SIDE == "LEFT" else -1.0

        # Once a bay is chosen we track THAT bay frame-to-frame by bearing
        # continuity, so a neighbouring slot can't steal the lock mid-arc.
        self.park_locked_bay = None
        self.park_locked_bay_bearing = None
        self.park_latest_scan = None
        # ===== END PARKING STATE =====

        # ===== COLLISION RECOVERY: STATE VARIABLES (NEW BLOCK) =====
        # 'IDLE' whenever we are driving normally. Only leaves IDLE when the
        # three-clause stuck test below fires.
        self.recovery_state = 'IDLE'
        self.recovery_state_entry_time = time.time()

        self.recovery_latest_scan = None      # most recent scan, for the PAUSE check
        self.recovery_stuck_frames = 0        # consecutive frames the test held
        self.recovery_range_history = []      # recent front ranges, for the
                                              # "is the distance changing?" test
        self.recovery_attempts = 0            # escalation level (resets after
                                              # RECOVERY_ATTEMPT_RESET clean secs)
        self.recovery_last_end_time = 0.0     # for the cooldown blackout
        self.recovery_counter_turn = 0.0      # steer held during the reverse
        self.recovery_park_state_at_trip = None   # so we can resume parking right
        self.recovery_given_up = False        # max attempts hit - stop trying
        # ===== END COLLISION RECOVERY STATE =====

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

        # ===== PARKING: ONE ADDED LINE =====
        # Deliberately called LAST. Whoever writes to target_speed/target_turn
        # last before the publish below wins, so running the parking tick here
        # gives parking FINAL AUTHORITY over the camera and the LIDAR avoidance,
        # without needing to touch either of them.
        self.check_parking_tick()
        # ===================================

        # ===== COLLISION RECOVERY: ONE ADDED LINE =====
        # Called AFTER parking, so it is the LAST writer of all. If we are
        # physically wedged against something, nothing else's opinion matters -
        # not the camera, not avoidance, not the park maneuver. Getting unstuck
        # comes first. This one line is what puts recovery at the top of the
        # priority ladder without touching any other controller.
        self.check_recovery_tick()
        # ==============================================

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
        # ===== PARKING: ONE ADDED GUARD =====
        # Once parking has begun, the delivery blind-approach timer must not be
        # able to slam the buggy to a stop mid-maneuver.
        if self.park_state != 'IDLE':
            return
        # ====================================

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
                    self.target_turn = (0.4+0.006*dx/dy)
                    self.target_speed = LANE_SHARP_SPEED     # ← dynamic, not hardcoded
                else:
                    # PHASE 2: Hit the Apex! (Hard Right Turn)
                    self.target_turn = -(0.4+0.006*dx/dy)
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
                    if self.revert_lane_frames >= 30:  # require 3 consecutive stable frames
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
        # ===== COLLISION RECOVERY: ONE ADDED GUARD =====
        # While reversing out of a collision the camera must be ignored
        # completely. Its steering opinion is based on a forward-driving model
        # and is actively wrong when the buggy is backing up.
        if self.recovery_state in RECOVERY_STATES_ACTIVE:
            return
        # ===============================================

        # ===== PARKING: ONE ADDED GUARD =====
        # The parking pad is a large WHITE surface. edge_vectors WILL read its
        # boundary as lane paint and steer the buggy back out of the bay.
        # From ENTRY onward the camera is therefore ignored entirely.
        #
        # NOTE: 'SEARCH' is deliberately NOT in this list. While hunting for a
        # bay the buggy still needs lane following to stay on the track.
        if self.park_state in PARK_STATES_CAMERA_OFF:
            return
        # ====================================

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
        # ===== COLLISION RECOVERY: ADDED BLOCK, ABOVE EVERYTHING =====
        # Recovery gets the scan before parking and before avoidance.
        #
        # Ordering rationale: if we are already wedged, obstacle avoidance has
        # ALREADY failed - it is the thing that was supposed to prevent this.
        # Letting it keep steering while we are pinned just grinds against the
        # obstacle. So recovery pre-empts it entirely.
        self.recovery_latest_scan = message

        if self.recovery_state in RECOVERY_STATES_ACTIVE:
            return          # maneuver is running from the timer; no sensing needed

        self._recovery_update_detector(message)
        if self.recovery_state in RECOVERY_STATES_ACTIVE:
            return          # tripped on this very frame
        # ===== END COLLISION RECOVERY BLOCK =====

        # ===== PARKING: ADDED BLOCK AT TOP OF CALLBACK =====
        # Parking gets first look at every scan.
        #
        # WHY THIS MATTERS MOST: your avoidance fires on anything inside
        # OBSTACLE_DISTANCE_THRESHOLD = 0.65 m and steers AWAY from it. Every
        # parking cone is well inside that. Left enabled during the maneuver,
        # the avoidance FSM would steer out of the bay while the parking FSM
        # steers into it, and the buggy would oscillate past the pad forever.
        #
        # So: during ENTRY/CREEP the avoidance below is skipped entirely.
        # During SEARCH it is KEPT ON - the pad cones sit abeam, outside the
        # +/-40 deg front sector, so they don't trigger it, and real obstacles
        # on the way to the pad still need dodging.
        if self.park_state != 'IDLE':
            self.park_latest_scan = message

            if self.park_state == 'SEARCH':
                self._park_do_search(message)
                if self.park_state != 'SEARCH':
                    return          # committed to ENTRY this very frame
                # otherwise fall through to normal obstacle avoidance below
            elif self.park_state == 'ENTRY':
                self._park_do_entry(message)
                return
            elif self.park_state == 'CREEP':
                self._park_do_creep(message)
                return
            else:
                return              # HOLD / SETTLE / PARKED / ABORT: no LIDAR work
        # ===== END PARKING BLOCK =====

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

        # 2. Handle incoming server command / response
        raw_msg = message.msg.strip().upper()

        # Mission Complete Response
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

            # ===== PARKING: ONE ADDED CALL =====
            # This is the ONLY trigger for the parking FSM. 'OK' from the server
            # is the authoritative "all patients delivered" signal - a hard,
            # server-confirmed fact, not a heuristic guess from a QR sighting.
            self.start_parking()
            # ===================================

        # Invalid Dropoff / Target Response -> Revert & Resume
        elif raw_msg == "INVALID":
            self.get_logger().warn(
                f"Server returned 'INVALID'. Reverting destination from '{self.current_destination}' "
                f"to '{self.previous_destination}' and resuming movement."
            )
            # Roll back destination and resume drive
            self.current_destination = self.previous_destination
            self.stopped_for_patient = False
            self.qr_approach_active = False
            self.pending_letter = None
            self.pending_building = None
            
            self.send_server_ack(message.uid)

        # New Destination Command (e.g., 'A', 'B', 'C', 'X', 'Y', 'Z')
        else:
            building = self.sign_to_building.get(raw_msg)
            if building is not None:
                # Store history before updating
                self.previous_destination = self.current_destination
                self.current_destination = building
                self.awaiting_hospital = False

                # Increment delivery count when returning from Hospital to Patient
                if "PATIENT" in building and "HOSPITAL" in self.previous_destination:
                    self.patients_delivered += 1
                    self.get_logger().info(f"Patient delivery confirmed! Count: {self.patients_delivered}/3")

                # Resume driving toward new target
                self.stopped_for_patient = False
                self.qr_approach_active = False
                self.pending_letter = None
                self.pending_building = None

                self.get_logger().info(f"New destination received: {building} (letter '{raw_msg}')")
                
                # Send ACK back to server
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
        # ===== PARKING: ONE ADDED GUARD =====
        # After the mission is over, a stray QR read must not re-arm the
        # delivery blind-approach machinery mid-park.
        if self.park_state != 'IDLE':
            return
        # ====================================

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
        # ===== PARKING: ONE ADDED GUARD =====
        # No more routing decisions once parking has started - a buffered turn
        # direction firing mid-maneuver would hijack drive_mode.
        if self.park_state != 'IDLE':
            return
        # ====================================

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

        if len(sign_dict) < 6:
            return

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

    # =========================================================================
    # ===== PARKING: ENTIRE NEW SECTION BELOW =====
    #
    #  Nothing above this line calls into here except:
    #     - publish_drive_commands()        -> check_parking_tick()
    #     - lidar_callback()                -> _park_do_search/_entry/_creep()
    #     - server_communication_callback() -> start_parking()
    #  and four one-line guards. Delete this section + those five hooks and the
    #  file reverts exactly to the pre-parking version.
    #
    #  THE FSM
    #  -------
    #     IDLE   -- dormant for the whole delivery run
    #       | (server sends 'OK')
    #     HOLD   -- stay stopped at the last hospital for PARK_START_DELAY,
    #       |       so the final ACK gets out and the stop is visible
    #     SEARCH -- LANE FOLLOWING STILL DRIVES. Parking only watches the park
    #       |       side for a bay in the LIDAR comb, and decides when to commit
    #     ENTRY  -- full lock toward the bay; hold the arc until the bay swings
    #       |       round into the front sector. Camera + avoidance OFF.
    #     CREEP  -- forward slowly, steering to keep left/right cone clearances
    #       |       symmetric, until the far rail is close
    #     SETTLE -- assert zero for PARK_SETTLE_DURATION
    #       |
    #     PARKED -- terminal. Zero published forever.
    #     ABORT  -- any unrecoverable timeout: halt safely. Stopping near the
    #               pad scores better than driving off it.
    # =========================================================================

    # ---- FSM plumbing --------------------------------------------------------

    def start_parking(self):
        """
        Public entry point. Called once, from the server 'OK' branch.

        Guarded against re-entry: a duplicate 'OK' (the server retries too)
        would otherwise reset the FSM and restart the arc from the wrong pose.
        """
        if self.park_state != 'IDLE':
            self.get_logger().warn(
                f"start_parking() ignored - already in park_state '{self.park_state}'.")
            return
        self.get_logger().info("=== ALL DELIVERIES COMPLETE - BEGINNING PARKING ===")
        self._park_transition('HOLD')

    def _park_transition(self, new_state):
        """Single choke point for park state changes, so every entry is timestamped."""
        self.get_logger().info(f"PARK FSM: {self.park_state} -> {new_state}")
        self.park_state = new_state
        self.park_state_entry_time = time.time()

    def _park_time_in_state(self):
        return time.time() - self.park_state_entry_time

    def check_parking_tick(self):
        """
        The TIME-driven half of the parking FSM, run from the 10 Hz timer.
        (The SENSOR-driven half lives in lidar_callback.)

        Split this way on purpose: HOLD / SETTLE / PARKED / ABORT are pure
        stopwatch states and must keep working even if /scan stops publishing.
        """
        if self.park_state == 'IDLE':
            return      # zero cost during the entire delivery run

        if self.park_state == 'HOLD':
            self.target_speed = 0.0
            self.target_turn = 0.0
            if self._park_time_in_state() >= PARK_START_DELAY:
                # Release the delivery-stop latch so LANE_FOLLOW can drive us
                # to the pad. Without this, edge_vectors_callback would keep
                # returning early on stopped_for_patient and we'd never move.
                self.stopped_for_patient = False
                self.qr_approach_active = False
                self.waiting_for_ack = False
                self._park_transition('SEARCH')

        elif self.park_state == 'SEARCH':
            # Lane following is driving. We only CAP the speed here, so bay
            # detection gets enough frames to see the comb before we blow past
            # the pad. Minimal interference: direction is still the camera's.
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
            # CRITICAL: zero is asserted EVERY tick, not once. An Ackermann
            # rover that receives a single zero-velocity message will coast and
            # drift out of the bay.
            self.target_speed = 0.0
            self.target_turn = 0.0

    # ---- LIDAR sector extraction --------------------------------------------

    def _park_beams_in_sector(self, scan, center_rad, half_span_rad):
        """
        Return [(bearing_rad, range_m), ...] for every beam in the sector,
        sorted by increasing bearing.

        Two deliberate choices:

        1. Bearings come from angle_min + i*angle_increment, NOT index
           fractions. Self-correcting if the scan isn't a full 360 deg or
           angle_min isn't -pi.

        2. Invalid beams (inf/nan/out of range) are KEPT with range = inf,
           not dropped. Dropping them would corrupt the angular ordering and
           merge two separate bays into one phantom bay. "No return" genuinely
           means "nothing there" = free space, which inf encodes correctly.
        """
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
        """Closest valid return in a sector, or inf if nothing is there."""
        beams = self._park_beams_in_sector(
            scan, math.radians(center_deg), math.radians(half_span_deg))
        finite = [r for (_, r) in beams if math.isfinite(r)]
        return min(finite) if finite else math.inf

    # ---- Bay detection: the core of the whole feature ------------------------

    def _park_find_bays(self, beams):
        """
        Given a bearing-sorted [(theta, range)] sector, return candidate bays.

        THE ALGORITHM
        -------------
        Classify each beam: STRUCTURE if r < PARK_CONE_RANGE_MAX, else FREE.
        Across the pad this gives a comb:

            cone   bay    cone   bay    cone   bay    cone
             |  #########  |  #########  |  #########  |
            [S][F F F F F][S][F F F F F][S][F F F F F][S]

        A bay is a run of FREE beams BOUNDED BY STRUCTURE ON BOTH SIDES.

        That bounding requirement is the single most important condition in
        this file: it is what separates a real bay from "I am looking past the
        end of the pad into open field", which is also a run of FREE beams but
        is bounded on ONE side only.

        Then four filters, each killing one specific failure mode:
          - both-sides-bounded -> open field
          - beam count         -> single-beam sensor noise
          - metric width band  -> too narrow to fit / absurdly wide (off pad)
          - depth gain         -> a seam between two same-range objects, which
                                  is a gap but not an opening
        """
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

            # Walk to the end of this free run.
            run_start = i
            while i < n and not is_structure[i]:
                i += 1
            run_end = i - 1

            # --- Filter 1: bounded by structure on BOTH sides ---
            left_tooth_idx = run_start - 1
            right_tooth_idx = run_end + 1
            if left_tooth_idx < 0 or right_tooth_idx >= n:
                continue        # open-ended: field, not a bay

            # --- Filter 2: minimum beam count (noise rejection) ---
            if (run_end - run_start + 1) < PARK_GAP_MIN_BEAMS:
                continue

            # --- Metric width, cone-to-cone, in Cartesian space ---
            # NOT angular: an angular threshold silently changes meaning with
            # distance and would pass narrow bays seen from far away.
            th_a, r_a = beams[left_tooth_idx]
            th_b, r_b = beams[right_tooth_idx]
            xa, ya = polar_to_xy(r_a, th_a)
            xb, yb = polar_to_xy(r_b, th_b)
            width = math.hypot(xb - xa, yb - ya)

            # --- Filter 3: width band ---
            if width < PARK_GAP_MIN_WIDTH or width > PARK_GAP_MAX_WIDTH:
                continue

            # --- Filter 4: the bay must be deeper than its bounding cones ---
            run_ranges = [r for (_, r) in beams[run_start:run_end + 1]]
            finite_run = [r for r in run_ranges if math.isfinite(r)]
            if finite_run:
                run_depth = sorted(finite_run)[len(finite_run) // 2]   # median
            else:
                run_depth = math.inf        # fully open behind = definitely deep
            tooth_depth = min(r_a, r_b)
            if run_depth < tooth_depth + PARK_BAY_MIN_DEPTH_GAIN:
                continue

            # --- Accepted. Bay entrance = midpoint of the two bounding cones ---
            cx = 0.5 * (xa + xb)
            cy = 0.5 * (ya + yb)
            bays.append({
                'x': cx,                        # forward distance to entrance
                'y': cy,                        # lateral offset to entrance
                'bearing': math.atan2(cy, cx),  # where to look for it
                'range': math.hypot(cx, cy),
                'width': width,
            })

        return bays

    def _park_select_bay(self, bays):
        """
        Choose which bay to aim for.

        FIRST valid bay in travel order, NOT the widest. Committing early beats
        driving the length of the pad hunting for a perfect slot - every metre
        spent searching is a metre nearer the end of the pad, past which there
        is no recovery.

        Once locked, prefer the bay nearest the previous lock bearing, so a
        neighbouring slot appearing mid-arc cannot steal the lock.
        """
        ahead = [b for b in bays if b['x'] > PARK_BAY_MIN_X]
        if not ahead:
            return None

        if self.park_locked_bay_bearing is None:
            return min(ahead, key=lambda b: b['x'])

        return min(bays, key=lambda b: abs(ang_diff(
            b['bearing'], self.park_locked_bay_bearing)))

    # ---- SEARCH --------------------------------------------------------------

    def _park_do_search(self, scan):
        """
        Watch the park side for a bay while LANE FOLLOWING drives.

        This method deliberately does NOT command speed or steering. The camera
        keeps the buggy on the track; parking only decides WHEN to take over.
        That keeps the takeover to a single, well-defined moment.
        """
        center = math.radians(PARK_SEARCH_SECTOR_CENTER_DEG) * self.park_side_sign
        beams = self._park_beams_in_sector(
            scan, center, math.radians(PARK_SEARCH_SECTOR_HALF_DEG))
        bay = self._park_select_bay(self._park_find_bays(beams))

        if bay is None:
            return          # nothing yet - keep lane following, keep looking

        self.park_locked_bay = bay
        self.park_locked_bay_bearing = bay['bearing']

        self.get_logger().info(
            f"Bay candidate: x={bay['x']:.2f} m, y={bay['y']:.2f} m, "
            f"w={bay['width']:.2f} m, brg={math.degrees(bay['bearing']):.0f} deg",
            throttle_duration_sec=0.5)

        # COMMIT TEST: start the arc once the bay entrance closes to the lead
        # distance. Turning any later and the rear inner wheel cuts the corner.
        if bay['x'] <= PARK_COMMIT_LEAD_X:
            self.get_logger().info(
                f"COMMITTING to bay: x={bay['x']:.2f} m "
                f"(lead={PARK_COMMIT_LEAD_X:.2f} m), width={bay['width']:.2f} m")
            # Clear the camera's residual steering so the arc starts clean.
            self.apex_active = False
            self._park_transition('ENTRY')

    # ---- ENTRY ---------------------------------------------------------------

    def _park_do_entry(self, scan):
        """
        Hold a full-lock arc toward the park side until the bay swings round
        into the front sector.

        NO ODOMETRY. The bay's own bearing IS the heading feedback: as the
        buggy rotates, the bay migrates from ~90 deg toward 0 deg. When it is
        within PARK_ENTRY_ALIGNED_DEG of straight ahead, the arc is done.

        The timeout is a real fallback, not decoration - the near cone often
        occludes the bay mid-arc. If that happens we assume the arc has carried
        us far enough and hand to CREEP, which re-acquires using symmetric side
        clearances instead of bay geometry.
        """
        # Widen the window: the bay is no longer abeam, it's sweeping forward.
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

        # Full lock toward the park side. park_side_sign does double duty here:
        # it aimed the sector above AND signs the steering, so flipping
        # PARK_SIDE mirrors the whole maneuver with one constant.
        self.rover_move_manual_mode(PARK_ENTRY_SPEED,
                                    PARK_TURN_FULL * self.park_side_sign)

    # ---- CREEP ---------------------------------------------------------------

    def _park_do_creep(self, scan):
        """
        Drive slowly into the bay, centering between the two bounding cones,
        until the far rail is close.

        CENTERING LAW:
            error = r_left - r_right
            turn  = PARK_CENTER_KP * error          (positive turn = steer left)

        Sign check: more room on the left => error > 0 => turn positive =>
        steer left => move toward the roomy side. Correct.

        The turn is clamped well below full lock - at creep speed a saturated
        steering command produces a lurch, not a correction.
        """
        front_r = self._park_min_range(scan, 0.0, PARK_FRONT_SECTOR_HALF_DEG)
        left_r = self._park_min_range(scan, 90.0, PARK_CENTER_SECTOR_HALF_DEG)
        right_r = self._park_min_range(scan, -90.0, PARK_CENTER_SECTOR_HALF_DEG)

        # --- Stop test: deep enough into the bay ---
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

        # --- Centering controller ---
        # Only run it when BOTH walls are visible. If one side reads inf we are
        # not between two cones yet (or one is occluded) and the error term is
        # meaningless - drive straight instead of chasing a phantom.
        if math.isfinite(left_r) and math.isfinite(right_r):
            error = left_r - right_r
            if abs(error) < PARK_CENTER_DEADBAND:
                turn = 0.0                      # inside deadband: hold straight
            else:
                turn = PARK_CENTER_KP * error
                turn = max(-PARK_CENTER_TURN_CLAMP,
                           min(PARK_CENTER_TURN_CLAMP, turn))
        else:
            turn = 0.0

        self.rover_move_manual_mode(PARK_CREEP_SPEED, turn)

    # ---- Final report --------------------------------------------------------

    def _park_report_final(self):
        """Log the final pose quality. Useful for tuning after each sim run."""
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

    # ---- Diagnostic ----------------------------------------------------------

    def park_dump_side_profile(self):
        """
        TUNING TOOL - run this BEFORE touching any other parking constant.

        Prints the raw comb on the park side:
            '#' = structure (cone/rail),  '.' = free space

        Park the buggy manually beside the pad, call this from a debug timer,
        and you should see a clean alternating pattern. If you don't, no amount
        of downstream tuning will help - PARK_CONE_RANGE_MAX is wrong and every
        filter after it is operating on garbage. Fix that number first.
        """
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

    # ===== END PARKING SECTION =====

    # =========================================================================
    # ===== COLLISION RECOVERY: ENTIRE NEW SECTION BELOW =====
    #
    #  Hooks into the rest of the file in only four places:
    #     - publish_drive_commands()  -> check_recovery_tick()   (last writer)
    #     - lidar_callback()          -> _recovery_update_detector()
    #     - edge_vectors_callback()   -> one guard line
    #     - __init__                  -> state variables
    #  Delete this section + those hooks and the file reverts exactly.
    #
    #  THE FSM
    #  -------
    #     IDLE    -- normal driving; detector watching in the background
    #       | (three-clause stuck test holds for STUCK_CONFIRM_FRAMES)
    #     REVERSE -- back up with counter-steer, duration scales with attempt #
    #       | (reverse duration elapsed)
    #     PAUSE   -- stop, look forward, decide
    #       |-- path clear      -> IDLE (hand control back, start cooldown)
    #       |-- still blocked   -> REVERSE again, escalated
    #       \-- attempts spent  -> IDLE, give up, drive on regardless
    #
    #  Getting stuck in a recovery loop is worse than the original collision,
    #  which is why every path out of PAUSE terminates.
    # =========================================================================

    # ---- Detection -----------------------------------------------------------

    def _recovery_front_range(self, scan):
        """
        Closest return in the forward cone. Reuses the parking sector helper -
        it is generic geometry, nothing parking-specific about it.
        """
        beams = self._park_beams_in_sector(
            scan, 0.0, math.radians(STUCK_FRONT_SECTOR_HALF_DEG))
        finite = [r for (_, r) in beams if math.isfinite(r)]
        return min(finite) if finite else math.inf

    def _recovery_clear_detector(self):
        """Reset the evidence buffer. Called whenever any gate fails."""
        self.recovery_stuck_frames = 0
        self.recovery_range_history = []

    def _recovery_update_detector(self, scan):
        """
        The three-clause stuck test, with all the false-positive gates in front
        of it. Runs every scan while recovery is IDLE.

        Read the gates as: "reasons this close-range reading is NORMAL and must
        not be treated as a collision."
        """
        if not RECOVERY_ENABLED:
            return

        # --- Reset escalation after a long stretch of clean driving ---
        # Without this, three unrelated collisions across a whole run would
        # exhaust the attempt budget and disable recovery for the rest of it.
        if (self.recovery_attempts > 0
                and (time.time() - self.recovery_last_end_time) > RECOVERY_ATTEMPT_RESET):
            self.recovery_attempts = 0
            self.recovery_given_up = False

        # --- GATE: already gave up on this obstacle ---
        if self.recovery_given_up:
            return

        # --- GATE: cooldown blackout right after a recovery ---
        # Stops the detector re-firing on the tail of the event it just handled.
        if (time.time() - self.recovery_last_end_time) < RECOVERY_COOLDOWN:
            self._recovery_clear_detector()
            return

        # --- GATE: we must actually be commanding forward motion ---
        # "Not moving" only means "stuck" if we asked to move. This single gate
        # eliminates the majority of would-be false positives.
        if self.target_speed < STUCK_MIN_CMD_SPEED:
            self._recovery_clear_detector()
            return

        # --- GATE: legitimately stopped at a delivery building ---
        if self.stopped_for_patient:
            self._recovery_clear_detector()
            return

        # --- GATE: parking states where being very close is the OBJECTIVE ---
        # CREEP deliberately crawls toward a rail until it is ~0.35 m away.
        # That is indistinguishable from a collision to this detector, so the
        # detector stands down rather than fighting the maneuver.
        if self.park_state in ('HOLD', 'CREEP', 'SETTLE', 'PARKED', 'ABORT'):
            self._recovery_clear_detector()
            return

        # --- CLAUSE 1 + 2: is something very close in front? ---
        front_r = self._recovery_front_range(scan)
        if (not math.isfinite(front_r)) or front_r > STUCK_FRONT_RANGE:
            self._recovery_clear_detector()
            return

        # --- CLAUSE 3: is that distance refusing to change? ---
        # THE KEY TEST. Approaching a wall => range shrinks every frame.
        # Range pinned flat while commanding forward => we are against it.
        self.recovery_range_history.append(front_r)
        if len(self.recovery_range_history) > STUCK_CONFIRM_FRAMES:
            self.recovery_range_history.pop(0)

        self.recovery_stuck_frames += 1

        if self.recovery_stuck_frames >= STUCK_CONFIRM_FRAMES:
            h = self.recovery_range_history
            if len(h) >= STUCK_CONFIRM_FRAMES and (max(h) - min(h)) < STUCK_RANGE_JITTER:
                self._recovery_start(front_r)
            else:
                # Range IS still changing - we are approaching, not wedged.
                # Keep watching, but don't let the frame counter run away.
                self.recovery_stuck_frames = STUCK_CONFIRM_FRAMES

    # ---- Maneuver ------------------------------------------------------------

    def _recovery_start(self, front_r):
        """Commit to a recovery. Chooses the counter-steer direction here."""
        self.recovery_attempts += 1
        self.recovery_park_state_at_trip = self.park_state

        # Counter-steer = opposite of whatever we were steering when we hit.
        # If we were going dead straight there is no "opposite", so alternate
        # by attempt number - that way a second attempt tries the other way out
        # instead of repeating a failed escape.
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
        """Single choke point for recovery state changes, so entries are stamped."""
        self.get_logger().info(
            f"RECOVERY FSM: {self.recovery_state} -> {new_state}")
        self.recovery_state = new_state
        self.recovery_state_entry_time = time.time()

    def _recovery_time_in_state(self):
        return time.time() - self.recovery_state_entry_time

    def check_recovery_tick(self):
        """
        The time-driven recovery maneuver, run from the 10 Hz timer and called
        LAST in publish_drive_commands so it overrides every other controller.
        """
        if self.recovery_state == 'IDLE':
            return          # zero cost during normal driving

        if self.recovery_state == 'REVERSE':
            # Escalating duration: each attempt backs up further than the last.
            duration = (RECOVERY_REVERSE_BASE_TIME
                        + (self.recovery_attempts - 1) * RECOVERY_REVERSE_TIME_STEP)

            self.target_speed = RECOVERY_REVERSE_SPEED
            self.target_turn = self.recovery_counter_turn

            if self._recovery_time_in_state() >= duration:
                self._recovery_transition('PAUSE')

        elif self.recovery_state == 'PAUSE':
            # Stop and look before committing to anything. Reversing straight
            # into a forward command produces a lurch and often re-collides.
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
                # Still boxed in. Escalate: longer reverse, flipped steer, so
                # the next attempt is genuinely different from the last.
                self.recovery_attempts += 1
                self.recovery_counter_turn = -self.recovery_counter_turn
                self.get_logger().warn(
                    f"Still blocked at {front_r:.2f} m. Escalating to attempt "
                    f"{self.recovery_attempts}/{RECOVERY_MAX_ATTEMPTS}, "
                    f"counter-steer={self.recovery_counter_turn:+.2f}")
                self._recovery_transition('REVERSE')

            else:
                # Budget spent. Hand control back anyway - an infinite reverse
                # loop burns the entire run, which is strictly worse than
                # letting the normal controllers have another go.
                self.get_logger().error(
                    f"Recovery exhausted after {RECOVERY_MAX_ATTEMPTS} attempts "
                    f"(front={front_r:.2f} m). Giving up, resuming normal drive.")
                self.recovery_given_up = True
                self._recovery_finish()

    def _recovery_finish(self):
        """
        Hand control back to the normal controllers and start the cooldown.

        Also repairs any state the collision left inconsistent - notably
        obstacle_in_front, which would otherwise stay latched True and keep the
        camera muted after we have already backed away.
        """
        self.recovery_last_end_time = time.time()
        self._recovery_clear_detector()

        # Clear stale avoidance latches from before the collision.
        self.obstacle_in_front = False
        self.frames_avoided = 0
        self.recovery_frames_remaining = 0

        # Clear stale camera latches - the apex memory would otherwise resume a
        # hard turn committed to before we hit anything.
        self.apex_active = False
        self.apex_blind_frames = 0

        # --- Parking resume policy ---
        # If we clipped a cone during the turn-in arc, we have now backed out
        # and the old bay lock describes a pose we are no longer in. Drop the
        # lock and go re-acquire a bay from scratch.
        if self.recovery_park_state_at_trip == 'ENTRY':
            self.get_logger().info(
                "Collision occurred during park ENTRY - dropping bay lock and "
                "returning to SEARCH to re-acquire.")
            self.park_locked_bay = None
            self.park_locked_bay_bearing = None
            self._park_transition('SEARCH')

        self.recovery_park_state_at_trip = None
        self._recovery_transition('IDLE')

    # ===== END COLLISION RECOVERY SECTION =====


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
