#!/usr/bin/env python3
# Copyright 2026 NXP / Autonomous Buggy Test Suite

import rclpy
from rclpy.node import Node
import random
from synapse_msgs.msg import ServerCommunication

class TestServerNode(Node):
    def __init__(self):
        super().__init__('test_server_node')

        # Subscribe to messages sent FROM the buggy (dest == 2)
        self.subscription = self.create_subscription(
            ServerCommunication,
            '/ServerCommunication',
            self.server_communication_callback,
            10
        )

        # Publisher to send commands/ACKs TO the buggy (dest == 1)
        self.publisher = self.create_publisher(
            ServerCommunication,
            '/ServerCommunication',
            10
        )

        # Mission tracking pools (buildings removed after assignment to guarantee uniqueness)
        self.available_patients = ['A', 'B', 'C']
        self.available_hospitals = ['X', 'Y', 'Z']
        
        self.deliveries_completed = 0
        self.max_deliveries = 3
        
        # Track last processed UID to prevent duplicate state triggers from retries
        self.last_processed_uid = None

        self.get_logger().info("Test Server Node started. Waiting for buggy messages...")

    def server_communication_callback(self, msg):
        # Only process messages intended for Server (dest == 2)
        if msg.dest != 2:
            return

        # Ignore incoming ACKs sent by the buggy back to the server
        if msg.ack == 1:
            self.get_logger().info(f"[SERVER] Received ACK from Buggy for UID={msg.uid}")
            return

        arrival_code = msg.msg.strip().upper()
        self.get_logger().info(
            f"[SERVER] Buggy arrived/reported: '{arrival_code}' (UID={msg.uid})"
        )

        # 1. Immediately send ACK back to buggy for the received message
        self.send_ack(msg.uid)

        # Prevent duplicate processing if buggy retried sending the same UID
        if msg.uid == self.last_processed_uid:
            self.get_logger().warn(f"[SERVER] Duplicate UID={msg.uid} received. ACK resent, skipping state logic.")
            return

        self.last_processed_uid = msg.uid

        # 2. Decide next command based on current state
        if arrival_code in ['A', 'B', 'C']:
            # Remove current patient from available pool if still present
            if arrival_code in self.available_patients:
                self.available_patients.remove(arrival_code)

            if self.available_hospitals:
                # Assign and pop a unique hospital
                next_hospital = self.available_hospitals.pop(random.randrange(len(self.available_hospitals)))
                self.get_logger().info(
                    f"[SERVER] Patient '{arrival_code}' picked up. "
                    f"Ordering buggy to unique Hospital: '{next_hospital}' (Remaining Hospitals: {self.available_hospitals})"
                )
                self.send_command(next_hospital)

        elif arrival_code in ['X', 'Y', 'Z']:
            self.deliveries_completed += 1
            self.get_logger().info(f"[SERVER] Delivery {self.deliveries_completed}/{self.max_deliveries} completed at Hospital '{arrival_code}'.")

            if self.deliveries_completed >= self.max_deliveries or not self.available_patients:
                # All unique deliveries done -> Send 'OK'
                self.get_logger().info("[SERVER] All unique deliveries complete! Sending 'OK' to finish mission.")
                self.send_command("OK")
            else:
                # Assign and pop a unique patient
                next_patient = self.available_patients.pop(random.randrange(len(self.available_patients)))
                self.get_logger().info(
                    f"[SERVER] Ordering buggy to unique Patient: '{next_patient}' (Remaining Patients: {self.available_patients})"
                )
                self.send_command(next_patient)
        else:
            self.get_logger().warn(f"[SERVER] Unknown arrival code '{arrival_code}' received.")

    def send_ack(self, uid_to_ack):
        ack_msg = ServerCommunication()
        ack_msg.src = 2   # Server ID
        ack_msg.dest = 1  # Buggy ID
        ack_msg.uid = uid_to_ack
        ack_msg.ack = 1
        ack_msg.msg = ""
        self.publisher.publish(ack_msg)
        self.get_logger().info(f"[SERVER -> BUGGY] Sent ACK for UID={uid_to_ack}")

    def send_command(self, command_str):
        cmd_msg = ServerCommunication()
        cmd_msg.src = 2   # Server ID
        cmd_msg.dest = 1  # Buggy ID
        cmd_msg.uid = random.randint(1, 255)  # Assign server UID
        cmd_msg.ack = 0
        cmd_msg.msg = command_str
        self.publisher.publish(cmd_msg)
        self.get_logger().info(f"[SERVER -> BUGGY] Sent Command: '{command_str}' (UID={cmd_msg.uid})")


def main(args=None):
    rclpy.init(args=args)
    node = TestServerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()