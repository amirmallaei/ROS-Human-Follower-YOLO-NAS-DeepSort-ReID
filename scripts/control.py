# scripts/control.py

import rospy
from geometry_msgs.msg import Twist
import numpy as np

class Controller:
    """
    Controls the robot's movement smoothly using a proportional controller.
    """

    def __init__(self, frame_width):
        """
        Initializes the Controller.
        Args:
            frame_width (int): The width of the camera frame.
        """
        # ROS Publisher
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=4)
        self.cmd_vel = Twist()

        # --- TUNING PARAMETERS FOR SMOOTH CONTROL ---

        # Proportional Gains: Higher values mean a more aggressive response.
        self.P_ANGULAR_GAIN = 0.004  # Controls turning speed based on how far off-center the target is.
        self.P_LINEAR_GAIN = 0.008   # Controls forward/backward speed based on distance from target.

        # Target Position: Where we want the person to be in the frame.
        self.frame_width = frame_width
        self.TARGET_Y2 = 354         # The target bottom-coordinate for the bounding box (ideal distance).

        # Dead Zones: Thresholds to prevent the robot from "wiggling" when it's close enough.
        self.ANGULAR_THRESHOLD = 15  # Pixels from center before the robot starts turning.
        self.Y2_THRESHOLD = 8        # Pixels from the target distance before the robot starts moving.

        # Speed Limits: Maximum safe speeds for your robot.
        self.MAX_LINEAR_SPEED = 0.5  # meters per second
        self.MAX_ANGULAR_SPEED = 0.8 # radians per second

    def move_robot(self, center_x, y2):
        """
        Calculates and publishes smooth velocity commands based on the object's position.
        Args:
            center_x (int): X-coordinate of the tracked object's center.
            y2 (int): Bottom Y-coordinate of the tracked object's bounding box.
        """
        # --- Angular (Turning) Control ---
        angular_error = (self.frame_width / 2) - center_x

        if abs(angular_error) > self.ANGULAR_THRESHOLD:
            # Error is significant, so we calculate a proportional speed.
            angular_velocity = self.P_ANGULAR_GAIN * angular_error
        else:
            # Error is negligible, so we don't turn.
            angular_velocity = 0.0

        # --- Linear (Distance) Control ---
        linear_error = self.TARGET_Y2 - y2

        if abs(linear_error) > self.Y2_THRESHOLD:
            # Error is significant, calculate proportional speed.
            linear_velocity = self.P_LINEAR_GAIN * linear_error
        else:
            # Error is negligible, don't move forward/backward.
            linear_velocity = 0.0

        # --- Apply Speed Limits (Safety Clamp) ---
        self.cmd_vel.linear.x = np.clip(linear_velocity, -self.MAX_LINEAR_SPEED, self.MAX_LINEAR_SPEED)
        self.cmd_vel.angular.z = np.clip(angular_velocity, -self.MAX_ANGULAR_SPEED, self.MAX_ANGULAR_SPEED)

        # Publish the command
        self.velocity_publisher.publish(self.cmd_vel)
        rospy.loginfo(f"Moving: Linear X={self.cmd_vel.linear.x:.2f}, Angular Z={self.cmd_vel.angular.z:.2f}")

    def stop_robot(self):
        """Stops the robot's movement completely."""
        self.cmd_vel.linear.x = 0.0
        self.cmd_vel.angular.z = 0.0
        self.velocity_publisher.publish(self.cmd_vel)
        rospy.loginfo("Robot stopped.")