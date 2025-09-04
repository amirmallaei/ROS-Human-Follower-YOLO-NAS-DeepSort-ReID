#!/usr/bin/env python3
# scripts/main_node.py

import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import os

# Import our custom modules
from detection import Detector
from tracking import Tracker
from control import Controller
from utils import load_class_names, generate_class_colors, compare_histograms

class HumanFollowerNode:
    """
    Main ROS node for the human following robot.
    Orchestrates detection, tracking, and control.
    """

    def __init__(self):
        """Initializes the main node."""
        rospy.init_node('human_follower_node', anonymous=True)

        # Configuration
        self.bhattacharyya_threshold = 0.35
        
        # ROS Communication
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/center_cam", CompressedImage, self._image_callback)

        # Frame and state variables
        self.frame = None
        self.frame_height = 480  # Default, will be updated
        self.frame_width = 640   # Default, will be updated

        # Tracking State
        self.following_id = None
        self.is_following = False
        self.is_lost = False
        self.has_captured_subject_roi = False
        self.motor_enabled = False
        self.subject_roi = None

        # Resolve package path to find config file
        package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(package_path, 'configs', 'coco.names')

        # Helper classes
        self.class_names = load_class_names(config_path)
        self.colors = generate_class_colors(len(self.class_names))

        # Core Components
        self.detector = Detector()
        self.tracker = Tracker()
        self.controller = Controller(self.frame_width)

    def _image_callback(self, img_msg):
        """Callback to receive and process camera frames."""
        self.frame = self.bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")
        if self.frame is not None and (self.frame_height != self.frame.shape[0] or self.frame_width != self.frame.shape[1]):
            self.frame_height, self.frame_width, _ = self.frame.shape
            self.controller.frame_width = self.frame_width

    def run(self):
        """Main loop of the node."""
        rospy.loginfo("Human Follower Node started.")
        while not rospy.is_shutdown():
            if self.frame is None:
                rospy.logwarn("Waiting for image frame...")
                rospy.sleep(0.1)
                continue

            detections = self.detector.detect(self.frame)
            tracks = self.tracker.update_tracks(detections, self.frame)
            self._manage_tracking_state(tracks)
            self._display_frame()

        cv2.destroyAllWindows()
        rospy.loginfo("Human Follower Node shut down.")

    def _manage_tracking_state(self, tracks):
        """Manages the logic for following, losing, and re-identifying a subject."""
        current_track_ids = [track.track_id for track in tracks if track.is_confirmed()]

        # Handle loss of all subjects
        if not current_track_ids and self.motor_enabled:
            rospy.loginfo("All subjects lost. Stopping motor.")
            self.is_lost = True
            self.motor_enabled = False
            self.controller.stop_robot()
        
        # Handle loss of the specific followed subject
        if self.is_following and self.following_id not in current_track_ids and not self.is_lost:
            rospy.loginfo(f"Followed subject {self.following_id} lost. Stopping.")
            self.is_lost = True
            self.motor_enabled = False
            self.controller.stop_robot()

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)

            self._draw_bounding_box(x1, y1, x2, y2, class_id, track_id)

            if class_id == 0:  # 'person' class in COCO
                center_x = int((x1 + x2) / 2)
                
                # Initial subject acquisition
                if not self.is_following:
                    self._acquire_new_subject(track_id, x1, y1, x2, y2)

                # Control robot if following this subject
                if self.following_id == track_id and self.motor_enabled:
                    self.controller.move_robot(center_x, y2)

                # Re-identification logic
                if self.is_lost and self.has_captured_subject_roi:
                    self._attempt_reidentification(track_id, x1, y1, x2, y2)

    def _acquire_new_subject(self, track_id, x1, y1, x2, y2):
        """Begins following a new human subject."""
        rospy.loginfo(f"Acquiring new subject to follow: ID {track_id}")
        self.is_following = True
        self.following_id = track_id
        self.motor_enabled = True
        self.is_lost = False
        
        # Capture upper body ROI for re-identification
        if y1 > 30 and y2 > y1 and x2 > x1:
            self.subject_roi = self.frame[y1:min(y2, int(y1 + (y2 - y1) * 0.5)), x1:x2]
            self.has_captured_subject_roi = True

    def _attempt_reidentification(self, track_id, x1, y1, x2, y2):
        """Compares a found human with the lost subject's ROI."""
        if y2 > y1 and x2 > x1:
            current_human_roi = self.frame[y1:min(y2, int(y1 + (y2 - y1) * 0.5)), x1:x2]
            distance = compare_histograms(self.subject_roi, current_human_roi)
            
            if distance < self.bhattacharyya_threshold:
                rospy.loginfo(f"Re-identified subject! New ID: {track_id}. Resuming follow.")
                self.following_id = track_id
                self.is_following = True
                self.is_lost = False
                self.motor_enabled = True
                self.subject_roi = current_human_roi # Update ROI

    def _draw_bounding_box(self, x1, y1, x2, y2, class_id, track_id):
        """Draws bounding boxes and labels on the frame."""
        color = self.colors[class_id].tolist()
        text = f"ID {track_id} - {self.class_names[class_id]}"
        cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(self.frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    def _display_frame(self):
        """Displays the current video frame."""
        cv2.imshow('Human Following Robot', self.frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    try:
        node = HumanFollowerNode()
        node.run()
    except rospy.ROSInterruptException:
        pass