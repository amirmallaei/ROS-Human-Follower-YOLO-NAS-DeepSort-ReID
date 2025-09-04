# scripts/utils.py

import cv2
import numpy as np
import rospy

def load_class_names(file_path):
    """
    Loads class names from a given file.

    Args:
        file_path (str): The path to the file containing class names, one per line.

    Returns:
        list: A list of class names.
    """
    try:
        with open(file_path, "r") as f:
            return f.read().strip().split("\n")
    except FileNotFoundError:
        rospy.logerr(f"Class names file not found at: {file_path}")
        rospy.signal_shutdown("Missing class names file.")
        return []

def generate_class_colors(num_classes):
    """
    Generates a list of random colors for visualizing object classes.

    Args:
        num_classes (int): The number of classes to generate colors for.

    Returns:
        numpy.ndarray: An array of RGB color tuples.
    """
    np.random.seed(42)  # For reproducible colors
    return np.random.randint(0, 255, size=(num_classes, 3))

def compare_histograms(roi1, roi2):
    """
    Compares two image ROIs using Bhattacharyya distance on HSV histograms.

    Args:
        roi1 (numpy.ndarray): The first Region of Interest (BGR image).
        roi2 (numpy.ndarray): The second Region of Interest (BGR image).

    Returns:
        float: The Bhattacharyya distance. Returns 1.0 if ROIs are invalid.
    """
    if roi1 is None or roi2 is None or roi1.size == 0 or roi2.size == 0:
        rospy.logwarn("Invalid ROI provided for histogram comparison.")
        return 1.0

    try:
        roi1_hsv = cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)
        roi2_hsv = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)

        hist1 = cv2.calcHist([roi1_hsv], [0, 1], None, [140, 256], [0, 180, 0, 256])
        hist2 = cv2.calcHist([roi2_hsv], [0, 1], None, [140, 256], [0, 180, 0, 256])

        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    except cv2.error as e:
        rospy.logerr(f"Error during histogram comparison: {e}")
        return 1.0