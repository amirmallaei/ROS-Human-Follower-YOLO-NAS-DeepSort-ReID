# scripts/tracking.py

from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    """Manages multi-object tracking using DeepSort."""

    def __init__(self, max_age=15):
        """
        Initializes the DeepSort tracker.

        Args:
            max_age (int): The maximum number of consecutive misses before a track is deleted.
        """
        self.tracker = DeepSort(max_age=max_age)

    def update_tracks(self, detections, frame):
        """
        Updates the tracker with new detections.

        Args:
            detections (list): The list of detections from the Detector.
            frame (numpy.ndarray): The current video frame.

        Returns:
            list: A list of active tracks from DeepSort.
        """
        return self.tracker.update_tracks(detections, frame=frame)