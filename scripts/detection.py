# scripts/detection.py

import torch
from super_gradients.training import models
import rospy

class Detector:
    """Handles object detection using the Yolo-NAS model."""

    def __init__(self, model_name="yolo_nas_l", confidence_threshold=0.5):
        """
        Initializes the Detector.

        Args:
            model_name (str): The name of the Yolo-NAS model to use.
            confidence_threshold (float): The minimum confidence for a detection.
        """
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"Detector using device: {self.device}")
        self.model = models.get(model_name, pretrained_weights="coco").to(self.device)

    def detect(self, frame):
        """
        Performs object detection on a given frame.

        Args:
            frame (numpy.ndarray): The input image frame.

        Returns:
            list: A list of detections in a format suitable for DeepSort.
        """
        detections_for_deepsort = []
        if frame is None:
            return detections_for_deepsort

        detect_result = next(iter(self.model.predict(frame, conf=self.confidence_threshold)))
        
        bboxes_xyxy = detect_result.prediction.bboxes_xyxy.tolist()
        confidences = detect_result.prediction.confidence.tolist()
        labels = detect_result.prediction.labels.tolist()

        for bbox, conf, label in zip(bboxes_xyxy, confidences, labels):
            if conf >= self.confidence_threshold:
                xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                width, height = xmax - xmin, ymax - ymin
                detections_for_deepsort.append([[xmin, ymin, width, height], conf, int(label)])
        
        return detections_for_deepsort