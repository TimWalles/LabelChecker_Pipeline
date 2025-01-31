from typing import List, NamedTuple
import pandas as pd


class BboxOverlap:
    @classmethod
    def bbox_overlap_check(
        cls,
        row1: pd.Series,
        row2: NamedTuple,
        iou_threshold: float,
    ) -> bool:
        """
        Check if two bounding boxes overlap.

        Args:
            row1 (pd.Series): first bounding box
            row2 (pd.Series): second bounding box
            iou_threshold (float): threshold for the Intersection over Union (IoU)

        Returns:
            bool: True if the bounding boxes overlap, False otherwise
        """
        # Calculate the Intersection over Union (IoU) of the two bounding boxes
        bbox1 = cls.get_coordinates(row1)
        bbox2 = cls.get_coordinates(row2)

        iou = cls.iou(bbox1=bbox1, bbox2=bbox2)

        # Check if the IoU is greater than the threshold
        return iou >= iou_threshold

    @staticmethod
    def get_coordinates(row: pd.Series) -> List[int]:
        return [row.SrcX, row.SrcY, row.SrcX + row.ImageW, row.SrcY + row.ImageH]

    @staticmethod
    def iou(
        bbox1: List,
        bbox2: List,
    ):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Args:
            bbox1 (list): bounding box in format [x1, y1, x2, y2]
            bbox2 (list): bounding box in format [x1, y1, x2, y2]

        Returns:
            float: value of the IoU
        """
        # Get the coordinates of the intersection rectangle
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        # Calculate area of intersection rectangle
        intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        # Calculate area of both the bounding boxes
        bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
        bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

        # Calculate the Intersection over Union (IoU)
        iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
        return iou
