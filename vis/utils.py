import supervision as sv
import numpy as np
from typing import List
from PIL import Image


def xywh_to_xyxy(boxes: List):
    converted_boxes = []
    for box in boxes:
        x, y, w, h = box
        x_min = x - w / 2
        y_min = y - h / 2
        x_max = x + w / 2
        y_max = y + h / 2
        converted_boxes.append([x_min, y_min, x_max, y_max])
    return converted_boxes


def annotate(
    image: Image,
    boxes: List,
    scores: List,
    labels: List,
):
    """
    return annotated image with bounding boxes and labels
    boxes: normalized, xywh format
    boxes and scores also support tensor format
    """

    w, h = image.size
    boxes = xywh_to_xyxy(boxes)
    boxes = np.array(boxes) * np.array([w, h, w, h]).reshape(1, 4)
    detections = sv.Detections(boxes)
    if scores:
        labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(labels, scores)]
    else:
        labels = [f"{phrase}" for phrase in labels]
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(
        scene=image, detections=detections, labels=labels
    )
    return annotated_frame


def load_image(image_path, size=None, keep_aspect_ratio=True):
    image = Image.open(image_path).convert("RGB")
    if type(size) == int:
        size = (size, size)
    if size:
        if keep_aspect_ratio:
            image.thumbnail(size)
        else:
            image = image.resize(size)
    return image
