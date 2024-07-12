import torch
import matplotlib.pyplot as plt
from PIL import Image
import supervision as sv
import numpy as np
from typing import List


def draw_bbox(image, boxes):
    """Draw bounding boxes on the image"""
    plt.imshow(image)
    ax = plt.gca()
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                color="red",
                linewidth=3,
            )
        )
    plt.axis("off")
    plt.show()


def draw_bbox_with_score(image, boxes, labels, scores, threshhold=0.1):
    """Draw bounding boxes with labels and scores on the image"""
    plt.imshow(image)
    ax = plt.gca()
    for box, label, score in zip(boxes, labels, scores):
        if score < threshhold:
            continue
        xmin, ymin, xmax, ymax = box
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                color="red",
                linewidth=3,
            )
        )
        text = f"{label} {round(score.item(), 3)}"
        ax.text(xmin, ymin, text, fontsize=12, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")
    plt.show()


def undo_transform(tensor):
    """
    Convert a tensor image back to a PIL image
    i.e. undo T.Normalize and T.ToTensor
    """
    # Denormalize the tensor
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)

    # Convert the tensor to a PIL image
    tensor = tensor.mul(255).clamp(0, 255).byte()
    np_image = tensor.permute(1, 2, 0).cpu().numpy()
    # pil_image = Image.fromarray(np_array)

    return np_image


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


def xyxy_to_xywh(boxes: List):
    converted_boxes = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        x = (x_min + x_max) / 2
        y = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min
        converted_boxes.append([x, y, w, h])
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
    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(labels, scores)]
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
