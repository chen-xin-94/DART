import json
from collections import defaultdict

import torch
import matplotlib.pyplot as plt
from PIL import Image
import supervision as sv
import numpy as np
from PIL import Image
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
    phrases: List,
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
    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, scores)]
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


## following functions are used for processing OVD results of original images


def get_labels_all(ann_dir, objs, combo, prompt_types, id_types=["0", "1"]):
    """
    collect all labels from all prompt types.

    id_type should be represented in the first character of the file name:
        int: original images, currently either '0' or '1'
        'b': background changed images
        'd': dreambooth generated images

    the returned labels dict is a dict of dict of dict:
        first-level key is prompt type,
        second-level  key is object id,
        third-level  key is label dict with keys: boxes, logits, phrases
    """

    labels = dict.fromkeys(prompt_types)
    labels["id_types"] = id_types
    counts = defaultdict(int)

    for prompt_type in prompt_types:
        _labels = {}
        ann_label_dir = ann_dir / prompt_type / "labels"
        for obj in objs:
            ann_label_dir_obj = ann_label_dir / obj
            # traverse json files in the directory
            for ann_file in ann_label_dir_obj.rglob("*.json"):
                id_type = ann_file.stem[0]
                if id_type not in id_types:
                    continue
                with open(ann_file, "r") as f:
                    ann = json.load(f)
                _labels[ann_file.stem] = {}
                # copy everything from the original json file
                _labels[ann_file.stem].update(ann)
                # create a new key "classes" for nms
                if obj not in combo:
                    # all classes should be obj
                    _labels[ann_file.stem]["classes"] = [
                        obj for _ in range(len(ann["boxes"]))
                    ]
                else:
                    # should consider the combo classes
                    combo_phrase = combo[obj]
                    _classes = []
                    for phrase in ann["phrases"]:
                        if phrase == combo_phrase:
                            _classes.append(
                                combo_phrase
                            )  # the combo_phrase should always be a obj name, so no further processing is needed
                        else:
                            _classes.append(obj)
                    _labels[ann_file.stem]["classes"] = _classes
                counts[prompt_type] += len(ann["boxes"])
        labels[prompt_type] = _labels

    return labels, counts


def get_labels_pre_rule(ann_dir, objs, combo, prompt_types, id_types=["0", "1"]):
    """
    collect all labels from all prompt types with a stricter rule.
    # The rule:
    # 1. if the image only gets one annotation, then keep it unless the score is too low (<0.2). In all our experiment, the unless part is actuall redundant, since we always specify a box_threshold with value bigger than 0.2, specifically {0.25, 0.3, 0.35, 0.4}.
    # 2. if the image has multiple annotations,
    #    1. If none of them is bigger than 0.5, simply keep the highest one.
    #    2. If there are ones with score >=0.5, keep them

    id_type should be represented in the first character of the file name:
        int: original images, currently either '0' or '1'
        'b': background changed images
        'd': dreambooth generated images

    the returned labels dict is a dict of dict of dict:
        first-level key is prompt type,
        second-level  key is object id,
        third-level  key is label dict with keys: boxes, logits, phrases
    """

    labels = dict.fromkeys(prompt_types)
    labels["id_types"] = id_types
    counts = defaultdict(int)
    for prompt_type in prompt_types:
        _labels = {}
        ann_label_dir = ann_dir / prompt_type / "labels"
        for obj in objs:
            ann_label_dir_obj = ann_label_dir / obj
            # traverse json files in the directory
            for ann_file in ann_label_dir_obj.rglob("*.json"):
                id_type = ann_file.stem[0]
                if id_type not in id_types:
                    continue
                with open(ann_file, "r") as f:
                    ann = json.load(f)
                _labels[ann_file.stem] = {
                    "boxes": [],
                    "logits": [],
                    "phrases": [],
                    "classes": [],
                }
                logits = ann["logits"]

                if len(logits) == 0:
                    continue
                elif len(logits) == 1 and logits[0] > 0.2:  # rule 1
                    _labels[ann_file.stem].update(ann)
                    counts[prompt_type] += 1
                else:  # rule 2
                    temp = defaultdict(list)
                    if max(logits) < 0.5:  # rule 2.1
                        i = logits.index(max(logits))
                        for key in ann:
                            temp[key].append(ann[key][i])
                        counts[prompt_type] += 1
                    else:  # rule 2.2
                        for i, logit in enumerate(logits):
                            if logit >= 0.5:
                                for key in ann:
                                    temp[key].append(ann[key][i])
                                counts[prompt_type] += 1
                    _labels[ann_file.stem].update(temp)

                # create a new key "classes" for nms
                if obj not in combo:
                    # all classes should be obj
                    _labels[ann_file.stem]["classes"] = [
                        obj for _ in range(len(_labels[ann_file.stem]["boxes"]))
                    ]
                else:
                    # should consider the combo classes
                    combo_phrase = combo[obj]
                    for phrase in _labels[ann_file.stem]["phrases"]:
                        if phrase == combo_phrase:
                            _labels[ann_file.stem]["classes"].append(
                                combo_phrase
                            )  # the combo_phrase should always be a obj name, so no further processing is needed
                        else:
                            _labels[ann_file.stem]["classes"].append(obj)

        labels[prompt_type] = _labels
    return labels, counts


def get_labels_post_rule(labels_all, id_to_name, combo, id_types=["0", "1"]):
    """
    collect labels from already merged annotations, i.e. labels_all['one_and_syn'], with a stricter rule.
    # The rule:
    # 1. if the image only gets one annotation, then keep it unless the score is too low (<0.2). In all our experiment, the unless part is actuall redundant, since we always specify a box_threshold with value bigger than 0.2, specifically {0.25, 0.3, 0.35, 0.4}.
    # 2. if the image has multiple annotations,
    #    1. If none of them is bigger than 0.5, simply keep the highest one.
    #    2. If there are ones with score >=0.5, keep them

    id_type should be represented in the first character of the file name:
        int: original images, currently either '0' or '1'
        'b': background changed images
        'd': dreambooth generated images

    the returned labels dict is a dict of dict of dict:
        first-level key is prompt type,
        second-level  key is object id,
        third-level  key is label dict with keys: boxes, logits, phrases
    """
    _labels = {}

    count = 0
    for id, ann in labels_all["one_and_syn"].items():
        obj = id_to_name[id + ".jpg"].split("/")[0]
        logits = ann["logits"]
        boxes = ann["boxes"]
        phrases = ann["phrases"]
        _labels[id] = {
            "boxes": [],
            "logits": [],
            "phrases": [],
            "classes": [],
        }
        if len(logits) == 0:
            continue
        elif len(logits) == 1 and logits[0] > 0.2:  # rule 1
            _labels[id] = ann
        else:  # rule 2
            if max(logits) < 0.5:  # rule 2.1
                i = logits.index(max(logits))
                for key in ann:
                    _labels[id][key] = [ann[key][i]]
            else:  # rule 2.2
                for i, logit in enumerate(logits):
                    if logit >= 0.5:
                        for key in ann:
                            _labels[id][key].append(ann[key][i])

        # create a new key "classes" for nms
        if obj not in combo:
            # all classes should be obj
            _labels[id]["classes"] = [obj for _ in range(len(_labels[id]["boxes"]))]
        else:
            # should consider the combo classes
            combo_phrase = combo[obj]

            for phrase in _labels[id]["phrases"]:
                if phrase == combo_phrase:
                    _labels[id]["classes"].append(
                        combo_phrase
                    )  # the combo_phrase should always be a obj name, so no further processing is needed
                else:
                    _labels[id]["classes"].append(obj)
    labels = {}
    labels["one_and_syn"] = _labels
    labels["id_types"] = list(set(id_types) & set(labels_all["id_types"]))

    return labels


def compute_stats_summary_pt(labels, prompt_types):
    stats_summary = dict.fromkeys(prompt_types)
    for prompt_type in prompt_types:
        stats = {"avg_score": 0.0, "num_ann": 0}
        for _, ann in labels[prompt_type].items():
            stats["avg_score"] += sum(ann["logits"])
            stats["num_ann"] += len(ann["logits"])
        if stats["num_ann"] > 0:
            stats["avg_score"] /= stats["num_ann"]
        else:
            stats["avg_score"] = 0.0
        stats["num_img"] = len(labels[prompt_type])
        stats_summary[prompt_type] = stats
    return stats_summary


def compute_obj_wise_stats_pt(labels, objs, id_to_name, prompt_types):
    stats_obj = {
        prompt_type: {
            obj: {"avg_score": 0.0, "num_ann": 0, "num_img": 0} for obj in objs
        }
        for prompt_type in prompt_types
    }
    for prompt_type in prompt_types:
        for id, ann in labels[prompt_type].items():
            obj = id_to_name[id + ".jpg"].split("/")[0]
            stats_obj[prompt_type][obj]["avg_score"] += sum(ann["logits"])
            stats_obj[prompt_type][obj]["num_ann"] += len(ann["logits"])
            stats_obj[prompt_type][obj]["num_img"] += 1

        for obj in objs:
            if stats_obj[prompt_type][obj]["num_ann"] != 0:
                stats_obj[prompt_type][obj]["avg_score"] /= stats_obj[prompt_type][obj][
                    "num_ann"
                ]

    return stats_obj


## following functions are used for processing OVD results of generated images
def get_labels_rule(objs, image_dir, ann_dir):
    """
    # The rule:
    # 1. if the image only gets one annotation, then keep it unless the score is too low (<0.2). In all our experiment, the unless part is actuall redundant, since we always specify a box_threshold with value bigger than 0.2, specifically {0.25, 0.3, 0.35, 0.4}.
    # 2. if the image has multiple annotations,
    #    1. If none of them is bigger than 0.5, simply keep the highest one.
    #    2. If there are ones with score >=0.5, keep them
    """
    labels = {}
    for obj in objs:
        obj_image_dir = image_dir / obj
        for image_path in obj_image_dir.rglob("*.jpg"):
            ann_path = ann_dir / obj / (image_path.stem + ".json")
            with open(ann_path, "r") as f:
                ann = json.load(f)
            id = image_path.stem
            labels[id] = {"boxes": [], "logits": [], "phrases": [], "classes": []}
            logits = ann["logits"]
            # Implement the rules
            if len(logits) == 0:
                continue
            elif len(logits) == 1 and logits[0] > 0.2:  # rule 1
                labels[id].update(ann)
            else:
                if max(logits) < 0.5:  # rule 2.1
                    i = logits.index(max(logits))
                    for key in ann:
                        labels[id][key].append(ann[key][i])
                else:  # rule 2.2
                    for i, logit in enumerate(logits):
                        if logit >= 0.5:
                            for key in ann:
                                labels[id][key].append(ann[key][i])
            # Add classes
            labels[id]["classes"] = [obj] * len(labels[id]["boxes"])
    return labels


def compute_stats_summary(labels):
    stats_summary = {"avg_score": 0.0, "num_ann": 0}
    for _, ann in labels.items():
        stats_summary["avg_score"] += sum(ann["logits"])
        stats_summary["num_ann"] += len(ann["logits"])
    if stats_summary["num_ann"] > 0:
        stats_summary["avg_score"] /= stats_summary["num_ann"]
    else:
        stats_summary["avg_score"] = 0.0
    stats_summary["num_img"] = len(labels)
    return stats_summary


def compute_obj_wise_stats(labels, objs):
    stats_obj = {obj: {"avg_score": 0.0, "num_ann": 0, "num_img": 0} for obj in objs}
    for id, ann in labels.items():
        if ann["boxes"] == []:
            continue
        obj = ann["classes"][0]  # since currently only one class
        stats_obj[obj]["avg_score"] += sum(ann["logits"])
        stats_obj[obj]["num_ann"] += len(ann["logits"])
        stats_obj[obj]["num_img"] += 1
    for obj in objs:
        if stats_obj[obj]["num_ann"] != 0:
            stats_obj[obj]["avg_score"] /= stats_obj[obj]["num_ann"]
    return stats_obj
