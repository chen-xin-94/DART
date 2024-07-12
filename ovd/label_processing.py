import os

# to turn off SupervisionWarnings, note that the following code must be run before importing anything from supervision
# https://github.com/roboflow/supervision/pull/962/commits/7d65a9445aa04d82334c6fe2828278e528eda1cb
os.environ["SUPERVISON_DEPRECATION_WARNING"] = "0"

from pathlib import Path
import json

import copy
import random
import torch
from torchvision.ops import nms, box_convert
from utils import (
    get_labels_all,
    compute_stats_summary_pt,
    compute_obj_wise_stats_pt,
    get_labels_pre_rule,
    get_labels_post_rule,
    load_image,
    annotate,
)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ann_dir",
    type=str,
    default="annotations",
    help="Path to the annotations directory",
)
parser.add_argument(
    "--label_dir", type=str, default="labels", help="Path to the label directory"
)
parser.add_argument(
    "--id_types", nargs="+", default=["0", "1"], help="Types of ids to use"
)
parser.add_argument(
    "--save_img", action="store_true", help="Whether to save images after nms"
)
parser.add_argument(
    "--pre_rule", action="store_true", help="Whether to use pre-rule to filter labels"
)
args = parser.parse_args()


dataset_dir = Path("/mnt/ssd2/xin/repo/DART/Liebherr_Product")
repo_dir = Path("/mnt/ssd2/xin/repo/GroundingDINO")

# Define the images directory and duplicates directory using Path objects
image_dir = dataset_dir / "images"
ann_dir = dataset_dir / args.ann_dir
metadata_dir = dataset_dir / "metadata"
label_dir = dataset_dir / args.label_dir
label_dir.mkdir(exist_ok=True)


# List all objects in the image directory
objs = sorted([obj.name for obj in image_dir.iterdir()])

# load classes and synonyms
with open(metadata_dir / "id_to_name.json", "r") as f:
    id_to_name = json.load(f)
with open(metadata_dir / "combo.json", "r") as f:
    combo = json.load(f)

# ## Extract all labels

prompt_types = ["one", "synonym_sep", "all"]

labels_all, counts_all = get_labels_all(
    ann_dir, objs, combo, prompt_types, args.id_types
)

# calculate average logits for each prompt type
stats_summary_all = compute_stats_summary_pt(labels_all, prompt_types)
# print(json.dumps(stats_summary_all, indent=4, sort_keys=True))

# calculate average logits for each prompt type and object
stats_obj_all = compute_obj_wise_stats_pt(labels_all, objs, id_to_name, prompt_types)
# print(json.dumps(stats_obj_all, indent=4, sort_keys=True))

# save the stats to a json file
with open(label_dir / "stats_summary_all.json", "w") as f:
    json.dump(stats_summary_all, f, indent=4)
with open(label_dir / "stats_obj_all.json", "w") as f:
    json.dump(stats_obj_all, f, indent=4)

# ### merge results from all prompt types into one

labels_all["one_and_syn"] = {}
for id in labels_all["one"]:
    labels_all["one_and_syn"][id] = copy.deepcopy(
        labels_all["one"][id]
    )  # NOTE: can't simply use =, it will be a reference
    if id in labels_all["synonym_sep"]:
        for key in labels_all["synonym_sep"][id]:
            labels_all["one_and_syn"][id][key] += labels_all["synonym_sep"][id][key]
# random check
for _ in range(10):
    id = random.choice(list(labels_all["synonym_sep"].keys()))
    assert len(labels_all["one_and_syn"][id]["boxes"]) == len(
        labels_all["one"][id]["boxes"]
    ) + len(labels_all["synonym_sep"][id]["boxes"])
# save dict
with open(label_dir / "labels_all.json", "w") as f:
    json.dump(labels_all, f)

# ## Extract labels with a stricter rule

# The rule:
# 1. if the image only gets one annotation, then keep it unless the score is too low (<0.2). In all our experiment, the unless part is actuall redundant, since we always specify a box_threshold with value bigger than 0.2, specifically {0.25, 0.3, 0.35, 0.4}.
# 2. if the image has multiple annotations,
#    1. If none of them is bigger than 0.5, simply keep the highest one.
#    2. If there are ones with score >=0.5, keep them

# get labels for target prompt type with strict rules
if args.pre_rule:
    prompt_types = ["one", "synonym_sep", "all"]
    labels, counts = get_labels_pre_rule(
        ann_dir, objs, combo, prompt_types, args.id_types
    )
else:
    prompt_types = ["one_and_syn"]
    labels = get_labels_post_rule(labels_all, id_to_name, combo, id_types=["0", "1"])

# calculate average logits for each prompt type
stats_summary = compute_stats_summary_pt(labels, prompt_types)
# print(json.dumps(stats_summary, indent=4, sort_keys=True))

# calculate average logits for each prompt type and object
stats_obj = compute_obj_wise_stats_pt(labels, objs, id_to_name, prompt_types)
# print(json.dumps(stats_obj, indent=4, sort_keys=True))

# ### merge results from all prompt types into one
if args.pre_rule:
    labels["one_and_syn"] = {}
    for id in labels["one"]:
        labels["one_and_syn"][id] = copy.deepcopy(
            labels["one"][id]
        )  # NOTE: can't simply use =, it will be a reference
        if id in labels["synonym_sep"]:
            for key in labels["synonym_sep"][id]:
                labels["one_and_syn"][id][key] += labels["synonym_sep"][id][key]
    for _ in range(10):
        id = random.choice(list(labels["synonym_sep"].keys()))
        assert len(labels["one_and_syn"][id]["boxes"]) == len(
            labels["one"][id]["boxes"]
        ) + len(labels["synonym_sep"][id]["boxes"])

# ## NMS
#
# merge synonyms as one class then nms

# merge with nms
ann_image_dir = ann_dir / "nms" / "images"
labels_nms = copy.deepcopy(labels["one_and_syn"])

for id in labels["one_and_syn"]:
    if labels["one_and_syn"][id]["boxes"] == []:
        continue

    boxes_xywh = torch.tensor(labels["one_and_syn"][id]["boxes"])
    boxes_xyxy = box_convert(boxes=boxes_xywh, in_fmt="cxcywh", out_fmt="xyxy")
    logits = torch.tensor(labels["one_and_syn"][id]["logits"])
    phrases = labels["one_and_syn"][id]["phrases"]

    obj = id_to_name[id + ".jpg"].split("/")[0]
    # # class-wise nms for obj in combo
    # if obj in combo:
    #     indices = [i for i, phrase in enumerate(phrases) if phrase == combo[obj]]
    #     # offset boxes:
    #     # since boxes are normalized, +=1 here is equivalent to move the boxes with combo obj to another photo along the diagonal and do nms for them there
    #     boxes_xyxy[indices] += 1
    #     # don't have to do anything for other boxes, they are mreged into one class then do nms automatically

    # nms, class-agnostic
    i = nms(boxes_xyxy, logits, 0.45)

    # label
    labels_nms[id]["boxes"] = [labels_nms[id]["boxes"][j] for j in i.tolist()]
    labels_nms[id]["logits"] = [labels_nms[id]["logits"][j] for j in i.tolist()]
    labels_nms[id]["phrases"] = [labels_nms[id]["phrases"][j] for j in i.tolist()]
    labels_nms[id]["classes"] = [labels_nms[id]["classes"][j] for j in i.tolist()]

# ## post-processing after nms
# if an image has only one label, then it should be the obj

for id in labels_nms:
    obj = id_to_name[id + ".jpg"].split("/")[0]
    if obj in combo:
        if len(labels_nms[id]["boxes"]) == 1:
            labels_nms[id]["phrases"] = ["post-procssing"]
            labels_nms[id]["classes"] = [obj]

    if args.save_img:
        boxes_xywh = torch.tensor(labels_nms[id]["boxes"])
        logits = torch.tensor(labels_nms[id]["logits"])
        phrases = labels_nms[id]["phrases"]
        image_path = image_dir / obj / (id + ".jpg")
        ann_image_path = ann_image_dir / obj / (id + ".jpg")
        # resume
        if not ann_image_path.exists():
            image = load_image(image_path, size=800)
            annotated_image = annotate(image, boxes_xywh, logits, phrases)
            ann_image_path.parent.mkdir(parents=True, exist_ok=True)
            annotated_image.save(ann_image_path)


# update labels
labels["nms"] = labels_nms

# overwrite with the updated labels
with open(label_dir / "labels.json", "w") as f:
    json.dump(labels, f)
# save the nms labels separately
with open(label_dir / "labels_nms.json", "w") as f:
    json.dump(labels_nms, f)

# ## after nms

# calculate average logits for each prompt type
stats_summary_nms = compute_stats_summary_pt(labels, prompt_types=["nms"])
print(json.dumps(stats_summary_nms, indent=4, sort_keys=True))

# calculate average logits for each prompt type and object
stats_obj_nms = compute_obj_wise_stats_pt(
    labels, objs, id_to_name, prompt_types=["nms"]
)
print(json.dumps(stats_obj_nms, indent=4, sort_keys=True))

# update stats_summary and stats_obj
stats_summary.update(stats_summary_nms)
stats_obj.update(stats_obj_nms)

# save the update
with open(label_dir / "stats_summary.json", "w") as f:
    json.dump(stats_summary, f, indent=4)
with open(label_dir / "stats_obj.json", "w") as f:
    json.dump(stats_obj, f, indent=4)

# get ids of images with no annotations
no_ann = []
for key in labels_nms:
    if len(labels_nms[key]["boxes"]) == 0:
        no_ann.append(key)
print(f"Number of images with no annotations after nms: {len(no_ann)}")

# save list of images with no annotations
with open(label_dir / "no_ann.json", "w") as f:
    json.dump(no_ann, f)
