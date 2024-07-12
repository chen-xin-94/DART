import os

# to turn off SupervisionWarnings, note that the following code must be run before importing anything from supervision
# https://github.com/roboflow/supervision/pull/962/commits/7d65a9445aa04d82334c6fe2828278e528eda1cb
os.environ["SUPERVISON_DEPRECATION_WARNING"] = "0"

from pathlib import Path
import json

import copy
import torch
from torchvision.ops import nms, box_convert
from utils import (
    get_labels_rule,
    compute_stats_summary,
    compute_obj_wise_stats,
)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--save_dir", type=str, default="labels_gen", help="Path to the label directory"
)
parser.add_argument(
    "--dataset_dir",
    type=str,
    default="/mnt/ssd2/xin/repo/DART/Liebherr_Product",
    help="Path to the dataset directory",
)
parser.add_argument(
    "--repo_dir",
    type=str,
    default="/mnt/ssd2/xin/repo/DART/diversification/dreambooth",
    help="Path to the repo directory",
)
args = parser.parse_args()

dataset_dir = Path(args.dataset_dir)
repo_dir = Path(args.repo_dir)

# Define the images directory and duplicates directory using Path objects
image_dir = repo_dir / "generated_data"
ann_dir = repo_dir / "generated_data_annotations" / "labels"
save_dir = dataset_dir / args.save_dir
save_dir.mkdir(exist_ok=True)

# List all objects in the image directory
objs = sorted([obj.name for obj in image_dir.iterdir() if obj.is_dir()])

labels = get_labels_rule(objs, image_dir, ann_dir)
stats_summary = compute_stats_summary(labels)
stats_obj = compute_obj_wise_stats(labels, objs)

# save
with open(save_dir / "labels.json", "w") as f:
    json.dump(labels, f)
with open(save_dir / "stats_summary.json", "w") as f:
    json.dump(stats_summary, f, indent=4)
with open(save_dir / "stats_obj.json", "w") as f:
    json.dump(stats_obj, f, indent=4)

# ## NMS

# merge with nms
labels_nms = copy.deepcopy(labels)

for id in labels:
    if len(labels[id]["boxes"]) == 0:
        continue
    boxes_xywh = torch.tensor(labels[id]["boxes"])
    boxes_xyxy = box_convert(boxes=boxes_xywh, in_fmt="cxcywh", out_fmt="xyxy")
    logits = torch.tensor(labels[id]["logits"])
    phrases = labels[id]["phrases"]
    obj = labels[id]["classes"][0]
    # nms
    i = nms(boxes_xyxy, logits, 0.45)
    # label
    labels_nms[id]["boxes"] = [labels_nms[id]["boxes"][j] for j in i.tolist()]
    labels_nms[id]["logits"] = [labels_nms[id]["logits"][j] for j in i.tolist()]
    labels_nms[id]["phrases"] = [labels_nms[id]["phrases"][j] for j in i.tolist()]
    labels_nms[id]["classes"] = [labels_nms[id]["classes"][j] for j in i.tolist()]

# save the nms labels separately
with open(save_dir / "labels_nms.json", "w") as f:
    json.dump(labels_nms, f)

# ## after nms
stats_summary_nms = compute_stats_summary(labels_nms)
stats_obj_nms = compute_obj_wise_stats(labels_nms, objs)

print(json.dumps(stats_summary_nms, indent=4, sort_keys=True))
print(json.dumps(stats_obj_nms, indent=4, sort_keys=True))

# save the update
with open(save_dir / "stats_summary_nms.json", "w") as f:
    json.dump(stats_summary_nms, f, indent=4)
with open(save_dir / "stats_obj_nms.json", "w") as f:
    json.dump(stats_obj_nms, f, indent=4)

# get ids of images with no annotations
no_ann = []
for key in labels_nms:
    if len(labels_nms[key]["boxes"]) == 0:
        no_ann.append(key)
print(f"Number of generated data with no annotations: {len(no_ann)}")

# save list of images with no annotations
with open(save_dir / "no_ann.json", "w") as f:
    json.dump(no_ann, f)
