import os

# to turn off SupervisionWarnings, note that the following code must be run before importing anything from supervision
# https://github.com/roboflow/supervision/pull/962/commits/7d65a9445aa04d82334c6fe2828278e528eda1cb
os.environ["SUPERVISON_DEPRECATION_WARNING"] = "0"

import itertools
import json
import yaml
import time
from matplotlib import pyplot as plt
from pathlib import Path
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
from utils import undo_transform

import argparse

prompt_types = ["one", "synonym", "all"]
# Parse command line arguments
parser = argparse.ArgumentParser(description="Labeling script")
parser.add_argument(
    "-p",
    "--prompt_type",
    type=str,
    choices=prompt_types,
    help="Prompt type",
    default="one",
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    help="Path to the YAML config file",
    default="configs/GroundingDINO/base.yaml",
)
parser.add_argument("--vis", action="store_true", help="Draw bounding boxes and save")
parser.add_argument("--obj", nargs="+", help="Object to label", default=None)

args = parser.parse_args()

# setup paths
dataset_dir = Path("/mnt/ssd2/xin/repo/DART/Liebherr_Product")
repo_dir = Path("/mnt/ssd2/xin/repo/GroundingDINO")

# setup configs
with open(args.config, "r") as f:
    config = yaml.safe_load(f)
prompt_type = args.prompt_type
config_path = repo_dir / config["config_path"]
weight_path = repo_dir / config["weight_path"]
box_threshold = config["box_threshold"]
text_threshold = config["text_threshold"]

image_dir = dataset_dir / "images"
metadata_dir = dataset_dir / "metadata"
model_type = "base" if "base" in args.config else "tiny"
if model_type == "base":
    ann_dir_name = "annotations"
else:  # tiny
    ann_dir_name = f"annotations_{model_type}"
if (
    not box_threshold == 0.27 and text_threshold == 0.25
):  # final thresholds after hyperparameter tuning
    ann_dir_name += f"_{box_threshold:.2f}_{text_threshold:.2f}"
ann_dir = dataset_dir / ann_dir_name / args.prompt_type
ann_label_dir = ann_dir / "labels"
ann_image_dir = ann_dir / "images"

# load classes and synonyms
with open(metadata_dir / "synonyms.json", "r") as f:
    synonyms = json.load(f)
with open(metadata_dir / "superclasses.json", "r") as f:
    superclasses = json.load(f)
with open(metadata_dir / "combo.json", "r") as f:
    combo = json.load(f)
# merge synonyms and superclasses
for obj, superclass in superclasses.items():
    synonyms[obj] = synonyms.get(obj, []) + [superclass]

if args.obj:
    objs = args.obj
else:
    objs = list(synonyms.keys())

# GroungingDINO
model = load_model(
    model_config_path=config_path, model_checkpoint_path=weight_path, device="cuda"
)
for obj in objs:
    obj_dir = image_dir / obj
    image_paths = sorted(
        [image_path for image_path in obj_dir.iterdir() if image_path.suffix == ".jpg"]
    )
    # prompt
    synonym = synonyms[obj]
    if prompt_type == "one":
        if obj in combo:  # if in combo, then the prompt should consist of two objects
            text_prompt = f"{obj} . {combo[obj]} ."
        else:
            text_prompt = f"{obj} ."
    elif (
        prompt_type == "synonym"
    ):  # NOTE: here obj is also included, but not included in labeling_sep.py
        text_prompt = " . ".join([obj] + synonym) + " ."
    elif prompt_type == "all":
        text_prompt = " . ".join(objs) + " ."
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    # labeling
    for image_path in image_paths:
        ann_dict_path = ann_label_dir / obj / image_path.name.replace(".jpg", ".json")
        # resume
        if ann_dict_path.exists():
            continue
        ann_dict_path.parent.mkdir(parents=True, exist_ok=True)
        # groundingDINO
        t0 = time.perf_counter()
        image_source, image = load_image(image_path)
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            remove_combined=True,  # https://github.com/IDEA-Research/GroundingDINO/issues/63#issuecomment-2092655839
        )
        t1 = time.perf_counter()
        print(f"Labeled {image_path.name} in {1000*(t1 - t0):.2f}ms.")

        # ann dict
        ann_dict_image = {
            "boxes": boxes.tolist(),
            "logits": logits.tolist(),
            "phrases": phrases,
        }

        # save
        with open(ann_dict_path, "w") as f:
            json.dump(ann_dict_image, f)
        if args.vis:
            # ann image
            np_image = undo_transform(image)
            # label the resized image
            annotated_frame = annotate(
                image_source=np_image, boxes=boxes, logits=logits, phrases=phrases
            )
            ann_image_path = ann_image_dir / obj / image_path.name
            ann_image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(ann_image_path), annotated_frame)
