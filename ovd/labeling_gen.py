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

# Parse command line arguments
parser = argparse.ArgumentParser(description="Labeling script")
parser.add_argument(
    "-c",
    "--config",
    type=str,
    help="Path to the YAML config file",
    default="configs/GroundingDINO/base.yaml",
)
parser.add_argument("--vis", action="store_true", help="Visualize the annotations")
parser.add_argument("--objs_", nargs="+", help="Object to label", default=None)
parser.add_argument(
    "-d",
    "--dreambooth_repo_dir",
    type=str,
    help="Path to the dataset directory",
    default="/mnt/ssd2/xin/repo/DART/diversification/dreambooth/",
)
parser.add_argument(
    "-r",
    "--gdino_repo_dir",
    type=str,
    help="Path to the repo directory",
    default="/mnt/ssd2/xin/repo/GroundingDINO",
)


args = parser.parse_args()

dreambooth_repo_dir = Path(args.dreambooth_repo_dir)
gdino_repo_dir = Path(args.gdino_repo_dir)


# setup configs
with open(args.config, "r") as f:
    config = yaml.safe_load(f)
config_path = gdino_repo_dir / config["config_path"]
weight_path = gdino_repo_dir / config["weight_path"]
box_threshold = config["box_threshold"]
text_threshold = config["text_threshold"]

image_dir = dreambooth_repo_dir / "generated_data"
ann_dir = dreambooth_repo_dir / "generated_data_annotations"
ann_label_dir = ann_dir / "labels"
ann_image_dir = ann_dir / "images"

# GroungingDINO
model = load_model(
    model_config_path=config_path, model_checkpoint_path=weight_path, device="cuda"
)

# labeling
for image_path in image_dir.rglob("*.jpg"):
    obj = image_path.parent.name
    ann_dict_path = ann_label_dir / obj / image_path.name.replace(".jpg", ".json")
    # resume
    if ann_dict_path.exists():
        continue
    ann_dict_path.parent.mkdir(parents=True, exist_ok=True)

    # groundingDINO
    text_prompt = f"{obj} ."
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
