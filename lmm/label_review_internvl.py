# Done: add obj name as a parameter for the model to do label review
# TODO: also add score as a parameter for the model to do label review
# TODO: let the model specify which box is wrong, i.e. need a way to specify mo

from pathlib import Path

import argparse
import json
import time

from transformers import AutoTokenizer, AutoModel
import torch
from utils import load_image

types = ["nms", "one", "synonym", "super", "all", "all_synonym"]
parser = argparse.ArgumentParser(description="Label review script")
parser.add_argument(
    "-t", "--type", type=str, choices=types, help="Type of labeling", default="nms"
)
args = parser.parse_args()

## setup paths
dataset_dir = Path("/mnt/ssd2/xin/repo/DART/Liebherr_Product")

image_dir = dataset_dir / "annotations" / args.type / "images"
review_dir = dataset_dir / "reviews" / args.type
label_dir = dataset_dir / "labels"

with open(label_dir / "no_ann.json", "r") as f:
    no_ann = json.load(f)
# subfolder names of image_dir are the class names
objs = [f for f in image_dir.iterdir() if f.is_dir()]

## load the existing reviews
if (label_dir / "reviews.json").exists():
    with open(label_dir / "reviews.json", "r") as f:
        reviews = json.load(f)
else:
    reviews = {}
# reviews should be a dict of dicts of dicts
# reviews[type][id] = parsed_response
if args.type in reviews:
    raise ValueError(f"Review for {args.type} already exists")
else:
    reviews[args.type] = {}

## load the model
path = "OpenGVLab/InternVL-Chat-V1-5"
# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
model = (
    AutoModel.from_pretrained(
        path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    )
    .eval()
    .cuda()
)

# # Otherwise, you need to set device_map='auto' to use multiple GPUs for inference.
# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# model = AutoModel.from_pretrained(
#     path,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True,
#     device_map='auto').eval()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

generation_config = dict(
    num_beams=1,
    max_new_tokens=512,
    do_sample=False,
)

# save structurely wrong responses
wrong_responses = {}

for obj in objs:
    obj_image_dir = image_dir / obj
    obj_review_dir = review_dir / obj
    obj_review_dir.mkdir(parents=True, exist_ok=True)
    for img_path in obj_image_dir.rglob("*.jpg"):
        if img_path.stem in no_ann:
            # skip images without annotations
            continue
        pixel_values = load_image(img_path, max_num=6).to(torch.bfloat16).cuda()
        question = f"""The image is a result of the labeling process. Each labeled bounding box is drawn as a colored rectangle on top of the image. The goal of the labeling process is to localize all target objects by bounding boxes. The target object is {obj}. Your job is to judge whether all targets are correctly labeled. Please consider the following questions and provide answers accordingly? 1. Is there any target object that is not localized by a bounding box? 2.Is there any bounding box that localizes a wrong object? 3. Are all labeled bounding boxes appropriate, i.e., without being too loose or too small? Please give me back the output in the following json format：```json\{{"Answer": "YES OR NO", "Description": "IF THE ANSWER IS NO, DESCRIBE THE PROBLEM; IF THE ANSWER IS YES, OUTPUT NOTHING IN THIS KEY"}}```"""
        t0 = time.perf_counter()
        response = model.chat(tokenizer, pixel_values, question, generation_config)
        print(f"Reviewed {img_path.name} in {time.perf_counter() - t0:.2f}s")
        try:
            parsed_response = json.loads(
                response.replace("```json\n", "").replace("```", "")
            )
        except json.JSONDecodeError:
            print(f"response is in wrong format for {img_path.name}")
            wrong_responses[img_path.stem] = response
        reviews[args.type][img_path.stem] = parsed_response
        # save the response
        with open(obj_review_dir / (img_path.stem + ".json"), "w") as f:
            json.dump(parsed_response, f)

# save the reviews
with open(review_dir / "reviews.json", "w") as f:
    json.dump(reviews, f)

if len(wrong_responses) > 0:
    with open(review_dir / "wrong_responses.json", "w") as f:
        json.dump(wrong_responses, f)
    print("wrong responses are saved in wrong_responses.json")
