from pathlib import Path
from PIL import Image
import json
from transformers import AutoTokenizer, AutoModel
import torch
from utils import load_image
import time
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch", action="store_true", help="Batch processing")
args = parser.parse_args()

gen_data_dir = Path("../diversification/dreambooth/generated_data")
response_dir = Path(
    "../diversification/dreambooth/generated_data_annotations/responses"
)

## load model
model_path = "OpenGVLab/InternVL-Chat-V1-5"
# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
model = (
    AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
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

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

generation_config = dict(
    num_beams=1,
    max_new_tokens=512,
    do_sample=False,
)

## inference
image_list = list(gen_data_dir.glob("**/*.jpg"))
for image_path in tqdm(image_list):
    # text prompt
    obj = image_path.parent.name
    questions = [
        f"Is this image suitable as training data object detection for {obj}? Answer YES or NO.",
        f"Does the main object in the image look like an authentic {obj}? Answer YES or NO.",
    ]
    # resume
    response_path = response_dir / obj / image_path.name.replace(".jpg", ".json")
    if response_path.exists():
        with open(response_path, "r") as f:
            saved_dict = json.load(f)
        if len(saved_dict.get("responses", [])) == len(questions):
            continue

    response_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    # visual prompt
    pixel_values = load_image(image_path, max_num=6).to(torch.bfloat16).cuda()

    saved_dict = {"questions": questions, "responses": []}

    if args.batch:  # batch processing
        pixel_values_batch = torch.cat([pixel_values] * len(questions), dim=0)
        image_counts = [pixel_values.size(0)] * len(questions)
        response = model.batch_chat(
            tokenizer,
            pixel_values_batch,
            image_counts=image_counts,
            questions=questions,
            generation_config=generation_config,
        )
        saved_dict["responses"] = response
    else:  # single processing
        for i, question in enumerate(questions):
            response = model.chat(tokenizer, pixel_values, question, generation_config)
            saved_dict["responses"].append(response)

    print(
        f"processed {image_path.name} with {len(questions)} questions in {time.perf_counter() - t0:.2f}s"
    )
    with open(response_path, "w") as f:
        json.dump(saved_dict, f)
