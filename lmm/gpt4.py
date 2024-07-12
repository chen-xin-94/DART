from pathlib import Path
import json

from openai import OpenAI, OpenAIError
import base64
from prompts import get_prompts
from tqdm import tqdm


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Function to get the response from the API
def get_response(client, prompts, base64_image):
    return client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompts["system"]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompts["task"]
                        + prompts["questions"]
                        + prompts["be-relax"]
                        + prompts["caveat"]
                        + prompts["output"],
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ],
        temperature=0.0,
    )


dataset_dir = Path("/mnt/ssd2/xin/repo/DART/Liebherr_Product")


# Define the images directory and duplicates directory using Path objects
image_annotated_dir = dataset_dir / "images_annotated"
meta_dir = dataset_dir / "metadata"
ann_dir = dataset_dir / "annotations"
label_dir = dataset_dir / "labels"
review_dir = dataset_dir / "reviews"


with open(label_dir / "labels.json", "r") as f:
    labels_nms = json.load(f)["nms"]
with open(meta_dir / "id_to_name.json", "r") as f:
    id_to_name = json.load(f)
with open(meta_dir / "combo.json", "r") as f:
    combo = json.load(f)
with open(meta_dir / "to_gpt.json", "r") as f:
    to_gpt = json.load(f)

for id in tqdm(to_gpt):
    obj = id_to_name[id + ".jpg"].split("/")[0]

    review_path = review_dir / obj / (id + ".json")
    review_path.parent.mkdir(parents=True, exist_ok=True)
    # resume
    if review_path.exists():
        continue

    print(f"Processing {image_annotated_dir}/{obj}/{id}.jpg")
    target = [obj]
    secondary_target = None
    if obj in combo:
        secondary_target = [combo[obj]]

    prompts = get_prompts(target, secondary_target)

    base64_image = encode_image(image_annotated_dir / obj / (id + ".jpg"))

    client = OpenAI()
    MODEL = "gpt-4o"
    try:
        response = get_response(client, prompts, base64_image)
    # except error, try one more time
    # since sometimes the first request fails, but the second request succeeds
    except OpenAIError as e:
        print(e)
        print(f"Retrying {image_annotated_dir}/{obj}/{id}.jpg")
        response = get_response(client, prompts, base64_image)

    with open(review_path, "w") as f:
        json.dump(json.loads(response.json()), f, indent=4)
