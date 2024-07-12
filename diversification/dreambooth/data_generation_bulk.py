import os

from pathlib import Path
from diffusers import DiffusionPipeline, AutoencoderKL
import torch
import argparse
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description="Data generation script")
parser.add_argument("-m", "--model", type=str, help="Model name", default="sdxl")
parser.add_argument(
    "-n", type=int, help="Number of images to generate for each prompt", default=10
)
parser.add_argument("--seed", type=int, help="Base seed", default=0)
parser.add_argument(
    "--steps", type=int, help="Number of diffusion steps", default=25
)  # 25 is a default value for euler scheduler
parser.add_argument("--guidance_scale", type=float, help="Guidance scale", default=7.5)
parser.add_argument(
    "--prompt_per_batch", type=int, help="Number of prompts per batch", default=16
)  # 16 is the maximum number for 80GB GPU memory
parser.add_argument(
    "--repo_dir",
    type=str,
    help="Repository directory",
    default="/mnt/ssd2/xin/repo/DART/diversification",
)
args = parser.parse_args()

N = args.n
SEED = args.seed
MODEL_NAME = args.model
STEPS = args.steps
GUIDANCE_SCALE = args.guidance_scale
prompt_per_batch = args.prompt_per_batch
repo_dir = Path(args.repo_dir)

class_data_dir = repo_dir / "dreambooth" / "class_data" / MODEL_NAME
instance_data_dir = repo_dir / "instance_data"
output_dir = repo_dir / "dreambooth" / "output" / MODEL_NAME
generated_data_dir = repo_dir / "dreambooth" / "generated_data_orig" / MODEL_NAME

objs_ = sorted([d.name for d in output_dir.iterdir() if d.is_dir()])

# load model
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.bfloat16
)
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
).to("cuda")

for obj_ in objs_:
    obj = obj_.replace("_", " ")
    obj_output_dir = output_dir / obj_
    obj_generated_data_dir = generated_data_dir / obj_
    instances = sorted([d.name for d in obj_output_dir.iterdir() if d.is_dir()])
    for instance in instances:

        placeholder = f"<{instance}> {obj}"
        if "underscore" in MODEL_NAME:
            placeholder = f"<{instance}> {obj_}"

        # 48 prompts in total
        prompts_dict = {
            "test": f"A photo of a {placeholder} on a construction site. The image is high quality and photorealistic. The {placeholder} may be partially visible, at a distance, or obscured. The background is complex, providing a realistic context.",
            "sunny_construction_site": f"A high-quality, photorealistic image of a {placeholder} under a bright, sunny sky at a bustling construction site.",
            "cloudy_construction_site": f"A detailed, photorealistic image of a {placeholder} operating at a construction site on a cloudy day.",
            "rainy_construction_site": f"A high-resolution, photorealistic image of a {placeholder} working at a muddy construction site during heavy rain.",
            "city_street": f"A photorealistic image of a {placeholder} on a busy city street, surrounded by tall buildings and traffic.",
            "rural_area": f"A high-quality, photorealistic image of a {placeholder} operating in a rural area, with fields and farmhouses in the background.",
            "mining_site": f"A detailed, photorealistic image of a {placeholder} working at a mining site, with rocky terrain and machinery in the background.",
            "harbor": f"A high-resolution, photorealistic image of a {placeholder} operating at a harbor, with ships and cranes in the background.",
            "left_side": f"A photorealistic image of a {placeholder} positioned on the left side of the frame at a construction site, with workers in the background.",
            "right_side": f"A high-quality, photorealistic image of a {placeholder} on the right side of the image, operating in a busy urban environment.",
            "in_the_distance": f"A detailed, photorealistic image of a {placeholder} visible in the distance at a construction site, with a panoramic view of the area.",
            "facing_left": f"A high-resolution, photorealistic image of a {placeholder} facing left, working at a construction site with scaffolding in the background.",
            "facing_right": f"A photorealistic image of a {placeholder} facing right, operating in a rural setting with rolling hills in the background.",
            "partially_visible": f"A high-quality, photorealistic image of a {placeholder} partially visible behind a building at a construction site.",
            "nighttime_construction": f"A photorealistic image of a {placeholder} working at night under artificial lights at a construction site.",
            "foggy_morning": f"A detailed, photorealistic image of a {placeholder} operating on a foggy morning, with limited visibility at a rural construction site.",
            "snowy_day": f"A high-resolution, photorealistic image of a {placeholder} working at a construction site during snowfall, with snow-covered equipment and ground.",
            "highway_construction": f"A photorealistic image of a {placeholder} operating on a highway under construction, with traffic cones and barriers.",
            "desert_construction": f"A high-quality, photorealistic image of a {placeholder} working at a construction site in a desert area, with sand dunes in the background.",
            "urban_demolition": f"A high-resolution, photorealistic image of a {placeholder} demolishing an old building in an urban area, with debris flying.",
            "windy_day": f"A high-quality, photorealistic image of a {placeholder}  working at a construction site on a windy day, with dust and debris blowing in the wind.",
            "industrial_area": f"A detailed, photorealistic image of a {placeholder}  operating in an industrial area, surrounded by factories and heavy machinery.",
            "riverbank_construction": f"A high-resolution, photorealistic image of a {placeholder} working on a riverbank, with water and vegetation in the background.",
            "urban_parking_lot": f"A photorealistic image of a {placeholder} operating in an urban parking lot construction site, with buildings and cars nearby.",
            "dam_construction": f"A high-quality, photorealistic image of {placeholder} working on a dam construction project, with water and concrete structures in the background.",
            "railway_construction": f"A detailed, photorealistic image of a {placeholder} working on a railway construction site, with tracks and trains in the background.",
            "solar_farm_construction": f"A high-resolution, photorealistic image of a {placeholder} operating at a solar farm construction site, with solar panels and wide open fields.",
            "skyscraper_construction": f"A photorealistic image of a {placeholder} working at a skyscraper construction site, with towering steel structures and cranes.",
            "wind_farm_construction": f"A high-quality, photorealistic image of a {placeholder} operating at a wind farm construction site, with wind turbines in the background.",
            "bridge_construction": f"A detailed, photorealistic image of a {placeholder} working on a bridge construction site over a river.",
            "mountain_construction": f"A high-quality, photorealistic image of a {placeholder} operating on a steep mountain construction site with rocky terrain.",
            "forest_construction": f"A photorealistic image of a {placeholder} working in a forest area, with tall trees and underbrush surrounding the site.",
            "coastal_construction": f"A detailed, photorealistic image of a {placeholder} operating at a coastal construction site, with the ocean and waves in the background.",
            "underground_construction": f"A high-resolution, photorealistic image of a {placeholder} working in an underground tunnel construction site, with dim lighting and rocky walls.",
            "large_scale_construction": f"A high-quality, photorealistic image of multiple {placeholder}s working together on a large-scale construction project, with cranes and scaffolding.",
            "airport_construction": f"A photorealistic image of a {placeholder} operating at an airport construction site, with planes and runways in the background.",
            "suburban_construction": f"A detailed, photorealistic image of a {placeholder} working in a suburban area, with houses and trees around the construction site.",
            "island_construction": f"A high-resolution, photorealistic image of a {placeholder} operating on an island construction site, surrounded by water and palm trees.",
            "urban_renewal": f"A high-quality, photorealistic image of a {placeholder} working on an urban renewal project, surrounded by modern buildings and greenery.",
            "industrial_site": f"A photorealistic image of a {placeholder} operating at an industrial construction site, with factories and smokestacks in the background.",
            "multiple_machines_sunny_site": f"A high-quality, photorealistic image of several {placeholder}s working together under a bright, sunny sky at a bustling construction site.",
            "multiple_machines_rainy_site": f"A high-resolution, photorealistic image of several {placeholder}s operating at a muddy construction site during heavy rain.",
            "multiple_machines_city_street": f"A photorealistic image of multiple {placeholder}s on a busy city street, surrounded by tall buildings and traffic.",
            "multiple_machines_rural_area": f"A high-quality, photorealistic image of several {placeholder}s operating in a rural area, with fields and farmhouses in the background.",
            "multiple_machines_harbor": f"A high-resolution, photorealistic image of several {placeholder}s operating at a harbor, with ships and cranes in the background.",
            "multiple_machines_night": f"A photorealistic image of several {placeholder}s working at night under artificial lights at a construction site.",
            "multiple_machines_snowy_day": f"A high-resolution, photorealistic image of several {placeholder}s working at a construction site during snowfall, with snow-covered equipment and ground.",
            "multiple_machines_highway": f"A photorealistic image of several {placeholder}s operating on a highway under construction, with traffic cones and barriers.",
        }

        # break the dict into prompt_per_batch parts depending on the memory usage of the gpu
        assert (
            len(prompts_dict.keys()) % prompt_per_batch == 0
        ), f"len(prompts_dict)={len(prompts_dict).keys()} is not divisible by prompt_per_batch={prompt_per_batch}"
        prompts_list = []
        for i in range(0, len(prompts_dict), prompt_per_batch):
            prompts_list.append(
                dict(list(prompts_dict.items())[i : i + prompt_per_batch])
            )

        instance_generated_data_dir = obj_generated_data_dir / instance
        # resume
        if (
            instance_generated_data_dir.exists()
            and len(list(instance_generated_data_dir.iterdir()))
            == len(prompts_list) * N
        ):
            continue

        lora_path = obj_output_dir / instance
        pipeline.load_lora_weights(lora_path)

        print(f"Generating images for {placeholder}...")

        for k, prompts in enumerate(prompts_list):
            ks = list(prompts.keys())
            vs = list(prompts.values())
            for i in tqdm(range(N)):

                if (instance_generated_data_dir / f"{ks[0]}_{i}.png").exists():
                    # resume
                    continue

                generator = [
                    torch.Generator("cuda").manual_seed(
                        SEED + (s + 1) * (i + 1) * (k + 1)
                    )
                    for s in range(len(prompts))
                ]
                images = pipeline(
                    prompt=vs,
                    generator=generator,
                    num_inference_steps=STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                ).images

                for j, image in enumerate(images):
                    image_path = instance_generated_data_dir / f"{ks[j]}_{i}.png"
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(image_path)

        pipeline.unload_lora_weights()
