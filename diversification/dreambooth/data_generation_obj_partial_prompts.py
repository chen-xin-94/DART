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
parser.add_argument("--objs", nargs="+", help="Object names", default=[])
parser.add_argument("--seed", type=int, help="Base seed", default=0)
parser.add_argument(
    "--steps", type=int, help="Number of diffusion steps", default=25
)  # 25 is a default value for euler scheduler
parser.add_argument("--guidance_scale", type=float, help="Guidance scale", default=7.5)
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
repo_dir = Path(args.repo_dir)


# TODO: choose specific objs_
if len(args.objs) == 0:
    # # for water-related objects
    # objs_ = [
    #     "gantry_crane",
    #     "maritime_crane",
    #     "pontoon_excavator",
    # ]
    # # for mining-related objects
    objs_ = [
        "mining_bulldozer",
        "mining_excavator",
        "mining_truck",
    ]
else:
    objs_ = args.objs

class_data_dir = repo_dir / "dreambooth" / "class_data" / MODEL_NAME
instance_data_dir = repo_dir / "instance_data"
output_dir = repo_dir / "dreambooth" / "output" / MODEL_NAME
generated_data_dir = repo_dir / "dreambooth" / "generated_data_orig" / MODEL_NAME

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

        instance_generated_data_dir = obj_generated_data_dir / instance
        # # resume
        # if instance_generated_data_dir.exists():
        #     continue

        lora_path = obj_output_dir / instance
        pipeline.load_lora_weights(lora_path)

        placeholder = f"<{instance}> {obj}"
        if "underscore" in MODEL_NAME:
            placeholder = f"<{instance}> {obj_}"

        print(f"Generating images for {placeholder}...")

        # TODO: specific prompts for specific objects
        # water-related objects
        # prompts = {
        #     "ocean_dockyard": f"A high-quality, photorealistic image of a {placeholder} operating in an ocean dockyard, with large ships and containers in the background.",
        #     "river_dredging": f"A detailed, photorealistic image of a {placeholder} dredging a river, with muddy water and vegetation along the banks.",
        #     "coastal_maintenance": f"A high-resolution, photorealistic image of a {placeholder} performing coastal maintenance, with waves crashing in the background.",
        #     "offshore_construction": f"A photorealistic image of a {placeholder} working on an offshore construction project, with deep sea and platform structures.",
        #     "flood_control": f"A high-quality, photorealistic image of a {placeholder} engaged in flood control operations, with sandbags and rising water levels.",
        #     "canal_work": f"A detailed, photorealistic image of a {placeholder} working on a canal, with narrow waterways and boats passing by.",
        #     "shipyard_repair": f"A photorealistic image of a {placeholder} in a shipyard, repairing large vessels with workers around.",
        #     "waterfront_park": f"A high-quality, photorealistic image of a {placeholder} operating in a waterfront park, with recreational areas and people nearby.",
        #     "marine_research": f"A detailed, photorealistic image of a {placeholder} assisting in marine research activities, with scientists and research equipment around.",
        #     "dockside_assembly": f"A photorealistic image of a {placeholder} involved in dockside assembly of marine structures, with cranes and construction materials.",
        # }
        # mining-related objects
        prompts = {
            "open_pit_mine": f"A high-quality, photorealistic image of a {placeholder} working in an open-pit mine, with terraced rock formations and heavy machinery around.",
            "underground_mine": f"A detailed, photorealistic image of a {placeholder} operating in an underground mine, with tunnels and mining equipment.",
            "mining_town": f"A high-resolution, photorealistic image of a {placeholder} working near a mining town, with workers' housing and facilities in the background.",
            "quarry_operation": f"A photorealistic image of a {placeholder} working in a quarry, with large stone blocks and cutting equipment.",
            "mountain_mining": f"A high-quality, photorealistic image of a {placeholder} operating in a mountain mining site, with steep slopes and rocky terrain.",
            "remote_mine": f"A photorealistic image of a {placeholder} working in a remote mining location, with rugged terrain and minimal infrastructure.",
            "ore_processing": f"A high-quality, photorealistic image of a {placeholder} involved in ore processing, with conveyor belts and sorting equipment.",
            "tailings_dam": f"A detailed, photorealistic image of a {placeholder} working near a tailings dam, with water and waste materials.",
            "mining_reclamation": f"A high-resolution, photorealistic image of a {placeholder} involved in mining reclamation, with land restoration and vegetation.",
        }
        ks = list(prompts.keys())
        vs = list(prompts.values())
        for i in tqdm(range(N)):
            generator = [
                torch.Generator("cuda").manual_seed(SEED + (s + 1) * (i + 1))
                for s in range(len(prompts))
            ]
            # all prompts at once
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
