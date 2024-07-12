from ultralytics import YOLO
from ray import tune
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cfg_dir", type=str, default="/mnt/ssd2/xin/repo/DART/yolo/cfg/datasets/"
)
parser.add_argument("--cfg", type=str, default="fine-tune.yaml")
parser.add_argument("--project", type=str, default="raytune")
parser.add_argument("--model", type=str, default="yolov8n.pt")
parser.add_argument("--trials", type=int, default=10)
args = parser.parse_args()

cfg = args.cfg_dir + args.cfg
project = args.project + "_" + args.model.replace(".pt", "")
name = args.cfg.replace(".yaml", "").replace("/", "_")

# Load a YOLOv8n model
model = YOLO(args.model)
# Start tuning hyperparameters for YOLOv8n training on the COCO8 dataset
result_grid = model.tune(
    data=cfg,
    epochs=60,
    space={
        "lr0": tune.loguniform(7e-5, 5e-4),  # TODO
        "lrf": tune.choice([0.1, 0.5, 0.01]),
        "warmup_epochs": tune.randint(1, 5),
        "cos_lr": tune.choice([True, False]),
        "batch": tune.choice([16, 32, 64]),
    },
    use_ray=True,
    project=project,
    gpu_per_trial=1,  # have to specify this, otherwise no GPU will be used
    name=name,
    optimizer="AdamW",  # https://github.com/ultralytics/ultralytics/issues/2849#issuecomment-1727931937,
    iterations=args.trials,
)
if result_grid.errors:
    print("One or more trials failed!")
else:
    print("No errors!")
for i, result in enumerate(result_grid):
    print(
        f"Trial #{i}: Configuration: {result.config}, Last Reported Metrics: {result.metrics}"
    )
