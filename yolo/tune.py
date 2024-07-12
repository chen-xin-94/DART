from ultralytics import YOLO
import argparse
import random
from scipy.stats import loguniform
import numpy as np

# random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cfg_dir", type=str, default="/mnt/ssd2/xin/repo/DART/yolo/cfg/datasets/"
)
parser.add_argument("--cfg", type=str, default="fine-tune.yaml")
parser.add_argument("--project", type=str, default=None)
parser.add_argument("--model", type=str, default="yolov8n.pt")
parser.add_argument("--trials", type=int, default=10)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--finer", action="store_true", default=False, help="finer search")
parser.add_argument(
    "--finest", action="store_true", default=False, help="finest search"
)

args = parser.parse_args()

cfg = args.cfg_dir + args.cfg

project = args.project if args.project else "tune_" + args.model.replace(".pt", "")
name = args.cfg.replace(".yaml", "").replace("/", "_")

lr0s = [
    1e-5,
    5e-5,
    1e-4,
    5e-4,
    1e-3,
    5e-3,
    # 1e-2,
]
low = None
high = None
if args.finer:
    lr0s = None
    low = 7e-5
    high = 7e-4
if args.finest:
    lr0s = None
    low = 7e-5
    high = 3e-4

lrfs = [0.01, 0.1, 0.5]
batches = [16, 32, 64]
warmup_epochs = [1, 2, 3, 4]
cos_lrs = [True, False]


def generate_one_sample(low, high, lr0s, lrfs, batches, warmup_epochs, cos_lrs):
    if lr0s:
        lr0_sample = random.choice(lr0s)
    else:
        lr0_sample = loguniform.rvs(low, high, size=1)[0].astype(np.float32).item()
    lrf_sample = random.choice(lrfs)
    batch_sample = random.choice(batches)
    warmup_epoch_sample = random.choice(warmup_epochs)
    cos_lr_sample = random.choice(cos_lrs)
    return lr0_sample, lrf_sample, batch_sample, warmup_epoch_sample, cos_lr_sample


for i in range(args.trials):
    # Load a YOLO model
    model = YOLO(args.model)
    # randomly sample hyperparameters
    lr0, lrf, batch, warmup_epoch, cos_lr = generate_one_sample(
        low, high, lr0s, lrfs, batches, warmup_epochs, cos_lrs
    )

    print(
        f"lr0: {lr0}, lrf: {lrf}, batch: {batch}, warmup_epoch: {warmup_epoch}, cos_lr: {cos_lr}"
    )
    results = model.train(
        data=cfg,
        epochs=60,
        batch=batch,
        lr0=lr0,
        lrf=lrf,
        warmup_epochs=warmup_epoch,
        cos_lr=cos_lr,
        project=project,
        name=name,
        optimizer="AdamW",
        patience=args.patience,
    )
