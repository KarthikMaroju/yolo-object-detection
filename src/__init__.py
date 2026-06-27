"""
train.py — Fine-tune YOLOv5 on a custom Roboflow dataset.

Usage:
    python src/train.py --data config.yaml --weights yolov5s.pt --epochs 50
"""

import argparse
import subprocess
import sys
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv5 on a custom dataset")
    parser.add_argument("--data", type=str, default="config.yaml")
    parser.add_argument("--weights", type=str, default="yolov5s.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--name", type=str, default="custom_run")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--yolov5-dir", type=str, default="yolov5")
    return parser.parse_args()


def check_yolov5(yolov5_dir):
    if not os.path.isdir(yolov5_dir):
        print(f"[ERROR] YOLOv5 not found at '{yolov5_dir}'")
        print("  Run: git clone https://github.com/ultralytics/yolov5")
        sys.exit(1)


def train(args):
    check_yolov5(args.yolov5_dir)
    data_path = os.path.abspath(args.data)

    cmd = [
        sys.executable,
        os.path.join(args.yolov5_dir, "train.py"),
        "--data",       data_path,
        "--weights",    args.weights,
        "--epochs",     str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--img",        str(args.img),
        "--name",       args.name,
        "--project",    "outputs/runs/train",
    ]
    if args.device:
        cmd += ["--device", args.device]

    print("\n🚀 Starting YOLOv5 training...")
    print(f"   Config  : {data_path}")
    print(f"   Weights : {args.weights}")
    print(f"   Epochs  : {args.epochs}")
    print(f"   Batch   : {args.batch_size}\n")

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n✅ Training complete!")
        print(f"   Best weights: outputs/runs/train/{args.name}/weights/best.pt")
    else:
        print("\n❌ Training failed.")
        sys.exit(result.returncode)


if __name__ == "__main__":
    train(parse_args())
