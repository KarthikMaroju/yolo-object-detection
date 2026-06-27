"""
evaluate.py — Evaluate YOLOv5 model on the validation set.
Reports: mAP@0.5, mAP@0.5:0.95, Precision, Recall.

Usage:
    python src/evaluate.py --weights models/best.pt --data config.yaml
"""

import argparse
import subprocess
import sys
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv5 model")
    parser.add_argument("--weights",    type=str, default="models/best.pt")
    parser.add_argument("--data",       type=str, default="config.yaml")
    parser.add_argument("--img",        type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--yolov5-dir", type=str, default="yolov5")
    parser.add_argument("--device",     type=str, default="")
    return parser.parse_args()


def evaluate(args):
    if not os.path.isdir(args.yolov5_dir):
        print(f"[ERROR] YOLOv5 not found: '{args.yolov5_dir}'")
        sys.exit(1)

    cmd = [
        sys.executable,
        os.path.join(args.yolov5_dir, "val.py"),
        "--weights",    os.path.abspath(args.weights),
        "--data",       os.path.abspath(args.data),
        "--img",        str(args.img),
        "--batch-size", str(args.batch_size),
        "--project",    "outputs/runs/eval",
        "--name",       "results",
        "--exist-ok",
    ]
    if args.device:
        cmd += ["--device", args.device]

    print("\n📊 Evaluating model...")
    print(f"   Weights : {args.weights}")
    print(f"   Data    : {args.data}\n")

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n✅ Evaluation complete!")
        print("   Results: outputs/runs/eval/results/")
    else:
        print("\n❌ Evaluation failed.")
        sys.exit(result.returncode)


if __name__ == "__main__":
    evaluate(parse_args())
