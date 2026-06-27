"""
detect.py — Run YOLOv5 inference on image / video / webcam.

Usage:
    python src/detect.py --source data/images/sample.jpg --weights models/best.pt
    python src/detect.py --source video.mp4 --weights models/best.pt
    python src/detect.py --source 0 --weights models/best.pt   # webcam
"""

import argparse
import subprocess
import sys
import os


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv5 inference")
    parser.add_argument("--source",     type=str,   required=True,
                        help="Image path, video path, or '0' for webcam")
    parser.add_argument("--weights",    type=str,   default="models/best.pt")
    parser.add_argument("--img",        type=int,   default=640)
    parser.add_argument("--conf",       type=float, default=0.4)
    parser.add_argument("--iou",        type=float, default=0.45)
    parser.add_argument("--save-dir",   type=str,   default="outputs/detections")
    parser.add_argument("--view-img",   action="store_true")
    parser.add_argument("--yolov5-dir", type=str,   default="yolov5")
    parser.add_argument("--device",     type=str,   default="")
    return parser.parse_args()


def validate(args):
    if not os.path.isdir(args.yolov5_dir):
        print(f"[ERROR] YOLOv5 not found: '{args.yolov5_dir}'")
        sys.exit(1)
    if args.source != "0" and not os.path.exists(args.source):
        print(f"[ERROR] Source not found: '{args.source}'")
        sys.exit(1)
    if not os.path.exists(args.weights):
        print(f"[ERROR] Weights not found: '{args.weights}'")
        sys.exit(1)


def detect(args):
    validate(args)
    os.makedirs(args.save_dir, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(args.yolov5_dir, "detect.py"),
        "--source",     args.source,
        "--weights",    os.path.abspath(args.weights),
        "--img",        str(args.img),
        "--conf-thres", str(args.conf),
        "--iou-thres",  str(args.iou),
        "--project",    os.path.abspath(args.save_dir),
        "--name",       "results",
        "--exist-ok",
    ]
    if args.view_img:
        cmd.append("--view-img")
    if args.device:
        cmd += ["--device", args.device]

    print("\n🔍 Running inference...")
    print(f"   Source  : {args.source}")
    print(f"   Weights : {args.weights}")
    print(f"   Conf    : {args.conf}")
    print(f"   IoU     : {args.iou}\n")

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✅ Done! Results saved to: {args.save_dir}/results/")
    else:
        print("\n❌ Inference failed.")
        sys.exit(result.returncode)


if __name__ == "__main__":
    detect(parse_args())
