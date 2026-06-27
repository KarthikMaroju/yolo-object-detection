"""
utils.py — Roboflow download, annotation parser, and result visualizer.
"""

import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def download_roboflow_dataset(api_key, workspace, project, version=1, fmt="yolov5"):
    """
    Download a dataset from Roboflow.

    Args:
        api_key:   Your Roboflow API key
        workspace: Roboflow workspace slug
        project:   Roboflow project slug
        version:   Dataset version (default: 1)
        fmt:       Export format  (default: 'yolov5')

    Returns:
        Local path string to the downloaded dataset

    Example:
        path = download_roboflow_dataset(
            api_key="YOUR_KEY",
            workspace="my-workspace",
            project="my-project",
        )
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        raise ImportError("Run: pip install roboflow")

    rf = Roboflow(api_key=api_key)
    dataset = (rf.workspace(workspace)
                 .project(project)
                 .version(version)
                 .download(fmt))
    print(f"✅ Dataset downloaded to: {dataset.location}")
    return dataset.location


def parse_yolo_annotation(label_path, img_w, img_h):
    """
    Parse a YOLO-format .txt annotation into absolute pixel bounding boxes.

    Each line format: <class_id> <cx> <cy> <w> <h>   (all values 0–1 normalised)

    Returns:
        List of dicts: {class_id, x1, y1, x2, y2}
    """
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            boxes.append({
                "class_id": cls,
                "x1": int((cx - bw / 2) * img_w),
                "y1": int((cy - bh / 2) * img_h),
                "x2": int((cx + bw / 2) * img_w),
                "y2": int((cy + bh / 2) * img_h),
            })
    return boxes


def visualize_detections(image_path, label_path, class_names, save_path=None):
    """
    Draw ground-truth bounding boxes on an image and display or save it.

    Args:
        image_path:  Path to the image file
        label_path:  Path to the YOLO .txt annotation
        class_names: Ordered list of class name strings
        save_path:   If set, save annotated image here instead of showing
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    boxes = parse_yolo_annotation(label_path, w, h)

    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img_rgb)
    colors = plt.cm.get_cmap("tab10", max(len(class_names), 1))

    for box in boxes:
        cls   = box["class_id"]
        label = class_names[cls] if cls < len(class_names) else str(cls)
        color = colors(cls)

        ax.add_patch(patches.Rectangle(
            (box["x1"], box["y1"]),
            box["x2"] - box["x1"],
            box["y2"] - box["y1"],
            linewidth=2, edgecolor=color, facecolor="none"
        ))
        ax.text(box["x1"], box["y1"] - 5, label,
                color=color, fontsize=10, fontweight="bold")

    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def show_detection_results(results_dir, max_images=6):
    """Display a grid of annotated YOLOv5 inference results."""
    paths = (list(Path(results_dir).glob("*.jpg")) +
             list(Path(results_dir).glob("*.png")))[:max_images]

    if not paths:
        print(f"No images found in: {results_dir}")
        return

    cols = min(3, len(paths))
    rows = (len(paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for ax, path in zip(axes, paths):
        img = cv2.imread(str(path))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(path.name, fontsize=9)
        ax.axis("off")

    for ax in axes[len(paths):]:
        ax.axis("off")

    plt.suptitle("Detection Results", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
