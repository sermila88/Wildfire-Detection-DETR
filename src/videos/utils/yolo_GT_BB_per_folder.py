#!/usr/bin/env python3
"""
Per-folder GT drawer & sorter for YOLO labels.

- For each scene folder under each split (train/val/test), decide the class:
  * "wildfire_smoke" if ANY label file in the scene has at least one box
    whose class id is in --smoke-classes (default: 0).
  * Otherwise "no_fire".
- Copies images into: <output_dir>/<split>/<class>/<scene_name>/
  * If in wildfire_smoke, draws GT boxes on frames that have labels.
  * If in no_fire, copies frames unchanged.
"""

import argparse
from pathlib import Path
import shutil
import cv2
from tqdm import tqdm

BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
BOX_COLOR = (0, 0, 255)  # red


def yolo_to_xyxy(xc, yc, w, h, W, H):
    x1 = int((xc - w / 2) * W)
    y1 = int((yc - h / 2) * H)
    x2 = int((xc + w / 2) * W)
    y2 = int((yc + h / 2) * H)
    return x1, y1, x2, y2


def frame_labels(label_path: Path):
    """Return list of (class_id, x_center, y_center, width, height) from a YOLO label file, or []."""
    if not label_path.exists():
        return []
    lines = [ln.strip() for ln in label_path.read_text().splitlines() if ln.strip()]
    labels = []
    for ln in lines:
        parts = ln.split()
        if len(parts) >= 5:
            try:
                cls = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                labels.append((cls, xc, yc, w, h))
            except ValueError:
                pass
    return labels


def scene_is_smoke(scene_dir: Path, smoke_classes: set[int]) -> bool:
    labels_dir = scene_dir / "labels"
    if not labels_dir.exists():
        return False
    for lbl in labels_dir.glob("*.txt"):
        if frame_labels(lbl):   # non-empty label file
            return True
    return False


def draw_boxes_on_image(img_path: Path, label_path: Path, out_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return
    H, W = img.shape[:2]
    labels = frame_labels(label_path)
    for (cls, xc, yc, w, h) in labels:
        x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, W, H)
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        # optional tiny class tag
        tag = f"{cls}"
        (tw, th), _ = cv2.getTextSize(tag, FONT, FONT_SCALE, FONT_THICKNESS)
        ty = max(y1 - 6, th + 6)
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), BOX_COLOR, -1)
        cv2.putText(img, tag, (x1 + 2, ty), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def process_scene(scene_dir: Path, split: str, out_base: Path, smoke_classes: set[int]) -> str:
    """Classify a scene, export images (draw GT if smoke), return class name."""
    is_smoke = scene_is_smoke(scene_dir, smoke_classes)
    cls_dir = "wildfire_smoke" if is_smoke else "no_fire"
    dest = out_base / split / cls_dir / scene_dir.name
    dest.mkdir(parents=True, exist_ok=True)

    # Iterate frames
    imgs = sorted(scene_dir.glob("*.jpg"))
    for img_path in imgs:
        lbl_path = scene_dir / "labels" / f"{img_path.stem}.txt"
        out_img = dest / img_path.name
        if is_smoke and lbl_path.exists():
            draw_boxes_on_image(img_path, lbl_path, out_img)
        else:
            # copy unchanged
            shutil.copy2(img_path, out_img)

    # Optionally copy labels folder for traceability
    labels_src = scene_dir / "labels"
    if labels_src.exists():
        labels_dest = dest / "labels"
        labels_dest.mkdir(exist_ok=True)
        for lf in labels_src.glob("*.txt"):
            shutil.copy2(lf, labels_dest / lf.name)

    return cls_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str,
                   default="/vol/bitbucket/si324/rf-detr-wildfire/src/videos/data")
    p.add_argument("--output_dir", type=str,
                   default="/vol/bitbucket/si324/rf-detr-wildfire/src/videos/bounding_boxes/GT_BB_per_folder")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--smoke-classes", type=str, default="0",
                   help="Comma-separated class IDs considered wildfire smoke (default: 0).")
    args = p.parse_args()

    in_root = Path(args.input_dir)
    out_root = Path(args.output_dir)
    smoke_classes = {int(s) for s in args.smoke_classes.split(",") if s.strip() != ""}

    stats = {}  # (split, class) -> count

    for split in args.splits:
        split_dir = in_root / split
        if not split_dir.exists():
            print(f"⚠️  Missing split: {split_dir} (skipping)")
            continue

        scene_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        print(f"\n▶ Processing split: {split}  |  {len(scene_dirs)} scenes")
        for scene_dir in tqdm(scene_dirs, desc=f"{split} scenes"):
            cls_dir = process_scene(scene_dir, split, out_root, smoke_classes)
            stats[(split, cls_dir)] = stats.get((split, cls_dir), 0) + 1

    # Report
    print("\n✅ Done. Sequence breakdown:")
    by_split = {}
    for (split, cls), cnt in stats.items():
        by_split.setdefault(split, {})[cls] = cnt

    for split in sorted(by_split.keys()):
        smk = by_split[split].get("wildfire_smoke", 0)
        nf = by_split[split].get("no_fire", 0)
        total = smk + nf
        print(f"  {split}: {total} scenes "
              f"(wildfire_smoke={smk}, no_fire={nf})")

    print(f"\nOutput saved under: {out_root}")


if __name__ == "__main__":
    main()
