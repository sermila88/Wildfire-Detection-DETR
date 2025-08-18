#!/usr/bin/env python3
"""
Counts YOLO-style dataset items.

Outputs:
- Split-level summary for train/val/test:
    frames, labels, labeled_frames (non-empty labels), non_labeled_frames, empty_label_files
- Per-sequence breakdown saved to CSV.

Assumptions (robust to both):
A) root/split/<sequence>/...images... and root/split/<sequence>/labels/*.txt
B) root/split/images/*.jpg and root/split/labels/*.txt

"Labeled frame" = image with a matching, NON-EMPTY label file.
"Empty label file" = .txt exists but 0 bytes or only whitespace (no boxes).
"Non-labeled frame" = image with no matching .txt at all.
"""

from pathlib import Path
import argparse
import csv

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def is_empty_label(label_path: Path) -> bool:
    try:
        if label_path.stat().st_size == 0:
            return True
        txt = label_path.read_text(encoding="utf-8", errors="ignore").strip()
        return len(txt) == 0
    except FileNotFoundError:
        return False

def find_sequences(split_dir: Path):
    """
    Return a list of sequence roots under this split.
    If split has images/labels at its root, treat the split_dir itself as one 'sequence'.
    Otherwise, each immediate subdir (except 'labels') is a sequence root.
    """
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    if images_dir.is_dir() and labels_dir.is_dir():
        return [split_dir]  # flat layout
    # otherwise, sequences are immediate subfolders that have images or a labels dir
    seqs = []
    for d in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        if d.name.lower() == "labels":
            continue
        if (d / "labels").is_dir():
            seqs.append(d)
        else:
            # has at least one image somewhere beneath?
            if any(is_image(p) for p in d.rglob("*") if p.is_file()):
                seqs.append(d)
    return seqs

def iter_images_in_sequence(seq_root: Path, split_dir: Path):
    """
    Yield image files within a sequence root, excluding anything inside 'labels' directories.
    For flat layout (split_dir has images/labels), seq_root == split_dir and images are under split_dir/images.
    """
    images_dir = seq_root / "images" if (seq_root / "images").is_dir() else seq_root
    for p in images_dir.rglob("*"):
        if p.is_file() and is_image(p) and "labels" not in {x.name.lower() for x in p.parents}:
            yield p

def label_for_image(img_path: Path, seq_root: Path, split_dir: Path) -> Path:
    """
    Map an image to its label file:
    - Prefer <sequence>/labels/<stem>.txt
    - For flat layout: <split>/labels/<stem>.txt
    - Fallback: same directory as image, <stem>.txt (rare)
    """
    stem = img_path.stem
    # 1) sequence-level labels
    seq_labels = seq_root / "labels" / f"{stem}.txt"
    if seq_labels.exists():
        return seq_labels
    # 2) flat split-level labels
    split_labels = split_dir / "labels" / f"{stem}.txt"
    if split_labels.exists():
        return split_labels
    # 3) same folder fallback
    same_dir = img_path.with_suffix(".txt")
    return same_dir

def count_split(split_dir: Path):
    """
    Returns:
      totals: dict for the split
      rows: per-sequence rows for CSV
    """
    if not split_dir.is_dir():
        return None, []

    sequences = find_sequences(split_dir)
    rows = []

    tot_frames = tot_labels = 0
    tot_labeled_frames = tot_non_labeled = 0
    tot_empty_labels = 0

    for seq_root in sequences:
        # sequence name
        seq_name = seq_root.name if seq_root != split_dir else "<root>"

        # label files present in this sequence
        seq_label_dir = seq_root / "labels" if (seq_root / "labels").is_dir() else (split_dir / "labels")
        label_files = []
        if seq_label_dir.is_dir():
            label_files = [p for p in seq_label_dir.rglob("*.txt") if p.is_file()]

        frames = 0
        labeled_frames = 0
        non_labeled = 0
        empty_labels = 0

        # count empty label files for this sequence
        for lf in label_files:
            if is_empty_label(lf):
                empty_labels += 1

        # iterate images and check matching labels
        for img in iter_images_in_sequence(seq_root, split_dir):
            frames += 1
            lab = label_for_image(img, seq_root, split_dir)
            if lab.exists():
                if not is_empty_label(lab):
                    labeled_frames += 1
                # if empty, we neither add to labeled_frames nor to non_labeled
            else:
                non_labeled += 1

        rows.append({
            "split": split_dir.name,
            "sequence": seq_name,
            "frames": frames,
            "label_files": len(label_files),
            "labeled_frames": labeled_frames,
            "non_labeled_frames": non_labeled,
            "empty_label_files": empty_labels,
        })

        # accumulate totals
        tot_frames += frames
        tot_labels += len(label_files)
        tot_labeled_frames += labeled_frames
        tot_non_labeled += non_labeled
        tot_empty_labels += empty_labels

    totals = {
        "split": split_dir.name,
        "frames": tot_frames,
        "labels": tot_labels,
        "labeled_frames": tot_labeled_frames,
        "non_labeled_frames": tot_non_labeled,
        "empty_label_files": tot_empty_labels,
        "num_sequences": len(sequences),
    }
    return totals, rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="Dataset root containing train/val/test")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    help="Which splits to scan")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Optional path to save per-sequence breakdown CSV")
    args = ap.parse_args()

    all_rows = []
    print("=== Dataset Counts ===")

    for split in args.splits:
        split_dir = args.root / split
        totals, rows = count_split(split_dir)
        if totals is None:
            print(f"[{split}] not found at {split_dir}")
            continue

        # top summary per split
        print(f"\n[{totals['split']}]")
        print(f"  frames: {totals['frames']:,}")
        print(f"  labels: {totals['labels']:,}")
        print(f"  labeled_frames (non-empty): {totals['labeled_frames']:,}")
        print(f"  non_labeled_frames (no .txt): {totals['non_labeled_frames']:,}")
        print(f"  empty_label_files: {totals['empty_label_files']:,}")
        print(f"  sequences: {totals['num_sequences']}")

        all_rows.extend(rows)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "split", "sequence", "frames", "label_files",
                    "labeled_frames", "non_labeled_frames", "empty_label_files"
                ],
            )
            writer.writeheader()
            for r in all_rows:
                writer.writerow(r)
        print(f"\nPer-sequence breakdown saved to: {args.csv}")

if __name__ == "__main__":
    main()
