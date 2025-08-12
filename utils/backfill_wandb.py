#!/usr/bin/env python3
"""
Backfill checkpoints (and optional plots/eval_results/logs) from outputs/<experiment>/*
into Weights & Biases Artifacts.

Usage:
  python backfill_wandb.py \
    --outputs-dir /vol/bitbucket/si324/rf-detr-wildfire/outputs \
    --project wildfire-detection \
    --entity your_wandb_entity \
    --include plots,eval_results,logs \
    --all-ckpts \
    # add --dry-run to preview

Notes:
- Requires: pip install wandb
- Auth once: wandb login  (or set WANDB_API_KEY)
- Offline-friendly: export WANDB_MODE=offline, then `wandb sync` later.
"""

from __future__ import annotations
import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

# ---------- discovery ----------

def list_experiment_dirs(outputs_dir: Path) -> List[Path]:
    if not outputs_dir.exists():
        return []
    return sorted([p for p in outputs_dir.iterdir() if p.is_dir()])

def list_experiment_checkpoints(exp_dir: Path) -> List[Path]:
    ckpt_dir = exp_dir / "checkpoints"
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        return []
    ckpts: List[Path] = []
    for ext in ("*.pt", "*.pth", "*.ckpt"):
        ckpts.extend(sorted(ckpt_dir.glob(ext)))
    return ckpts

# ---------- selection / hashing ----------

def pick_topk(ckpts: List[Path], upload_all: bool, k: int = 2) -> List[Path]:
    if upload_all:
        return ckpts
    # prefer tagged, then newest
    tags = ("best", "last", "ema")
    tagged = [p for p in ckpts if any(t in p.name.lower() for t in tags)]
    uniq_tagged = list(dict.fromkeys(tagged))
    remainder = [p for p in ckpts if p not in uniq_tagged]
    picked = uniq_tagged + sorted(remainder, key=lambda x: x.stat().st_mtime, reverse=True)
    return picked[:k] if picked else []

def sha_of_paths(paths: Iterable[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        st = p.stat()
        h.update(p.name.encode())
        h.update(str(st.st_size).encode())
        h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()

def infer_aliases(ckpt_paths: Iterable[Path]) -> List[str]:
    aliases = ["latest"]
    names = " ".join(p.name.lower() for p in ckpt_paths)
    if "best" in names: aliases.append("best")
    if "ema" in names: aliases.append("ema")
    return list(dict.fromkeys(aliases))

# ---------- uploader ----------

def ensure_wandb() -> None:
    try:
        import wandb  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "wandb is required. Install with `pip install wandb` and login via `wandb login`."
        ) from exc

def artifact_path(project: str, entity: str | None, name: str, alias: str = "latest") -> str:
    # Formats: "entity/project/name:alias" or "project/name:alias"
    base = f"{entity}/{project}" if entity else project
    return f"{base}/{name}:{alias}"

def upload_dir_as_artifact(run, project: str, entity: str | None, exp_dir: Path,
                           subdir: str, dry_run: bool) -> int:
    import wandb
    src = exp_dir / subdir
    if not src.exists() or not src.is_dir():
        return 0
    art_name = f"{exp_dir.name}-{subdir}"
    if dry_run:
        print(f"[dry-run] would upload dir: {src} -> artifact: {art_name}")
        return 1
    art = wandb.Artifact(name=art_name, type=subdir.rstrip("s") or "artifact",
                         metadata={"experiment_name": exp_dir.name,
                                   "source_outputs_dir": str(exp_dir.parent)})
    art.add_dir(str(src))
    run.log_artifact(art)
    return 1

def backfill(outputs_dir: Path, project: str, entity: str | None,
             include_dirs: List[str], all_ckpts: bool, dry_run: bool) -> int:
    import wandb
    run = wandb.init(
        project=project,
        entity=entity,
        job_type="backfill",
        name=f"backfill-{int(time.time())}",
        config={"outputs_dir": str(outputs_dir),
                "include": include_dirs,
                "all_ckpts": all_ckpts},
        settings=wandb.Settings(start_method="thread")
    )
    api = wandb.Api()
    created = 0

    for exp_dir in list_experiment_dirs(outputs_dir):
        ckpts = list_experiment_checkpoints(exp_dir)
        if not ckpts:
            # still allow optional dirs to upload even if no ckpts
            for d in include_dirs:
                created += upload_dir_as_artifact(run, project, entity, exp_dir, d, dry_run)
            continue

        ckpts = pick_topk(ckpts, all_ckpts)
        digest = sha_of_paths(ckpts)
        art_name = f"{exp_dir.name}-checkpoints"

        # Idempotency: skip if the latest artifact has same digest
        skip = False
        try:
            existing = api.artifact(artifact_path(project, entity, art_name, "latest"))
            if existing and existing.metadata and existing.metadata.get("digest") == digest:
                print(f"Skip {exp_dir.name}: checkpoints up-to-date (digest match).")
                skip = True
        except Exception:
            pass  # not found → proceed

        if dry_run:
            print(f"[dry-run] {exp_dir.name}: {len(ckpts)} checkpoint(s)")
            for p in ckpts: print(f"  -> {p}")
            created += 1
        else:
            if not skip:
                print(f"Uploading {len(ckpts)} checkpoint(s) for {exp_dir.name}")
                art = wandb.Artifact(
                    name=art_name,
                    type="model",
                    metadata={
                        "experiment_name": exp_dir.name,
                        "num_checkpoints": len(ckpts),
                        "source_outputs_dir": str(outputs_dir),
                        "digest": digest,
                    },
                )
                for p in ckpts:
                    art.add_file(str(p), name=f"{exp_dir.name}/checkpoints/{p.name}")
                aliases = infer_aliases(ckpts)
                run.log_artifact(art, aliases=aliases)
                created += 1

        # Also upload extra dirs (plots/eval_results/logs) if requested
        for d in include_dirs:
            created += upload_dir_as_artifact(run, project, entity, exp_dir, d, dry_run)

    run.finish()
    return created

# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Backfill W&B Artifacts from outputs/")
    p.add_argument("--outputs-dir", type=Path, required=True,
                   help="Root outputs directory with experiment subfolders")
    p.add_argument("--project", type=str, required=True, help="W&B project name")
    p.add_argument("--entity", type=str, default=os.environ.get("WANDB_ENTITY"),
                   help="W&B entity (username or team). Omit to use default.")
    p.add_argument("--include", type=str, default="",
                   help="Comma-separated extra subdirs per experiment to upload (e.g. 'plots,eval_results,logs')")
    p.add_argument("--all-ckpts", action="store_true",
                   help="Upload all checkpoints (default: only top-2 / tagged)")
    p.add_argument("--dry-run", action="store_true", help="List actions without uploading")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    ensure_wandb()

    include_dirs = [s.strip() for s in args.include.split(",") if s.strip()]
    created = backfill(
        outputs_dir=args.outputs_dir,
        project=args.project,
        entity=args.entity,
        include_dirs=include_dirs,
        all_ckpts=args.all_ckpts,
        dry_run=args.dry_run,
    )
    print(f"Completed: created {created} artifact(s) in project '{args.project}'.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.", file=sys.stderr)
        sys.exit(130)
