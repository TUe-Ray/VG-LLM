#!/usr/bin/env python3
"""
Convert media files (images / video-frame directories) referenced in training
JSON annotation files into sharded HDF5 archives.

Storage layout
--------------
Each HDF5 shard file stores images as raw JPEG bytes in uint8 datasets.
The HDF5 key is the *relative media path* (forward-slash separated), so the
group hierarchy mirrors the filesystem directory structure.  For example:

  media/spar/scannet/images/scene0371_01/image_color/41.jpg
  → stored in shard (md5("spar/scannet/images/scene0371_01/image_color/41.jpg") % NUM_SHARDS)
  → key:  "spar/scannet/images/scene0371_01/image_color/41.jpg"
  → access: h5["spar/scannet/images/scene0371_01/image_color"]["41.jpg"][()]

Video frame directories (e.g. llava_hound/frames/rT3BozSwtkA/) are expanded
into individual frame files during conversion.  At training time the dataloader
can replace os.listdir(video_dir_path) with sorted(h5[video_dir_path].keys()).

Shard assignment
----------------
  shard_idx = int(md5(rel_path).hexdigest(), 16) % NUM_SHARDS

This is deterministic and stable.  The dataloader only needs to know NUM_SHARDS
to look up the right shard file at runtime.

A metadata.json is written to the output directory with num_shards and stats.
Partially completed runs can be resumed (already-present keys are skipped).

Usage
-----
python convert_to_hdf5.py \\
    --json_files /path/to/spar_234k.json,/path/to/llava_hound_64k.json \\
    --media_root  /leonardo_scratch/fast/EUHPC_D32_006/data/vgllm/media \\
    --output_dir  /leonardo_scratch/fast/EUHPC_D32_006/data/vgllm/hdf5 \\
    --num_shards  32 \\
    --num_workers 32 \\
    --quality     95

Stripe the output directory on Lustre before running:
    lfs setstripe -c 8 /path/to/hdf5_output_dir
"""

import argparse
import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_shard_idx(rel_path: str, num_shards: int) -> int:
    """Stable, deterministic shard assignment based on MD5 of the path string."""
    return int(hashlib.md5(rel_path.encode()).hexdigest(), 16) % num_shards


def encode_to_jpeg(full_path: str, quality: int) -> bytes:
    """Open any image, convert to RGB, re-encode as JPEG bytes."""
    with Image.open(full_path) as img:
        img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=False)
        return buf.getvalue()


def write_dataset(h5: h5py.File, rel_path: str, jpeg_bytes: bytes) -> None:
    """Write JPEG bytes into h5 at the hierarchical key given by rel_path."""
    parts = rel_path.split("/")
    grp = h5
    for part in parts[:-1]:
        grp = grp.require_group(part)
    leaf = parts[-1]
    if leaf not in grp:
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        grp.create_dataset(leaf, data=arr, dtype=np.uint8)


def key_exists(h5: h5py.File, rel_path: str) -> bool:
    """Check whether a dataset at rel_path already exists in h5."""
    try:
        return isinstance(h5[rel_path], h5py.Dataset)
    except KeyError:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 – collect all media paths from JSON annotation files
# ──────────────────────────────────────────────────────────────────────────────

def collect_paths_from_json(json_file: str):
    """
    Parse one JSON annotation file and return:
      image_paths  – set of relative image file paths
      video_dirs   – set of relative video frame directory paths
    """
    ext = json_file.rsplit(".", 1)[-1].lower()
    if ext == "jsonl":
        with open(json_file, encoding="utf-8") as fh:
            data = [json.loads(line) for line in fh if line.strip()]
    else:
        with open(json_file, encoding="utf-8") as fh:
            data = json.load(fh)

    image_paths: set = set()
    video_dirs: set = set()

    for sample in data:
        imgs = sample.get("images", sample.get("image"))
        if imgs is not None:
            if isinstance(imgs, str):
                imgs = [imgs]
            for p in imgs:
                image_paths.add(p.replace("\\", "/"))
        elif "video" in sample:
            video_dirs.add(sample["video"].replace("\\", "/"))

    return image_paths, video_dirs


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 – expand video directories into individual frame file paths
# ──────────────────────────────────────────────────────────────────────────────

def expand_video_dirs(video_dirs: set, media_root: str):
    """
    For each video directory, list files (sorted) and return all relative
    frame paths.  Directories that don't exist on disk are logged and skipped.
    """
    frame_paths: set = set()
    missing = []

    for video_dir in sorted(video_dirs):
        full_dir = os.path.join(media_root, video_dir)
        if not os.path.isdir(full_dir):
            missing.append(video_dir)
            continue
        for fname in os.listdir(full_dir):
            if os.path.isfile(os.path.join(full_dir, fname)):
                frame_paths.add(f"{video_dir}/{fname}")

    if missing:
        log.warning(f"{len(missing)} video directories not found on disk (first 3: {missing[:3]})")

    return frame_paths


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 – per-shard worker (each worker owns one shard file)
# ──────────────────────────────────────────────────────────────────────────────

def build_one_shard(args):
    """
    Worker function: opens/creates one HDF5 shard and writes all assigned paths.
    Returns (shard_idx, n_written, n_skipped, n_errors, error_list).
    """
    shard_idx, paths, media_root, output_dir, quality = args

    shard_path = os.path.join(output_dir, f"shard_{shard_idx:04d}.h5")
    mode = "a" if os.path.exists(shard_path) else "w"

    n_written = 0
    n_skipped = 0
    errors = []

    try:
        with h5py.File(shard_path, mode) as h5:
            for rel_path in paths:
                # Resume support: skip if already present
                if key_exists(h5, rel_path):
                    n_skipped += 1
                    continue

                full_path = os.path.join(media_root, rel_path)
                try:
                    jpeg_bytes = encode_to_jpeg(full_path, quality)
                    write_dataset(h5, rel_path, jpeg_bytes)
                    n_written += 1
                except FileNotFoundError:
                    errors.append((rel_path, "FileNotFoundError"))
                except Exception as exc:
                    errors.append((rel_path, str(exc)))
    except Exception as exc:
        errors.append((f"__shard_{shard_idx}__", f"HDF5 open error: {exc}"))

    return shard_idx, n_written, n_skipped, errors


# ──────────────────────────────────────────────────────────────────────────────
# Main orchestration
# ──────────────────────────────────────────────────────────────────────────────

def build_hdf5(
    json_files: list,
    media_root: str,
    output_dir: str,
    num_shards: int,
    num_workers: int,
    quality: int,
    dry_run: bool,
):
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # ── Phase 1: collect paths from all annotation files ──────────────────────
    all_image_paths: set = set()
    all_video_dirs:  set = set()

    for jf in json_files:
        log.info(f"Scanning annotation file: {jf}")
        imgs, vids = collect_paths_from_json(jf)
        log.info(f"  → {len(imgs):,} unique image paths, {len(vids):,} video dirs")
        all_image_paths |= imgs
        all_video_dirs  |= vids

    # ── Phase 2: expand video directories ─────────────────────────────────────
    if all_video_dirs:
        log.info(f"Expanding {len(all_video_dirs):,} video frame directories …")
        frame_paths = expand_video_dirs(all_video_dirs, media_root)
        log.info(f"  → {len(frame_paths):,} individual frame files found")
    else:
        frame_paths = set()

    all_paths = all_image_paths | frame_paths
    log.info(f"Total unique media files to convert: {len(all_paths):,}")

    if dry_run:
        log.info("Dry-run mode: no files written.")
        return

    # ── Phase 3: assign paths to shards ───────────────────────────────────────
    shard_bins: dict = defaultdict(list)
    for rel_path in all_paths:
        shard_bins[get_shard_idx(rel_path, num_shards)].append(rel_path)

    shard_sizes = [len(shard_bins[i]) for i in range(num_shards)]
    log.info(f"Shard size – min: {min(shard_sizes):,}  max: {max(shard_sizes):,}  "
             f"avg: {sum(shard_sizes)//num_shards:,}")

    # Build job list: one job per shard
    jobs = [
        (i, shard_bins[i], media_root, output_dir, quality)
        for i in range(num_shards)
    ]

    # ── Phase 4: process shards in parallel ───────────────────────────────────
    n_workers = min(num_workers, num_shards)
    log.info(f"Launching {n_workers} parallel workers for {num_shards} shards …")

    total_written = 0
    total_skipped = 0
    all_errors = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(build_one_shard, job): job[0] for job in jobs}
        with tqdm(total=num_shards, desc="Shards", unit="shard") as pbar:
            for fut in as_completed(futures):
                shard_idx, n_w, n_s, errs = fut.result()
                total_written += n_w
                total_skipped += n_s
                all_errors.extend(errs)
                pbar.set_postfix(written=total_written, skipped=total_skipped, errors=len(all_errors))
                pbar.update(1)

    elapsed = time.time() - t0
    log.info(f"Conversion complete in {elapsed:.1f}s")
    log.info(f"  Written:  {total_written:,}")
    log.info(f"  Skipped (already existed): {total_skipped:,}")
    log.info(f"  Errors:   {len(all_errors):,}")

    # ── Write metadata ─────────────────────────────────────────────────────────
    metadata = {
        "num_shards": num_shards,
        "total_files": len(all_paths),
        "n_written": total_written,
        "n_skipped": total_skipped,
        "n_errors": len(all_errors),
        "quality": quality,
        "json_files": json_files,
        "media_root": media_root,
        "elapsed_seconds": round(elapsed, 1),
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    log.info(f"Metadata written to {meta_path}")

    if all_errors:
        err_path = os.path.join(output_dir, "errors.json")
        with open(err_path, "w") as fh:
            json.dump(all_errors, fh, indent=2)
        log.warning(f"Error list ({len(all_errors)} entries) written to {err_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert media files to sharded HDF5 archives for fast HPC training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--json_files",
        required=True,
        help="Comma-separated list of training annotation JSON/JSONL files",
    )
    parser.add_argument(
        "--media_root",
        required=True,
        help="Root directory that contains all media files (images / video frames)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory where HDF5 shard files will be written",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=32,
        help="Number of HDF5 shard files.  Recommend 32–64 for typical HPC Lustre setups",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of parallel worker processes (one worker = one shard at a time)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG re-encoding quality (0–100).  95 is virtually lossless",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Scan annotation files and report stats without writing any HDF5 data",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_hdf5(
        json_files=[p.strip() for p in args.json_files.split(",")],
        media_root=args.media_root,
        output_dir=args.output_dir,
        num_shards=args.num_shards,
        num_workers=args.num_workers,
        quality=args.quality,
        dry_run=args.dry_run,
    )
