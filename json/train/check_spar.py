import json
import os
from pathlib import Path
import random

def iter_items(annotation_path: str):
    p = Path(annotation_path)
    if p.suffix == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON file should contain a list of items")
        for x in data:
            yield x

def resolve_path(rel_or_abs: str, data_root: Path) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (data_root / p)

def check_spar_data(annotation_path="spar_234k.json",
                    data_root=None,
                    sample_size=None,
                    seed=42):
    """
    Check whether all media files referenced by SPAR/LLaVA-style annotations exist.

    data_root:
      - should be the directory that makes annotation paths resolvable.
      - For the example "spar/scannet/images/...", often data_root should be ".../data/media"
        so that data_root/"spar/..." exists.
    """

    if data_root is None:
        # 你可以改成你 repo 的實際位置，例如 Path("data/media")
        data_root = Path(os.environ.get("SCRATCH", ".")) / "spar_workspace" / "spar_data"
    else:
        data_root = Path(data_root)

    items = list(iter_items(annotation_path))

    if sample_size:
        random.seed(seed)
        items = random.sample(items, min(sample_size, len(items)))

    missing = []
    checked_files = 0

    print(f"Annotation: {annotation_path}")
    print(f"Data root:  {data_root}")
    print(f"Checking {len(items)} items")
    print("-" * 60)

    for item in items:
        # SPAR: images is a list
        if "images" in item and isinstance(item["images"], list):
            paths = item["images"]
        # LLaVA-Video style: "video" points to frames folder (optional)
        elif "video" in item and isinstance(item["video"], str):
            paths = [item["video"]]
        # fallback: common single fields
        else:
            maybe = item.get("file") or item.get("path") or item.get("image")
            paths = [maybe] if isinstance(maybe, str) else []

        if not paths:
            missing.append(("NO_PATH_FIELD", item.get("id", "<no-id>")))
            continue

        for rel in paths:
            checked_files += 1
            full = resolve_path(rel, data_root)
            if not full.exists():
                missing.append((str(rel), item.get("id", "<no-id>")))

    ok = (len(missing) == 0)
    print(f"Checked files: {checked_files}")
    print(f"Missing:       {len(missing)}")

    if missing:
        print("\nExamples of missing:")
        for x in missing[:10]:
            print(f"  - path={x[0]} (id={x[1]})")
        if len(missing) > 10:
            print(f"  ... and {len(missing)-10} more")

    return ok

if __name__ == "__main__":
    # 常見：把 data_root 指到 data/media
    # check_spar_data("data/train/spar_234k.json", data_root="data/media", sample_size=100)

    check_spar_data(sample_size=100)