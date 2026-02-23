#!/usr/bin/env python3
# split_and_check_spar.py

import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter

PREFIXES = {
    "scannet": "spar/scannet/",
    "scannetpp": "spar/scannetpp/",
    "structured3d": "spar/structured3d/",
}

GROUP_ORDER = ["scannet", "scannetpp", "structured3d", "unknown"]


def iter_items(annotation_path):
    annotation_path = Path(annotation_path)
    if annotation_path.suffix == ".jsonl":
        with annotation_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with annotation_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        yield from data


def get_paths(item):
    if isinstance(item.get("images"), list):
        return [p for p in item["images"] if isinstance(p, str)]
    if isinstance(item.get("video"), str):
        return [item["video"]]
    return []


def infer_group(item):
    paths = get_paths(item)
    for p in paths:
        for g, pref in PREFIXES.items():
            if p.startswith(pref):
                return g
    return "unknown"


def resolve_path(p, data_root):
    pp = Path(p)
    return pp if pp.is_absolute() else (data_root / pp)


def write_json(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", default="splits_out")
    parser.add_argument(
        "--mode",
        choices=["split", "check", "both"],
        default="both",
        help="split = only split json; check = only check missing; both = do both",
    )
    args = parser.parse_args()

    ann_path = Path(args.ann)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)

    split_items = defaultdict(list)
    items_total = Counter()
    items_missing = Counter()
    files_total = Counter()
    files_missing = Counter()
    missing_ids = defaultdict(list)

    for item in iter_items(ann_path):
        group = infer_group(item)

        # only collect split data if needed
        if args.mode in ["split", "both"]:
            split_items[group].append(item)

        if args.mode in ["check", "both"]:
            items_total[group] += 1
            paths = get_paths(item)

            any_miss = False
            for p in paths:
                files_total[group] += 1
                if not resolve_path(p, data_root).exists():
                    files_missing[group] += 1
                    any_miss = True

            if any_miss:
                items_missing[group] += 1
                missing_ids[group].append(item.get("id", "<no-id>"))

    # -------- SPLIT --------
    if args.mode in ["split", "both"]:
        print("\nSaving split jsons...")
        for g in GROUP_ORDER:
            out_path = out_dir / f"{ann_path.stem}_{g}.json"
            write_json(out_path, split_items.get(g, []))
            print(f"  {g:12s}: {len(split_items.get(g, []))} items")

    # -------- CHECK --------

    
    if args.mode in ["check", "both"]:
        print("\nMissing statistics:")
        print("-" * 72)
        print(f"{'group':12s} {'items':>8s} {'items_miss':>10s} {'files':>10s} {'missing':>10s} {'miss%':>8s}")

        for g in GROUP_ORDER:
            tot_f = files_total[g]
            miss_f = files_missing[g]
            miss_pct = (100.0 * miss_f / tot_f) if tot_f else 0.0
            print(f"{g:12s} {items_total[g]:8d} {items_missing[g]:10d} {tot_f:10d} {miss_f:10d} {miss_pct:7.2f}%")
    # Save missing ids
    if args.mode in ["check", "both"]:
        print("\nSaving missing id lists...")
        for g in GROUP_ORDER:
            if missing_ids[g]:
                out_file = f"missing_ids_{g}.txt"
                with open(out_file, "w") as f:
                    for _id in missing_ids[g]:
                        f.write(str(_id) + "\n")
                print(f"  {g:12s}: {len(missing_ids[g])} ids -> {out_file}")


if __name__ == "__main__":
    main()