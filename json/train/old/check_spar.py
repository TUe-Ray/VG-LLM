import json
import os
from pathlib import Path
from collections import defaultdict, Counter

GROUP_PREFIXES = {
    "scannet": "spar/scannet/",
    "scannetpp": "spar/scannetpp/",
    "structured3d": "spar/structured3d/",
    "rxr": "rxr/",
}

def infer_group_from_path(p: str) -> str:
    for g, pref in GROUP_PREFIXES.items():
        if p.startswith(pref):
            return g
    return "unknown"

def check_missing_by_source(annotation_path, data_root):
    data_root = Path(data_root)

    # load json list
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)

    # stats
    files_total = Counter()     # per group: total referenced files
    files_missing = Counter()   # per group: missing files
    items_total = Counter()     # per group: number of items
    items_any_missing = Counter()  # per group: items with >=1 missing file

    missing_examples = defaultdict(list)  # group -> list of (id, path)

    for item in data:
        item_id = item.get("id", "<no-id>")

        paths = item.get("images", [])
        if not isinstance(paths, list) or not paths:
            # 沒有 images 的項目也算到 unknown
            items_total["unknown"] += 1
            items_any_missing["unknown"] += 1
            continue

        # group 用第一個 path 判（你的資料很乾淨，會一致）
        g = infer_group_from_path(paths[0])
        items_total[g] += 1

        any_miss = False
        for rel in paths:
            if not isinstance(rel, str):
                continue
            files_total[g] += 1
            full = (Path(rel) if Path(rel).is_absolute() else data_root / rel)
            if not full.exists():
                files_missing[g] += 1
                any_miss = True
                if len(missing_examples[g]) < 10:
                    missing_examples[g].append((item_id, rel))

        if any_miss:
            items_any_missing[g] += 1

    # print report
    groups = ["scannet", "scannetpp", "structured3d", "rxr", "unknown"]
    print(f"Annotation: {annotation_path}")
    print(f"Data root:  {data_root}\n")

    print("Per-source file missing stats:")
    print("-" * 72)
    print(f"{'group':12s} {'items':>8s} {'items_miss':>10s} {'files':>10s} {'missing':>10s} {'miss%':>8s}")
    for g in groups:
        tot_f = files_total[g]
        miss_f = files_missing[g]
        miss_pct = (100.0 * miss_f / tot_f) if tot_f else 0.0
        print(f"{g:12s} {items_total[g]:8d} {items_any_missing[g]:10d} {tot_f:10d} {miss_f:10d} {miss_pct:7.2f}%")

    print("\nExamples (up to 10 missing paths per group):")
    for g in groups:
        if files_missing[g] == 0:
            continue
        print(f"\n[{g}]")
        for item_id, rel in missing_examples[g]:
            print(f"  - id={item_id} path={rel}")

if __name__ == "__main__":
    check_missing_by_source(
        annotation_path="spar_sampled_4src_4k.json",
        data_root="/leonardo_scratch/large/userexternal/shuang00/spar_workspace/spar_data"
    )