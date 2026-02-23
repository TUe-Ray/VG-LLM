import json
from pathlib import Path
from collections import defaultdict

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

def save_missing_items(annotation_path, data_root):
    data_root = Path(data_root)

    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    missing_items = defaultdict(list)

    for item in data:
        paths = item.get("images", [])
        if not paths:
            continue

        group = infer_group_from_path(paths[0])

        any_missing = False
        for rel in paths:
            full = data_root / rel
            if not full.exists():
                any_missing = True
                break

        if any_missing:
            missing_items[group].append(item)

    # ✅ 只存 scannetpp + structured3d
    for g in ["scannetpp", "structured3d"]:
        if missing_items[g]:
            out_file = f"missing_{g}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(missing_items[g], f, ensure_ascii=False, indent=2)
            print(f"Saved {len(missing_items[g])} items to {out_file}")
        else:
            print(f"No missing items in {g}")

if __name__ == "__main__":
    save_missing_items(
        annotation_path="spar_sampled_4src_4k.json",
        data_root="/leonardo_scratch/large/userexternal/shuang00/spar_workspace/spar_data"
    )