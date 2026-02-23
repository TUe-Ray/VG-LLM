import json
import random
from pathlib import Path
from collections import defaultdict

GROUP_PREFIXES = {
    "scannet": "spar/scannet/",
    "scannetpp": "spar/scannetpp/",
    "structured3d": "spar/structured3d/",
    "rxr": "rxr/",
}

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

def get_paths(item):
    """Return a list of media paths from an item."""
    if "images" in item and isinstance(item["images"], list):
        return [p for p in item["images"] if isinstance(p, str)]
    if "video" in item and isinstance(item["video"], str):
        return [item["video"]]
    # fallback common single fields
    for k in ("image", "file", "path"):
        if isinstance(item.get(k), str):
            return [item[k]]
    return []

def infer_group(item):
    """Infer which group this item belongs to, by media path prefixes."""
    paths = get_paths(item)
    for p in paths:
        for g, pref in GROUP_PREFIXES.items():
            if p.startswith(pref):
                return g
    # 如果你的 rxr 路徑不是 "rxr/..."，可以在這裡補規則
    return "unknown"

def stratified_sample(items, per_group=None, total=None, seed=42):
    """
    per_group: dict like {"scannet": 1000, "rxr": 500, ...}
    total: if set, sample total size proportional to group sizes (excluding unknown by default)
    """
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for it in items:
        buckets[infer_group(it)].append(it)

    # Print counts
    print("Counts by group:")
    for g in ["scannet", "scannetpp", "structured3d", "rxr", "unknown"]:
        print(f"  {g:12s}: {len(buckets.get(g, []))}")

    sampled = []

    if per_group:
        for g, k in per_group.items():
            pool = buckets.get(g, [])
            if not pool:
                print(f"[WARN] group '{g}' has 0 items; skip")
                continue
            kk = min(k, len(pool))
            sampled.extend(rng.sample(pool, kk))
        return sampled

    if total is not None:
        # proportional sampling across known groups
        known_groups = ["scannet", "scannetpp", "structured3d", "rxr"]
        pools = {g: buckets.get(g, []) for g in known_groups}
        sizes = {g: len(pools[g]) for g in known_groups}
        s_sum = sum(sizes.values())
        if s_sum == 0:
            print("[ERROR] No known-group items found.")
            return []

        # initial allocation
        alloc = {g: int(total * sizes[g] / s_sum) for g in known_groups}
        # fix rounding to match total
        remainder = total - sum(alloc.values())
        # distribute remainder to largest pools
        for g in sorted(known_groups, key=lambda x: sizes[x], reverse=True):
            if remainder <= 0:
                break
            alloc[g] += 1
            remainder -= 1

        for g in known_groups:
            pool = pools[g]
            if pool and alloc[g] > 0:
                sampled.extend(rng.sample(pool, min(alloc[g], len(pool))))
        return sampled

    raise ValueError("Provide either per_group or total")

def write_items(out_path, items):
    out_path = Path(out_path)
    if out_path.suffix == ".jsonl":
        with out_path.open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    else:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)

if __name__ == "__main__":
    # === 修改這裡 ===
    ann = "spar_234k.json"   # 或 spar_7m.jsonl / 其他
    out = "spar_sampled_4src_4k.json"  # .json or .jsonl
    seed = 42

    # 方式 A：每個來源各抽 N 筆（最常用）
    per_group = {
        "scannet": 1000,
        "scannetpp": 1000,
        "structured3d": 1000,
        "rxr": 1000,
    }

    # 方式 B：總數 total，按各來源比例抽（用這行就把 per_group 註解掉）
    # total = 4000

    items = list(iter_items(ann))
    sampled = stratified_sample(items, per_group=per_group, seed=seed)
    write_items(out, sampled)

    print(f"\nWrote sampled set: {out} (n={len(sampled)})")