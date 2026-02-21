#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import math


def split_json(input_file, num_splits, output_prefix=None):
    input_file = Path(input_file)

    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    chunk_size = math.ceil(total / num_splits)

    print(f"Total items: {total}")
    print(f"Splitting into {num_splits} chunks (~{chunk_size} each)\n")

    if output_prefix is None:
        output_prefix = input_file.stem

    for i in range(num_splits):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total)
        chunk = data[start:end]

        out_file = input_file.parent / f"{output_prefix}_part{i+1}.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False)

        print(f"Saved {len(chunk)} items -> {out_file}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--prefix", default=None)

    args = parser.parse_args()

    split_json(args.input, args.splits, args.prefix)