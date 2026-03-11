import os
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64
from collections.abc import Sequence
import h5py
from torch.utils.data import get_worker_info
import gzip
from collections import defaultdict

# ── Per-worker HDF5 handle cache ──────────────────────────────────────────────
# Lives at module level so it is local to each forked DataLoader worker process.
# Handles are opened lazily and kept open for the lifetime of the worker.
_worker_h5_handles: dict = {}

_profile_state = defaultdict(
    lambda: {
        "count": 0,

        # dataset
        "total": 0.0,
        "sample_copy": 0.0,
        "video_to_images": 0.0,
        "list_frames": 0.0,
        "open_images": 0.0,
        "path_to_pil": 0.0,
        "draw_marks": 0.0,
        "prepare_inputs": 0.0,
        "grid_merge": 0.0,
        "text_preprocess": 0.0,
        "rope": 0.0,
        "final_pack": 0.0,
        "unaccounted": 0.0,

        # new: fair decode metrics
        "video_hdf5_decode_time": 0.0,
        "video_hdf5_decode_images": 0,
        "image_hdf5_decode_time": 0.0,
        "image_hdf5_decode_images": 0,

        # collator
        "collator_count": 0,
        "collator_total": 0.0,
        "pad_ids": 0.0,
        "collect_images": 0.0,
        "cat_images": 0.0,
        "cat_videos": 0.0,
        "geometry_inputs": 0.0,
    }
)

def _profile_enabled():
    return os.environ.get("PROFILE_DATA_LOADING", "0") == "1"

def _profile_every():
    return int(os.environ.get("PROFILE_EVERY_N", "100"))

def _profile_key():
    wi = get_worker_info()
    worker_id = wi.id if wi is not None else -1
    rank = os.environ.get("RANK", "0")
    return rank, worker_id

def _profile_should_print():
    if not _profile_enabled():
        return False
    if os.environ.get("PROFILE_ONLY_RANK0", "1") == "1" and os.environ.get("RANK", "0") != "0":
        return False
    wi = get_worker_info()
    return wi is None or wi.id == 0

def _profile_update(stats: dict, is_collator: bool = False):
    if not _profile_enabled():
        return

    key = _profile_key()
    state = _profile_state[key]

    if is_collator:
        state["collator_count"] += 1
    else:
        state["count"] += 1

    for k, v in stats.items():
        state[k] += float(v)

    if not _profile_should_print():
        return

    if not is_collator:
        n = state["count"]
        if n > 0 and n % _profile_every() == 0:
            video_ms_per_image = (
                1000.0 * state["video_hdf5_decode_time"] / state["video_hdf5_decode_images"]
                if state["video_hdf5_decode_images"] > 0 else 0.0
            )
            image_ms_per_image = (
                1000.0 * state["image_hdf5_decode_time"] / state["image_hdf5_decode_images"]
                if state["image_hdf5_decode_images"] > 0 else 0.0
            )

            print(
                "[DATA PROF] "
                f"rank={key[0]} worker={key[1]} n={n} "
                f"total={state['total']/n:.4f}s "
                f"sample_copy={state['sample_copy']/n:.4f}s "
                f"video_to_images={state['video_to_images']/n:.4f}s "
                f"list_frames={state['list_frames']/n:.4f}s "
                f"open_images={state['open_images']/n:.4f}s "
                f"path_to_pil={state['path_to_pil']/n:.4f}s "
                f"draw_marks={state['draw_marks']/n:.4f}s "
                f"prepare_inputs={state['prepare_inputs']/n:.4f}s "
                f"grid_merge={state['grid_merge']/n:.4f}s "
                f"text_preprocess={state['text_preprocess']/n:.4f}s "
                f"rope={state['rope']/n:.4f}s "
                f"final_pack={state['final_pack']/n:.4f}s "
                f"unaccounted={state['unaccounted']/n:.4f}s "
                f"video_hdf5_decode_ms_per_image={video_ms_per_image:.3f} "
                f"image_hdf5_decode_ms_per_image={image_ms_per_image:.3f}"
            )

    else:
        n = state["collator_count"]
        if n > 0 and n % _profile_every() == 0:
            print(
                "[COLLATE PROF] "
                f"rank={key[0]} worker={key[1]} n={n} "
                f"collator_total={state['collator_total']/n:.4f}s "
                f"pad_ids={state['pad_ids']/n:.4f}s "
                f"collect_images={state['collect_images']/n:.4f}s "
                f"cat_images={state['cat_images']/n:.4f}s "
                f"cat_videos={state['cat_videos']/n:.4f}s "
                f"geometry_inputs={state['geometry_inputs']/n:.4f}s"
            )

def _get_h5_handle(shard_path: str) -> h5py.File:
    """Return a cached read-only HDF5 file handle for the given shard."""
    if shard_path not in _worker_h5_handles:
        _worker_h5_handles[shard_path] = h5py.File(shard_path, "r", swmr=True)
    return _worker_h5_handles[shard_path]


def _get_shard_path(hdf5_dir: str, rel_path: str, num_shards: int) -> str:
    """Return the shard file that contains *rel_path* (same formula as converter)."""
    idx = int(hashlib.md5(rel_path.encode()).hexdigest(), 16) % num_shards
    return os.path.join(hdf5_dir, f"shard_{idx:04d}.h5")
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader
import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2
from .utils import prepare_image_inputs

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path, max_samples: int=-1):
    with open(path, "r") as f:
        # return [json.loads(line) for line in f]
        ret = []
        for line in f:
            ret.append(json.loads(line))
            if max_samples !=-1 and len(ret) >= max_samples:
                break
    return ret


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw: List = [],
    visual_type: str = "image",
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."
    if visual_type not in ["image", "video"]:
        raise ValueError("visual_type must be either 'image' or 'video'")

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                visual_tag = f"<{visual_type}>"
                if visual_tag in content:
                    parts = content.split(visual_tag)
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|{visual_type}_pad|>"
                            * grid_thw[visual_replicate_index]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")  #["llava_hound", "SPAR_234"]
        dataset_list = data_list(dataset) # 解析抽樣比例（例如 %30 → 0.3）/把 config + sampling_rate 組成 list
        print(f"Loading datasets: {dataset_list}")
        #表示「N 個 vision tokens 對應的原始像素面積」。控制 vision token 數量
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        #CAREFUL in train_qwen: data_args.model_type = "qwen2.5vl"
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"], max_samples=data_args.max_samples)
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                ann["data_path"] = data["data_path"]
                ann["tag"] = data["tag"]
            list_data_dict += annotations #=ist_data_dict.extend(annotations) ,把 annotations 裡的每一筆 sample加進 list_data_dict

        print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels
        self.use_hdf5 = getattr(data_args, "use_hdf5", False)
        self.hdf5_path = getattr(data_args, "hdf5_path", None)
        self.hdf5_num_shards = getattr(data_args, "hdf5_num_shards", 32)
        self.video_frames_index = None

        if self.use_hdf5:
            if self.hdf5_path is None or not os.path.isdir(self.hdf5_path):
                raise ValueError(
                    f"use_hdf5=True but hdf5_path='{self.hdf5_path}' is not a directory. "
                    "Pass the directory that contains the shard_XXXX.h5 files."
                )

            index_path = os.path.join(self.hdf5_path, "video_frames_index.json.gz")
            if os.path.isfile(index_path):
                with gzip.open(index_path, "rt", encoding="utf-8") as fh:
                    self.video_frames_index = json.load(fh)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            if "image" in sample:
                image_num = len(sample["image"])
            elif "images" in sample:
                image_num = len(sample["images"])
            elif "video" in sample:
                image_num = getattr(self.data_args, "video_max_frames", 8)
            else:
                image_num = 0
            length_list.append(image_num * 252 + cur_len)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            if "image" in sample:
                image_num = len(sample["image"])
            elif "images" in sample:
                image_num = len(sample["images"])
            elif "video" in sample:
                image_num = getattr(self.data_args, "video_max_frames", 8)
            else:
                image_num = 0
            cur_len += image_num*252
            tag = sample.get("tag", "2d")
            cur_len = -cur_len if tag == "2d" else cur_len
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))


    # ── HDF5 helpers ─────────────────────────────────────────────────────────

    def _hdf5_open_image(self, rel_path: str) -> Image.Image:
        """Open a single image from the appropriate HDF5 shard (read-only)."""
        shard_path = _get_shard_path(self.hdf5_path, rel_path, self.hdf5_num_shards)
        h5 = _get_h5_handle(shard_path)
        jpeg_bytes = h5[rel_path][()].tobytes()
        return Image.open(BytesIO(jpeg_bytes)).convert("RGB")

    def _hdf5_open_image_from_shard(self, shard_idx: int, rel_path: str) -> Image.Image:
        shard_path = os.path.join(self.hdf5_path, f"shard_{shard_idx:04d}.h5")
        h5 = _get_h5_handle(shard_path)
        jpeg_bytes = h5[rel_path][()].tobytes()
        return Image.open(BytesIO(jpeg_bytes)).convert("RGB")

    def _hdf5_list_video_frames(self, video_dir: str):
        """
        Return:
        frame_rel_paths: sorted full relative frame paths
        shard_idx: int | None
        """
        video_dir = video_dir.replace("\\", "/")

        # Fast path: use sidecar index
        if self.video_frames_index is not None and video_dir in self.video_frames_index:
            item = self.video_frames_index[video_dir]
            frame_rel_paths = [f"{video_dir}/{fname}" for fname in item["frames"]]
            return frame_rel_paths, item["shard_idx"]

        # Fallback path for old HDF5 layout: scan all shards
        frame_rel_paths = []
        for shard_idx in range(self.hdf5_num_shards):
            shard_path = os.path.join(self.hdf5_path, f"shard_{shard_idx:04d}.h5")
            h5 = _get_h5_handle(shard_path)
            try:
                grp = h5[video_dir]
                for fname in grp.keys():
                    frame_rel_paths.append(f"{video_dir}/{fname}")
            except KeyError:
                pass
        return sorted(frame_rel_paths), None

    # ─────────────────────────────────────────────────────────────────────────

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw
    
    def draw_visual_marks(self, images, spar_info):

        if spar_info is None:
            return
        info = json.loads(spar_info)
        task_type = info["type"]
        from .draw_marker import DRAW_FUNCTIONS
        draw_fn = DRAW_FUNCTIONS[task_type]
        if len(images) == 1:
            draw_fn(images[0], info)
        else:
            draw_fn(images, info)
        # for j, img in enumerate(images):
        #     # write to local
        #     img.save(f"images/img_{j}.jpg", format="JPEG")

    def process_video(self, video_file):
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(
            images=None, videos=video, return_tensors="pt"
        )
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            self.data_args.image_processor.temporal_patch_size / fps
        ] * len(grid_thw)
        return video_tensor, grid_thw, second_per_grid_ts
    

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e
    
    def read_video_images(self, source, prof=None):
        # read video images from the source
        assert isinstance(source["video"], str), "video should be a string"

        if prof is None:
            prof = {}

        def get_frame_indices(total_frames, fps=1):
            video_length = total_frames / fps
            interval = getattr(self.data_args, "base_interval", 2)
            num_frames_to_sample = round(video_length / interval)
            video_min_frames = getattr(self.data_args, "video_min_frames", 4)
            video_max_frames = getattr(self.data_args, "video_max_frames", 8)
            target_frames = min(
                max(num_frames_to_sample, video_min_frames), video_max_frames
            )
            frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
            frame_idx = np.unique(frame_idx)
            return frame_idx

        # ── HDF5 path ────────────────────────────────────────────────────────
        if self.use_hdf5:
            video_dir = source["video"].replace("\\", "/")

            t0 = time.perf_counter()
            frame_rel_paths, shard_idx = self._hdf5_list_video_frames(video_dir)
            prof["list_frames"] = prof.get("list_frames", 0.0) + (time.perf_counter() - t0)

            if not frame_rel_paths:
                print(f"No frames found in HDF5 for video dir: {video_dir}")
                raise FileNotFoundError

            frame_idx = get_frame_indices(len(frame_rel_paths), fps=1)
            selected = [frame_rel_paths[i] for i in frame_idx]

            t1 = time.perf_counter()
            t1 = time.perf_counter()
            if shard_idx is not None:
                images = [self._hdf5_open_image_from_shard(shard_idx, p) for p in selected]
            else:
                images = [self._hdf5_open_image(p) for p in selected]
            decode_time = time.perf_counter() - t1

            prof["open_images"] = prof.get("open_images", 0.0) + decode_time
            prof["video_hdf5_decode_time"] = prof.get("video_hdf5_decode_time", 0.0) + decode_time
            prof["video_hdf5_decode_images"] = prof.get("video_hdf5_decode_images", 0) + len(selected)

            return images

        # ── Original filesystem path ──────────────────────────────────────────
        video_file = os.path.join(source["data_path"], source["video"])
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
            raise FileNotFoundError

        # check whether video_file is a directory
        if os.path.isdir(video_file):
            frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))]
            frame_files.sort()
            frame_idx = get_frame_indices(len(frame_files), 1)
            images = [frame_files[i] for i in frame_idx]
            images = [Image.open(frame).convert("RGB") for frame in images]
        elif any([video_file.endswith(ext) for ext in [".mp4", ".avi", ".mov"]]):
            vr = VideoReader(video_file, num_threads=4)
            total_frames = len(vr)
            avg_fps = vr.get_avg_fps()
            frame_idx = get_frame_indices(total_frames, avg_fps)
            video = vr.get_batch(frame_idx).asnumpy()
            images = [Image.fromarray(frame).convert("RGB") for frame in video]
        return images

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        """
        取出第 i 筆 annotation
        self.list_data_dict 是你在 __init__ 時把所有 dataset 的 annotations 合併後的 list。
        所以 sources 一開始是一個 dict，像：
        {
        "conversations": [...],
        "image": [...]/"video": "...",
        "data_path": "...",
        "tag": "2d"
        }

        
        下面這版有兩個目的：
        不再修改 self.list_data_dict
        保持你現在的行為邏輯不變
        
        """
        # sources = self.list_data_dict[i]
        # #把單筆 dict 包成 list
        # if isinstance(i, int):
        #     sources = [sources]
        # assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        # video = None

        prof = {
            "total": 0.0,
            "sample_copy": 0.0,
            "video_to_images": 0.0,
            "list_frames": 0.0,
            "open_images": 0.0,
            "path_to_pil": 0.0,
            "draw_marks": 0.0,
            "prepare_inputs": 0.0,
            "grid_merge": 0.0,
            "text_preprocess": 0.0,
            "rope": 0.0,
            "final_pack": 0.0,
            "unaccounted": 0.0,
            "video_hdf5_decode_time": 0.0,
            "video_hdf5_decode_images": 0,
            "image_hdf5_decode_time": 0.0,
            "image_hdf5_decode_images": 0,
        }

        t_item0 = time.perf_counter()

        t0 = time.perf_counter()
        sample = copy.deepcopy(self.list_data_dict[i])
        prof["sample_copy"] += time.perf_counter() - t0

        sources = [sample]
        video = None
        # Convert video-dir sample into image sequence lazily, but do NOT mutate original dataset
        if "video" in sample:
            t0 = time.perf_counter()
            sample["images"] = self.read_video_images(sample, prof=prof)
            prof["video_to_images"] += time.perf_counter() - t0
            num_image = len(sample["images"])
            sample["conversations"][0]["value"] = sample["conversations"][0]["value"].replace(
                DEFAULT_VIDEO_TOKEN, "".join([DEFAULT_IMAGE_TOKEN] * num_image)
            )
            sample.pop("video", None)

        # Replace "<image>\n" with "<image>"
        sample["conversations"][0]["value"] = sample["conversations"][0]["value"].replace(
            f"{DEFAULT_IMAGE_TOKEN}\n", DEFAULT_IMAGE_TOKEN
        )

        # Rename images -> image for downstream logic
        if "images" in sample:
            sample["image"] = sample["images"]

        if "image" in sample:
            image_folder = sample["data_path"]
            image_file = sample["image"]

            if isinstance(image_file, List):
                if isinstance(image_file[0], str):
                    t0 = time.perf_counter()
                    if self.use_hdf5:
                        image_file = [
                            self._hdf5_open_image(file.replace("\\", "/"))
                            for file in image_file
                        ]
                        decode_time = time.perf_counter() - t0
                        prof["path_to_pil"] += decode_time
                        prof["image_hdf5_decode_time"] += decode_time
                        prof["image_hdf5_decode_images"] += len(image_file)
                    else:
                        image_file = [
                            os.path.join(image_folder, file) for file in image_file
                        ]
                        image_file = [Image.open(img).convert("RGB") for img in image_file]
                        prof["path_to_pil"] += time.perf_counter() - t0
                elif isinstance(image_file[0], Image.Image):
                    pass
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

            t_marks = time.perf_counter()
            self.draw_visual_marks(image_file, sample.get("spar_info", None))
            prof["draw_marks"] += time.perf_counter() - t_marks

            image, grid_thw, geometry_encoder_inputs = [], [], []
            t_prepare = time.perf_counter()
            for file in image_file:
                ret = prepare_image_inputs(file, self.data_args.image_processor)
                image.append(ret["pixel_values"])
                geometry_encoder_inputs.append(ret["geometry_encoder_inputs"])
                grid_thw.append(ret["image_grid_thw"])
            prof["prepare_inputs"] += time.perf_counter() - t_prepare

            t0 = time.perf_counter()
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in copy.deepcopy(grid_thw)
            ]
            prof["grid_merge"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            conv_sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                conv_sources, self.tokenizer, grid_thw=grid_thw_merged, visual_type="image"
            )
            prof["text_preprocess"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                torch.stack(grid_thw, dim=0),
            )
            prof["rope"] += time.perf_counter() - t0

        elif "video" in sample:
            # This branch is kept for true video-file inputs (.mp4/.avi/.mov)
            video_file = sample["video"]
            video_folder = sample["data_path"]

            if isinstance(video_file, List):
                if len(video_file) > 1:
                    video_file = [os.path.join(video_folder, file) for file in video_file]
                    results = [self.process_video(file) for file in video_file]
                    video, grid_thw, second_per_grid_ts = zip(*results)
                else:
                    video_file = os.path.join(video_folder, video_file[0])
                    video, grid_thw, second_per_grid_ts = self.process_video(video_file)
                    video = [video]
            else:
                video_file = os.path.join(video_folder, video_file)
                video, grid_thw, second_per_grid_ts = self.process_video(video_file)
                video = [video]

            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]

            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]
            t0 = time.perf_counter()
            conv_sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                conv_sources, self.tokenizer, grid_thw=grid_thw_merged, visual_type="video"
            )
            prof["text_preprocess"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                video_grid_thw=torch.stack(grid_thw, dim=0),
                second_per_grid_ts=second_per_grid_ts,
            )
            prof["rope"] += time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            conv_sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                conv_sources, self.tokenizer, grid_thw=None
            )
            prof["text_preprocess"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            position_ids = (
                torch.arange(0, data_dict["input_ids"].size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)
            )
            prof["rope"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0],
                position_ids=position_ids,
            )

        if "image" in sample:
            data_dict["pixel_values"] = image
            data_dict["image_grid_thw"] = grid_thw
            if getattr(self.data_args, "use_geometry_encoder", False):
                data_dict["geometry_encoder_inputs"] = geometry_encoder_inputs
        elif "video" in sample:
            data_dict["pixel_values_videos"] = video
            data_dict["video_grid_thw"] = grid_thw

        data_dict["tag"] = sample.get("tag", "2d")
        prof["final_pack"] += time.perf_counter() - t0
        prof["total"] += time.perf_counter() - t_item0

        measured = (
            prof["sample_copy"]
            + prof["video_to_images"]
            + prof["list_frames"]
            + prof["open_images"]
            + prof["path_to_pil"]
            + prof["draw_marks"]
            + prof["prepare_inputs"]
            + prof["grid_merge"]
            + prof["text_preprocess"]
            + prof["rope"]
            + prof["final_pack"]
        )
        prof["unaccounted"] += max(0.0, prof["total"] - measured)

        _profile_update(prof)
        return data_dict
        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        coll_prof = {
            "collator_total": 0.0,
            "pad_ids": 0.0,
            "collect_images": 0.0,
            "cat_images": 0.0,
            "cat_videos": 0.0,
            "geometry_inputs": 0.0,
        }
        t_coll0 = time.perf_counter()
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        t0 = time.perf_counter()
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        coll_prof["pad_ids"] += time.perf_counter() - t0
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        t0 = time.perf_counter()
        images = list(
            itertools.chain(
                *(
                    instance["pixel_values"]
                    for instance in instances
                    if "pixel_values" in instance
                )
            )
        )
        videos = list(
            itertools.chain(
                *(
                    instance["pixel_values_videos"]
                    for instance in instances
                    if "pixel_values_videos" in instance
                )
            )
        )
        coll_prof["collect_images"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = list(
                itertools.chain(
                    *(
                        instance["image_grid_thw"]
                        for instance in instances
                        if "image_grid_thw" in instance
                    )
                )
            )
            grid_thw = torch.stack(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None
        coll_prof["cat_images"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = list(
                itertools.chain(
                    *(
                        instance["video_grid_thw"]
                        for instance in instances
                        if "video_grid_thw" in instance
                    )
                )
            )
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None
        coll_prof["cat_videos"] += time.perf_counter() - t0
        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
                
        # assume all data in a batch has geometry_encoder_inputs
        t0 = time.perf_counter()
        if "geometry_encoder_inputs" in instances[0]:
            geometry_encoder_inputs = [torch.stack(instance["geometry_encoder_inputs"]) for instance in instances]
            batch["geometry_encoder_inputs"] = geometry_encoder_inputs
            assert len(set([instance["tag"] for instance in instances])) == 1, "all data in a batch should have the same tag"
            batch["tag"] = instances[0]["tag"]
        coll_prof["geometry_inputs"] += time.perf_counter() - t0
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )

        seq_lens = torch.tensor(
            [0] + [len(seq) for seq in input_ids], dtype=torch.int32
        )
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=0)
        labels = torch.cat(labels, dim=0)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids.unsqueeze(0),
            labels=labels.unsqueeze(0),
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            itertools.chain(
                *(
                    instance["pixel_values"]
                    for instance in instances
                    if "pixel_values" in instance
                )
            )
        )
        videos = list(
            itertools.chain(
                *(
                    instance["pixel_values_videos"]
                    for instance in instances
                    if "pixel_values_videos" in instance
                )
            )
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = list(
                itertools.chain(
                    *(
                        instance["image_grid_thw"]
                        for instance in instances
                        if "image_grid_thw" in instance
                    )
                )
            )
            grid_thw = torch.stack(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None
        
        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = list(
                itertools.chain(
                    *(
                        instance["video_grid_thw"]
                        for instance in instances
                        if "video_grid_thw" in instance
                    )
                )
            )
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

                
        # assume all data in a batch has geometry_encoder_inputs
        if "geometry_encoder_inputs" in instances[0]:
            raise NotImplementedError("FlattenedDataCollatorForSupervisedDataset does not support geometry_encoder_inputs")

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    if data_args.data_flatten:
        data_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass