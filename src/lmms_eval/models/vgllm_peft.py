from typing import Optional, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from lmms_eval.api.registry import register_model
from lmms_eval.models.vgllm import VGLLM
from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT


@register_model("vgllm_peft")
class VGLLM_PEFT(VGLLM):
    """VGLLM evaluation wrapper for separately loaded Qwen base, VGGT, and PEFT adapters."""

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        geometry_encoder_path: Optional[str] = None,
        peft: Optional[str] = None,
        merge_peft: Optional[bool] = False,
        processor_pretrained: Optional[str] = None,
        use_geometry_encoder: Optional[bool] = True,
        geometry_encoder_type: Optional[str] = "vggt",
        feature_fusion_method: Optional[str] = "add",
        fusion_num_layers: Optional[int] = 1,
        geometry_merger_type: Optional[str] = "mlp",
        reference_frame: Optional[str] = "first",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] = False,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        max_length: Optional[int] = None,
        add_frame_index: bool = False,
        **kwargs,
    ) -> None:
        # Keep constructor strict for reproducible LMMS eval runs.
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        if geometry_encoder_path is None:
            raise ValueError("`geometry_encoder_path` must be provided for `vgllm_peft`.")

        super(VGLLM, self).__init__()

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.add_frame_index = add_frame_index
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        config = AutoConfig.from_pretrained(pretrained)
        setattr(config, "use_geometry_encoder", use_geometry_encoder)
        setattr(config, "geometry_encoder_type", geometry_encoder_type)
        setattr(config, "feature_fusion_method", feature_fusion_method)
        setattr(config, "fusion_num_layers", fusion_num_layers)
        setattr(config, "geometry_merger_type", geometry_merger_type)
        if geometry_encoder_type == "vggt":
            setattr(config, "reference_frame", reference_frame)

        model_load_kwargs = {
            "pretrained_model_name_or_path": pretrained,
            "config": config,
            "geometry_encoder_path": geometry_encoder_path,
            "device_map": self.device_map,
            "use_safetensors": True,
        }
        if use_flash_attention_2:
            model_load_kwargs["attn_implementation"] = "flash_attention_2"
            model_load_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_load_kwargs["torch_dtype"] = "auto"

        self._model = Qwen2_5_VLForConditionalGenerationWithVGGT.from_pretrained(**model_load_kwargs).eval()

        if peft is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError("`peft` is required for `vgllm_peft`. Install it via `pip install peft`.") from exc

            eval_logger.info(f"Loading LoRA adapter from {peft}")
            self._model = PeftModel.from_pretrained(self._model, peft, is_trainable=False)
            if merge_peft:
                eval_logger.info("Merging LoRA adapter into the base model for inference")
                self._model = self._model.merge_and_unload()
            self._model = self._model.eval()

        processor_name = processor_pretrained or pretrained
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self.processor = AutoProcessor.from_pretrained(processor_name, max_pixels=max_pixels, min_pixels=min_pixels, padding_side="left")
        self._tokenizer = AutoTokenizer.from_pretrained(processor_name, padding_side="left")

        if max_length is not None:
            eval_logger.warning(f"Setting max_length to {max_length}")
            setattr(self.processor.tokenizer, "model_max_length", max_length)
            setattr(self._tokenizer, "model_max_length", max_length)

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
            self._model = self.model.to("cuda").to(torch.bfloat16)
