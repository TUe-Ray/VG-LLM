# Training script for Qwen Vision-Language models with optional geometry encoder integration
# 支援幾何編碼器整合的 Qwen 視覺語言模型訓練腳本
# This script supports both Qwen2VL and Qwen2.5VL models with configurable parameter tuning
# 此腳本支援 Qwen2VL 和 Qwen2.5VL 模型，具有可配置的參數調整功能
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Import necessary libraries for model training, data processing, and distributed computing
# 匯入模型訓練、資料處理和分散式運算所需的必要函式庫
import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path

# Add project root to Python path for local module imports
# 將專案根目錄加入 Python 路徑以匯入本機模組
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import custom training components and model modifications
# 匯入自訂訓練元件和模型修改
import qwen_vl.train.trainer
import qwen_vl.train.sampler
from trainer import replace_qwen2_vl_attention_class

# Import Hugging Face transformers and local modules
# 匯入 Hugging Face transformers 和本機模組
from transformers import (
    Qwen2VLForConditionalGeneration,
)
from qwen_vl.data.data_qwen import make_supervised_data_module

from qwen_vl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer, AutoConfig, set_seed, enable_full_determinism

# Global variable to track the local rank in distributed training
# 用於追蹤分散式訓練中本機排名的全域變數
local_rank = None

#local_rank 通常由 torchrun 或 Accelerate 設定（每張 GPU 一個 process）。
#這個函式保證 只有 rank0（通常是第一張卡）才會印 log，避免多卡時同樣訊息印 8 次。
def rank0_print(*args):
    """Print messages only from rank 0 process in distributed training to avoid duplicate outputs."""
    """在分散式訓練中僅從排名0程序列印訊息以避免重複輸出。"""
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    Safely save model state dict to disk, handling both DeepSpeed and regular training scenarios.
    安全地將模型狀態字典儲存到磁碟，處理 DeepSpeed 和一般訓練情境。
    For DeepSpeed training, uses trainer's built-in save method.
    對於 DeepSpeed 訓練，使用訓練器的內建儲存方法。
    For regular training, manually saves CPU state dict to avoid GPU memory issues.
    對於一般訓練，手動儲存 CPU 狀態字典以避免 GPU 記憶體問題。

    From GPT
    如果用 DeepSpeed（trainer.deepspeed=True）
    DeepSpeed 可能把參數分片（ZeRO stage 2/3），此時你不能簡單 model.state_dict() 再存，因為每張卡只拿到一部分。
    所以它直接用 trainer.save_model() 走 HF + DeepSpeed 的正規保存流程（會在需要時做 gather/整理）。
    torch.cuda.synchronize() 是為了確保前面的 GPU 操作完成再存，避免 race condition。

    如果不是 DeepSpeed
    它自己把 state_dict 搬到 CPU 再存：

    好處：存檔時不吃爆 GPU 記憶體

    避免某些情況下直接存 GPU tensor 造成額外開銷
    """

    if trainer.deepspeed:
        # DeepSpeed handles model saving automatically with proper state aggregation
        # DeepSpeed 透過適當的狀態聚合自動處理模型儲存
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    # For non-DeepSpeed training, manually handle state dict saving
    # 對於非 DeepSpeed 訓練，手動處理狀態字典儲存
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        # Move state dict to CPU to free GPU memory before saving
        # 將狀態字典移至 CPU 以在儲存前釋放 GPU 記憶體
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    """
    Configure which model components should be trained based on model arguments.
    根據模型參數配置應訓練哪些模型元件。
    This function enables/disables gradient computation for different model parts:
    此函式啟用/停用不同模型部分的梯度計算：
    - Visual encoder (vision transformer)
    - 視覺編碼器（視覺變換器）
    - Multimodal MLP (vision-language connector)
    - 多模態 MLP（視覺語言連接器）
    - Language model backbone
    - 語言模型主幹
    - Geometry encoder (if used)
    - 幾何編碼器（如果使用）
    """
    # Configure visual encoder training
    # 配置視覺編碼器訓練
    if model_args.tune_mm_vision:
        # Enable training of visual encoder parameters (vision transformer)
        # 啟用視覺編碼器參數（視覺變換器）的訓練
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        # Freeze visual encoder parameters
        # 凍結視覺編碼器參數
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    # Configure multimodal MLP connector training
    # 配置多模態 MLP 連接器訓練
    if model_args.tune_mm_mlp:
        # Enable training of vision-language fusion layer (merger/projector)
        # 啟用視覺語言融合層（合併器/投影器）的訓練
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        # Freeze multimodal connector parameters
        # 凍結多模態連接器參數
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    # Configure language model backbone training
    # 配置語言模型主幹訓練
    if model_args.tune_mm_llm:
        # Enable training of language model parameters and output head
        # 啟用語言模型參數和輸出頭的訓練
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        # Freeze language model backbone and output head
        # 凍結語言模型主幹和輸出頭
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False

    # Configure geometry encoder (VGGT) - always frozen when present
    # 配置幾何編碼器（VGGT）- 存在時總是凍結
    if model_args.use_geometry_encoder:
        # Geometry encoder (vggt) is kept frozen during training
        # 幾何編碼器（vggt）在訓練期間保持凍結狀態
        for n, p in model.geometry_encoder.named_parameters():
            p.requires_grad = False

def train(attn_implementation="flash_attention_2"):
    """
    Main training function that handles model loading, data preparation, and training execution.
    處理模型載入、資料準備和訓練執行的主要訓練函式。
    Supports both Qwen2VL and Qwen2.5VL with optional geometry encoder integration.
    支援帶有可選幾何編碼器整合的 Qwen2VL 和 Qwen2.5VL。
    
    Args:
        attn_implementation: Attention mechanism implementation (default: flash_attention_2)
        attn_implementation: 注意力機制實作（預設：flash_attention_2）
    """
    global local_rank

    # Parse command line arguments into structured dataclasses
    # 將命令列引數解析為結構化資料類別
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    # ---------------- W&B logging ----------------
    # Enable Weights & Biases logging via Hugging Face Trainer integration.
    # This avoids manual wandb.init/log and is safe under DDP (only rank0 will log).
    try:
        report_to = getattr(training_args, "report_to", None)
        # If user didn't specify any reporters, default to wandb
        if report_to is None or report_to == [] or report_to == "none" or report_to == ["none"]:
            training_args.report_to = ["wandb"]
        # HF also accepts a single string; normalize to list for consistency
        elif isinstance(report_to, str):
            training_args.report_to = [report_to]

        # Give W&B a sensible default run name (can still be overridden by --run_name)
        if getattr(training_args, "run_name", None) in (None, ""):
            training_args.run_name = Path(training_args.output_dir).name
    except Exception:
        # If TrainingArguments doesn't expose these fields in a custom fork, just skip.
        pass
    
    # Set random seed for reproducible training
    # 設定隨機種子以實現可重現的訓練
    set_seed(training_args.seed)
    # enable_full_determinism(training_args.seed)  # Commented out - may impact performance
    # enable_full_determinism(training_args.seed)  # 已註解 - 可能影響效能

    # Set local rank for distributed training coordination
    # 設定本機排名以協調分散式訓練
    local_rank = training_args.local_rank
    
    # Create output directory if it doesn't exist
    # 如果輸出目錄不存在則建立
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Model loading logic - different paths for Qwen2.5VL vs Qwen2VL
    # 模型載入邏輯 - Qwen2.5VL 與 Qwen2VL 的不同路徑
    if "qwen2_5" in model_args.model_name_or_path.lower():
        # Handle Qwen2.5VL model loading
        # 處理 Qwen2.5VL 模型載入
        if not model_args.use_geometry_encoder:
            # Standard Qwen2.5VL without geometry encoder
            # 不帶幾何編碼器的標準 Qwen2.5VL
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,#ATTN2
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
        else:
            # Qwen2.5VL with integrated geometry encoder (VGGT)
            # 整合幾何編碼器（VGGT）的 Qwen2.5VL
            from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT
            
            # Load and validate model configuration
            # 載入並驗證模型配置
            config = AutoConfig.from_pretrained(model_args.model_name_or_path)
            if hasattr(config, "use_geometry_encoder") and config.use_geometry_encoder != model_args.use_geometry_encoder:
                raise ValueError(
                    "The use_geometry_encoder in config and model_args are not consistent. "
                    "Please check the model config."
                )

            # Update config with geometry encoder settings
            # 用幾何編碼器設定更新配置
            for k in [
                "use_geometry_encoder", 
                "geometry_encoder_type", 
                "reference_frame", #INCOMPLETE:  Remove for pi3
                "feature_fusion_method", 
                "fusion_num_layers",
                "geometry_merger_type" 
                
            ]:
                setattr(config, k, getattr(model_args, k))

            # Validate geometry encoder path is provided
            # 驗證已提供幾何編碼器路徑
            assert model_args.geometry_encoder_path is not None, \
                "geometry_encoder_path must be set in the config when use_geometry_encoder is True."
            
            # Load model with geometry encoder
            # 載入帶幾何編碼器的模型
            model = Qwen2_5_VLForConditionalGenerationWithVGGT.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                geometry_encoder_path=model_args.geometry_encoder_path
            )

        # Set up image processor and model type for Qwen2.5VL
        # 為 Qwen2.5VL 設定影像處理器和模型類型
        # 用 AutoProcessor 是因為 Qwen2.5-VL 的 processor 打包了影像處理與一些規則。
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        # Handle standard Qwen2VL model loading (no geometry encoder support)
        # 處理標準 Qwen2VL 模型載入（不支援幾何編碼器）
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        
        # Set up image processor and model type for Qwen2VL
        # 為 Qwen2VL 設定影像處理器和模型類型
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"

    # Apply attention class replacement for flattened data processing if needed
    # 如果需要，套用注意力類別替換以進行扁平化資料處理
    #這通常是在做 flatten 後序列更長 / 記憶體更吃緊 時，用更省記憶體或不同 attention 實作。
    # INCOMPLETE: check why data_flatten is needed for this
    #你需要去看 trainer.replace_qwen2_vl_attention_class 具體怎麼替換
    # （它很可能 monkey patch transformers 裡某個 Attention 模組）。
    
    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    
    # Disable caching during training for memory efficiency
    # 在訓練期間停用快取以提高記憶體效率
    model.config.use_cache = False

    # Set up gradient checkpointing if enabled (for memory efficiency)
    # 如果啟用，設定梯度檢查點（以提高記憶體效率）
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            # Modern transformers version with built-in method
            # 帶有內建方法的現代 transformers 版本
            model.enable_input_require_grads()
        else:
            # Fallback for older versions - manually register hook
            # 舊版本的備用方案 - 手動註冊掛鉤
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Initialize tokenizer with training-specific settings
    # 使用訓練特定設定初始化分詞器
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",  # Right padding for causal LM training
        # 因果語言模型訓練的右填充
        use_fast=False,  # Use slow tokenizer for compatibility
        # 使用慢速分詞器以確保相容性
    )
    
    # Configure which model components to train based on arguments
    # 根據引數配置要訓練的模型元件
    set_model(model_args, model)

    # Print trainable parameter information from rank 0 only
    # 僅從排名0列印可訓練參數資訊
    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()

    # Print model configuration for debugging
    # 列印模型配置以進行偵錯
    print(model.config)
    
    # Pass geometry encoder flag to data module if used
    # 如果使用幾何編碼器，將標記傳遞給資料模組
    if model_args.use_geometry_encoder:
        setattr(data_args, "use_geometry_encoder", model_args.use_geometry_encoder)
    
    # Create supervised training data module (handles data loading and preprocessing)
    # 建立監督式訓練資料模組（處理資料載入和預處理）
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # Initialize Hugging Face trainer with model, tokenizer, and data
    # 使用模型、分詞器和資料初始化 Hugging Face 訓練器
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    # Check for existing checkpoints and resume training if found
    # 檢查現有檢查點，如果找到則恢復訓練
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        # Start training from scratch
        # 從頭開始訓練
        trainer.train()
    
    # Save final training state and processor
    # 儲存最終訓練狀態和處理器
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    # Copy chat template from source model to output directory
    # 將聊天範本從來源模型複製到輸出目錄
    source_path = os.path.join(model_args.model_name_or_path, "chat_template.json")
    template_path = os.path.join(training_args.output_dir, "chat_template.json")
    shutil.copy2(source_path, template_path)

    # Re-enable caching for inference after training
    # 訓練後重新啟用快取以進行推論
    model.config.use_cache = True

    # Save final model using safe method (handles both DeepSpeed and regular training)
    # 使用安全方法儲存最終模型（處理 DeepSpeed 和一般訓練）
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


# Entry point - start training with Flash Attention 2 for efficiency
# 入口點 - 使用 Flash Attention 2 開始訓練以提高效率
if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
