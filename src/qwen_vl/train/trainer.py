# 這個檔案是用於訓練 Qwen VL 模型的自訂訓練器
# 主要功能包括：
# 1. 實作自訂的 Flash Attention 機制以提升訓練效率
# 2. 替換模型的注意力機制
# 3. 創建具有不同學習率的優化器（支援視覺塔和投影器使用不同學習率）
# 4. 提供可訓練參數的視覺化工具

import os
from typing import Dict, List, Optional, Sequence

import datasets
import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from torch.utils.data import DataLoader, Sampler
from transformers import Trainer
from transformers.cache_utils import Cache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLModel,
)
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    get_parameter_names,
    has_length,
    is_sagemaker_mp_enabled,
)
from transformers.trainer_utils import seed_worker


def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    **kwargs,
):
    """
    自訂的 Flash Attention 前向傳播函數
    
    功能說明：
    - 使用 Flash Attention 算法來加速注意力計算，降低記憶體使用
    - 處理變長序列（variable-length sequences）的注意力計算
    - 支援因果遮罩（causal masking）用於自回歸生成
    
    參數說明：
        query_states (`torch.Tensor`):
            查詢狀態張量，用於計算注意力分數
        key_states (`torch.Tensor`):
            鍵狀態張量，與查詢進行點積運算
        value_states (`torch.Tensor`):
            值狀態張量，根據注意力權重進行加權求和
        attention_mask (`torch.Tensor`):
            注意力遮罩，這裡用作累積序列長度索引（cu_seqlens）
        dropout (`float`):
            注意力 dropout 比率，防止過擬合
        softmax_scale (`float`, *optional*):
            QK^T 在 softmax 前的縮放因子，預設為 1/sqrt(head_dim)
        use_top_left_mask (`bool`, 預設 `False`):
            處理不同版本 flash_attn 的遮罩對齊差異
        softcap (`float`, *optional*):
            注意力邏輯值的軟上限，例如用於 gemma2 模型
        deterministic (`bool`, *optional*):
            是否啟用確定性計算模式（flash_attn>=2.4.1）
    """
    # 確保批次大小為 1（所有張量的第一維度都是 1）
    assert query_states.size(0) == key_states.size(0) == value_states.size(0) == 1
    
    # 移除批次維度，因為我們使用變長注意力
    query_states = query_states.squeeze(0)
    key_states = key_states.squeeze(0)
    value_states = value_states.squeeze(0)
    
    # attention_mask 在這裡實際上是累積序列長度（cumulative sequence lengths）
    cu_seqlens = attention_mask

    # 計算最大序列長度（用於 Flash Attention）
    with torch.no_grad():
        max_seqlen = max(
            [
                cu_seqlens[idx + 1] - cu_seqlens[idx]
                for idx in range(cu_seqlens.size(0) - 1)
            ]
        ).item()

    # 處理因果遮罩的對齊問題
    if not use_top_left_mask:
        causal = is_causal
    else:
        # 當查詢長度為 1 時不使用因果遮罩（用於推理階段）
        causal = is_causal and query_length != 1

    # 準備 Flash Attention 的額外參數
    flash_kwargs = {}

    # 如果有設定軟上限，加入參數字典
    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    # 呼叫變長 Flash Attention 函數
    attn_output = flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q=cu_seqlens,  # 查詢的累積序列長度
        cu_seqlens_k=cu_seqlens,  # 鍵的累積序列長度
        max_seqlen_q=max_seqlen,  # 查詢的最大序列長度
        max_seqlen_k=max_seqlen,  # 鍵的最大序列長度
        dropout_p=dropout,
        softmax_scale=softmax_scale,
        causal=causal,  # 是否使用因果遮罩
        **flash_kwargs,
    )

    # 恢復批次維度以保持張量形狀一致性
    attn_output = attn_output.unsqueeze(0)
    query_states = query_states.unsqueeze(0)
    key_states = key_states.unsqueeze(0)
    value_states = value_states.unsqueeze(0)

    return attn_output


def _update_causal_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Cache,
    output_attentions: bool,
):
    """
    更新因果遮罩的函數
    
    功能說明：
    - 這是一個簡化的版本，直接返回原始的 attention_mask
    - 因為我們在 _flash_attention_forward 中已經處理了遮罩邏輯
    - 避免額外的遮罩計算以提升效率
    
    參數說明：
        attention_mask: 原始注意力遮罩
        input_tensor: 輸入張量
        cache_position: 快取位置資訊
        past_key_values: 過去的鍵值對快取
        output_attentions: 是否輸出注意力權重
    
    返回：
        直接返回原始的 attention_mask
    """

    """
    From GPT
    但這份 patch 直接回傳原本的 attention_mask，等於告訴模型：

    「不要再自己改 mask 了，我提供的 mask 已經是你該用的格式/結果。」

    這通常是因為：

    你已經把 mask 改成 cu_seqlens（不是 0/1 mask）

    或 flatten/packing 導致原本內建 mask 邏輯不適用

    讓內建 _update_causal_mask 繼續動會把格式弄壞或 shape 不匹配

    ✅ 作用：避免模型內部 mask 邏輯干擾 varlen attention 的特殊 mask 格式
    """
    return attention_mask


def replace_qwen2_vl_attention_class():
    """
    替換 Qwen2 VL 模型的注意力機制
    
    功能說明：
    - 將原始的注意力實作替換為我們自訂的 Flash Attention 版本
    - 同時支援 Qwen2 VL 和 Qwen2.5 VL 兩個版本
    - 使用猴子補丁（monkey patching）的方式修改 transformers 套件中的函數
    - 這樣可以在不修改原始套件的情況下使用自訂實作
    """
    import transformers
    import transformers.modeling_flash_attention_utils

    # 替換 Qwen2 VL 的 Flash Attention 實作
    transformers.models.qwen2_vl.modeling_qwen2_vl._flash_attention_forward = (
        _flash_attention_forward
    )
    # 替換 Qwen2 VL 的因果遮罩更新函數
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel._update_causal_mask = (
        _update_causal_mask
    )
    # 替換 Qwen2.5 VL 的 Flash Attention 實作
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl._flash_attention_forward = (
        _flash_attention_forward
    )
    # 替換 Qwen2.5 VL 的因果遮罩更新函數
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel._update_causal_mask = (
        _update_causal_mask
    )


def print_trainable_parameters_visual(self) -> None:
    """
    印出視覺模組中所有可訓練參數的狀態
    
    功能說明：
    - 檢查視覺轉換器（Vision Transformer）中每個注意力區塊（block）的訓練狀態
    - 檢查合併器模組（merger module）的訓練狀態
    - 以易讀的格式輸出哪些區塊是可訓練的，哪些是凍結的
    - 這對於理解模型的哪些部分正在被訓練非常有用
    
    輸出資訊：
    - 可訓練的注意力區塊索引列表
    - 不可訓練的注意力區塊索引列表
    - 合併器模組是否可訓練
    """
    trainable_blocks = []      # 儲存可訓練區塊的索引
    non_trainable_blocks = []  # 儲存不可訓練區塊的索引

    # 遍歷所有視覺注意力區塊，檢查訓練狀態
    for block_idx, block in enumerate(self.blocks):
        # 檢查該區塊的所有參數是否都需要梯度（可訓練）
        is_trainable = all(param.requires_grad for param in block.parameters())
        if is_trainable:
            trainable_blocks.append(block_idx)
        else:
            non_trainable_blocks.append(block_idx)

    # 檢查合併器模組是否有任何可訓練的參數
    is_merger_trainable = any(param.requires_grad for param in self.merger.parameters())

    # 印出結果
    print("Vision Module - Attention Blocks:")
    print(
        f"Trainable Block Indices: {trainable_blocks if trainable_blocks else 'None'}"
    )
    print(
        f"Non-Trainable Block Indices: {non_trainable_blocks if non_trainable_blocks else 'None'}"
    )
    print(f"Merger Module Trainable: {is_merger_trainable}")


def print_trainable_parameters(self) -> None:
    """
    印出語言模型（LLM）模組中所有可訓練參數的狀態
    
    功能說明：
    - 檢查詞嵌入層（embedding layer）的訓練狀態
    - 檢查每個解碼器層（decoder layer）的訓練狀態
    - 以易讀的格式輸出哪些層是可訓練的，哪些是凍結的
    - 幫助追蹤和驗證訓練配置是否正確
    
    輸出資訊：
    - 詞嵌入層是否可訓練
    - 可訓練的解碼器層索引列表
    - 不可訓練的解碼器層索引列表
    """
    # 檢查詞嵌入層是否有可訓練的參數
    is_embed_trainable = any(
        param.requires_grad for param in self.embed_tokens.parameters()
    )
    print(f"LLM Module - Embed Tokens Trainable: {is_embed_trainable}")

    # 遍歷所有解碼器層，檢查訓練狀態
    trainable_layers = []      # 儲存可訓練層的索引
    non_trainable_layers = []  # 儲存不可訓練層的索引

    for layer_idx, layer in enumerate(self.layers):
        # 檢查該層是否有任何可訓練的參數
        is_trainable = any(param.requires_grad for param in layer.parameters())
        if is_trainable:
            trainable_layers.append(layer_idx)
        else:
            non_trainable_layers.append(layer_idx)

    # 印出層的訓練狀態
    print(
        f"LLM Module - Trainable Layer Indices: {trainable_layers if trainable_layers else 'None'}"
    )
    print(
        f"LLM Module - Non-Trainable Layer Indices: {non_trainable_layers if non_trainable_layers else 'None'}"
    )


def create_optimizer(self):
    """
    創建自訂優化器，支援不同模組使用不同的學習率
    
    功能說明：
    - 為模型的不同部分設定不同的學習率和權重衰減策略
    - 支援三種主要的學習率配置：
      1. 視覺塔（vision tower）的學習率
      2. 多模態投影器（mm_projector/merger）的學習率
      3. 其他模型部分使用預設學習率
    - 對於 LayerNorm 和 bias 參數不使用權重衰減
    
    學習率分組策略：
    - 如果設定了投影器學習率：
      - 如果也設定了視覺塔學習率：創建 6 個參數組
        1. 主模型參數（有權重衰減）
        2. 視覺塔參數（有權重衰減，使用視覺塔學習率）
        3. 主模型參數（無權重衰減）
        4. 視覺塔參數（無權重衰減，使用視覺塔學習率）
        5. 投影器參數（有權重衰減，使用投影器學習率）
        6. 投影器參數（無權重衰減，使用投影器學習率）
      - 如果只設定了投影器學習率：創建 4 個參數組
        1-2. 主模型參數（有/無權重衰減）
        3-4. 投影器參數（有/無權重衰減，使用投影器學習率）
    - 如果未設定特殊學習率：創建 2 個參數組
      1. 有權重衰減的參數
      2. 無權重衰減的參數
    
    返回：
        配置好的優化器實例
    """
    opt_model = self.model

    if self.optimizer is None:
        # 獲取所有應該使用權重衰減的參數名稱
        # LayerNorm 層通常不需要權重衰減
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        # 排除 bias 參數（bias 也不使用權重衰減）
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        
        # 情況 1：設定了投影器（merger）的學習率
        if self.args.mm_projector_lr is not None and self.args.mm_projector_lr != 0:
            # 找出所有投影器相關的參數
            projector_parameters = [
                name for name, _ in opt_model.named_parameters() if "merger" in name
            ]
            
            # 情況 1.1：同時設定了視覺塔的學習率
            if self.args.vision_tower_lr is not None and self.args.vision_tower_lr != 0:
                # 找出所有視覺塔相關的參數
                vision_tower_parameters = [
                    name for name, _ in opt_model.named_parameters() if "visual" in name
                ]
                
                # 創建 6 個參數組（主模型、視覺塔、投影器各有有/無權重衰減兩組）
                optimizer_grouped_parameters = [
                    # 組 1：主模型參數，有權重衰減
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    # 組 2：視覺塔參數，有權重衰減，使用視覺塔學習率
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.vision_tower_lr,
                    },
                    # 組 3：主模型參數，無權重衰減
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n not in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    # 組 4：視覺塔參數，無權重衰減，使用視覺塔學習率
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and n in vision_tower_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.vision_tower_lr,
                    },
                    # 組 5：投影器參數，有權重衰減，使用投影器學習率
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    # 組 6：投影器參數，無權重衰減，使用投影器學習率
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            # 情況 1.2：只設定了投影器學習率，沒有設定視覺塔學習率
            else:
                # 創建 4 個參數組（主模型和投影器各有有/無權重衰減兩組）
                optimizer_grouped_parameters = [
                    # 組 1：主模型參數，有權重衰減
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    # 組 2：主模型參數，無權重衰減
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    # 組 3：投影器參數，有權重衰減，使用投影器學習率
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    # 組 4：投影器參數，無權重衰減，使用投影器學習率
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
        # 情況 2：沒有設定特殊的學習率
        else:
            # 創建 2 個參數組（標準配置）
            optimizer_grouped_parameters = [
                # 組 1：所有需要權重衰減的參數
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                # 組 2：所有不需要權重衰減的參數
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        # 獲取優化器類別和參數（例如 AdamW、SGD 等）
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        # 創建優化器實例
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return self.optimizer


# ============================================================================
# 應用猴子補丁（Monkey Patches）
# 將我們自訂的函數替換到對應的類別中
# ============================================================================

# 替換 Trainer 的優化器創建方法
Trainer.create_optimizer = create_optimizer

# 為 Qwen2 VL 的視覺轉換器添加參數狀態印出方法
Qwen2VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
# 為 Qwen2 VL 模型添加參數狀態印出方法
Qwen2VLModel.print_trainable_parameters = print_trainable_parameters

# 為 Qwen2.5 VL 的視覺轉換器添加參數狀態印出方法
Qwen2_5_VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
# 為 Qwen2.5 VL 模型添加參數狀態印出方法
Qwen2_5_VLModel.print_trainable_parameters = print_trainable_parameters
