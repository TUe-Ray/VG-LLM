
import torch
import torch.nn as nn
from typing import Optional

from .base import BaseGeometryEncoder, GeometryEncoderConfig

class Pi3XEncoder(BaseGeometryEncoder):
    """PI3X geometry encoder wrapper."""
    
    def __init__(self, config: GeometryEncoderConfig):
        super().__init__(config)
        print("initialize the Pi3X encoder
        ")

    def encode(self, images: torch.Tensor) -> torch.Tensor: