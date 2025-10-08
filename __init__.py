"""
OneLLM: Multimodal Large Language Model

A unified multimodal large language model capable of understanding and generating
content across image, video, audio, point cloud, and other modalities.
"""

from .model import MetaModel
from .util import LayerOutputCapture, create_layer_capture

__version__ = "0.1.0"
__all__ = ['MetaModel', 'LayerOutputCapture', 'create_layer_capture']