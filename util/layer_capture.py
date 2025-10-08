"""
Layer Output Capture Utility for OneLLM

This module provides functionality to capture and save intermediate layer outputs
from OneLLM model during inference.
"""

import os
import torch
import json
from datetime import datetime
from typing import Dict, List, Optional, Any


class LayerOutputCapture:
    """
    Captures and saves intermediate layer outputs from OneLLM model.

    Features:
    - Captures layer outputs during forward pass
    - Saves outputs as .pt files with metadata
    - Supports different modalities (image, text)
    - Organized file structure for easy analysis
    """

    def __init__(self, output_dir: str = "./layer_outputs", enabled: bool = True):
        """
        Initialize LayerOutputCapture.

        Args:
            output_dir: Directory to save layer outputs
            enabled: Whether capturing is enabled
        """
        self.enabled = enabled
        self.output_dir = output_dir
        self.current_sample_id = None
        self.captured_layers = {}

        if self.enabled:
            os.makedirs(self.output_dir, exist_ok=True)

    def set_sample(self, sample_id: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Set current sample being processed.

        Args:
            sample_id: Unique identifier for the sample
            metadata: Additional metadata to save
        """
        if not self.enabled:
            return

        self.current_sample_id = sample_id
        self.captured_layers = {}

        # Create sample directory
        sample_dir = os.path.join(self.output_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        # Save metadata
        if metadata:
            metadata_path = os.path.join(sample_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

    def capture_layer(self, layer_name: str, output: torch.Tensor,
                     additional_info: Optional[Dict[str, Any]] = None):
        """
        Capture output from a specific layer.

        Args:
            layer_name: Name/identifier of the layer
            output: Tensor output from the layer
            additional_info: Additional information to save
        """
        if not self.enabled or self.current_sample_id is None:
            return

        # Store in memory
        self.captured_layers[layer_name] = {
            'output': output.detach().cpu(),
            'shape': list(output.shape),
            'dtype': str(output.dtype),
            'timestamp': datetime.now().isoformat(),
            'additional_info': additional_info or {}
        }

        # Save to file immediately
        self._save_layer_output(layer_name)

    def _save_layer_output(self, layer_name: str):
        """Save captured layer output to disk."""
        sample_dir = os.path.join(self.output_dir, self.current_sample_id)
        file_path = os.path.join(sample_dir, f"{layer_name}.pt")

        layer_data = self.captured_layers[layer_name]
        torch.save(layer_data, file_path)

        print(f"Saved layer '{layer_name}' output: {layer_data['shape']} -> {file_path}")

    def save_summary(self):
        """Save a summary of all captured layers for current sample."""
        if not self.enabled or self.current_sample_id is None:
            return

        sample_dir = os.path.join(self.output_dir, self.current_sample_id)
        summary_path = os.path.join(sample_dir, "capture_summary.json")

        summary = {
            'sample_id': self.current_sample_id,
            'total_captured_layers': len(self.captured_layers),
            'captured_layers': list(self.captured_layers.keys()),
            'capture_time': datetime.now().isoformat()
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Saved capture summary for sample '{self.current_sample_id}'")

    def finish_sample(self):
        """Finish processing current sample and save summary."""
        if self.enabled and self.current_sample_id is not None:
            self.save_summary()
            self.current_sample_id = None
            self.captured_layers = {}


class LayerCaptureHook:
    """
    Hook class to register with PyTorch modules for automatic output capture.
    """

    def __init__(self, capture_instance: LayerOutputCapture, layer_name: str):
        """
        Initialize hook.

        Args:
            capture_instance: LayerOutputCapture instance
            layer_name: Name to identify this layer
        """
        self.capture = capture_instance
        self.layer_name = layer_name

    def hook_fn(self, module, input, output):
        """Hook function called during forward pass."""
        self.capture.capture_layer(self.layer_name, output)


# Utility function to create capture instance
def create_layer_capture(output_dir: str = "./layer_outputs",
                        enabled: bool = True) -> LayerOutputCapture:
    """
    Create and return a LayerOutputCapture instance.

    Args:
        output_dir: Directory to save outputs
        enabled: Whether to enable capturing

    Returns:
        LayerOutputCapture instance
    """
    return LayerOutputCapture(output_dir=output_dir, enabled=enabled)