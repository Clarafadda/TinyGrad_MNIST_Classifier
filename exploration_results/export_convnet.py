#!/usr/bin/env python3
"""
Export CNN model to WebGPU
Usage: python export_convnet.py
"""

from pathlib import Path
from typing import Callable
from tinygrad import Tensor, nn
from tinygrad.device import Device
from tinygrad.nn.state import load_state_dict, safe_load, safe_save
from tinygrad.helpers import Context
from export_model import export_model
import os

# Désactiver JIT pour l'export
os.environ['JIT'] = '0'


class Model:
    def __init__(self):
        self.layers: list[Callable[[Tensor], Tensor]] = [
            nn.Conv2d(1, 32, 5), Tensor.silu,
            nn.Conv2d(32, 32, 5), Tensor.silu,
            nn.BatchNorm(32), Tensor.max_pool2d,
            nn.Conv2d(32, 64, 3), Tensor.silu,
            nn.Conv2d(64, 64, 3), Tensor.silu,
            nn.BatchNorm(64), Tensor.max_pool2d,
            lambda x: x.flatten(1), nn.Linear(576, 10),
        ]

    def __call__(self, x: Tensor) -> Tensor: return x.sequential(self.layers)


if __name__ == "__main__":
    model_name = "mnist_convnet"
    dir_name = Path("mnist_convnet")

    if not (dir_name / f"{model_name}.safetensors").exists():
        print(f"❌ Error: {model_name}.safetensors not found!")
        print(f"   Please train the model first with: python train_convnet.py")
        exit(1)

    print(f"📦 Loading trained model from {dir_name}/{model_name}.safetensors...")

    with Context(JIT=0):
        Device.DEFAULT = "WEBGPU"
        model = Model()

        # Load trained weights
        state_dict = safe_load(dir_name / f"{model_name}.safetensors")
        load_state_dict(model, state_dict)

        # Create dummy input
        input = Tensor.randn(1, 1, 28, 28)

        print("🔄 Exporting to WebGPU format...")
        try:
            prg, *_, state = export_model(model, Device.DEFAULT.lower(), input, model_name=model_name)

            # Save WebGPU files
            safe_save(state, dir_name / f"{model_name}.webgpu.safetensors")
            with open(dir_name / f"{model_name}.js", "w") as text_file:
                text_file.write(prg)

            print(f"✅ Export successful!")
            print(f"   📄 {dir_name}/{model_name}.js")
            print(f"   📄 {dir_name}/{model_name}.webgpu.safetensors")

        except Exception as e:
            print(f"❌ Export failed: {e}")
            import traceback

            traceback.print_exc()
            exit(1)