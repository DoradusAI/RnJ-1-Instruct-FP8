#!/usr/bin/env python3
"""
Quantize RnJ-1-Instruct to FP8 using llmcompressor (Neural Magic)

This script performs dynamic FP8 quantization (W8A8) on the RnJ-1-Instruct model.
No calibration data is needed - it's a fast, deterministic conversion.

Requirements:
    pip install llmcompressor torch transformers accelerate

Usage:
    python quantize_rnj1_fp8.py

Output:
    ./RnJ-1-Instruct-FP8/ - vLLM-compatible FP8 model

Hardware Requirements:
    - 32GB+ RAM for model loading
    - GPU optional (CPU quantization supported)
"""

import os
import sys
from pathlib import Path

def main():
    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
    except ImportError:
        print("Error: llmcompressor not installed")
        print("Install with: pip install llmcompressor torch transformers accelerate")
        sys.exit(1)

    # Configuration
    MODEL_PATH = "DoradusAI/RnJ-1-Instruct"
    OUTPUT_PATH = "./RnJ-1-Instruct-FP8"

    print(f"Quantizing {MODEL_PATH} to FP8...")
    print(f"Output directory: {OUTPUT_PATH}")

    # FP8 Dynamic quantization recipe
    # - Weights: FP8 E4M3 (channel-wise, symmetric)
    # - Activations: FP8 E4M3 (token-wise, dynamic)
    # - lm_head kept in BF16 for output quality
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"],
    )

    # Run quantization
    oneshot(
        model=MODEL_PATH,
        output_dir=OUTPUT_PATH,
        recipe=recipe,
        num_calibration_samples=0,  # Dynamic quant needs no calibration
        save_compressed=True,
    )

    print(f"\nQuantization complete!")
    print(f"Model saved to: {OUTPUT_PATH}")
    print(f"\nTo test with vLLM:")
    print(f"  python -m vllm.entrypoints.openai.api_server \\")
    print(f"    --model {OUTPUT_PATH} \\")
    print(f"    --trust-remote-code")


if __name__ == "__main__":
    main()
