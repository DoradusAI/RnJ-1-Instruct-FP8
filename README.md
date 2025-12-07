---
library_name: transformers
pipeline_tag: text-generation
license: gemma
language:
- en
base_model:
- DoradusAI/RnJ-1-Instruct
tags:
- gemma3
- rnj
- doradus
- instruction-following
- fp8
- quantized
- vllm
- sglang
---

# RnJ-1-Instruct-FP8

<div align="center">
  <img src="https://doradusonline.com/rnj-banner.png" width="55%" alt="RnJ-1" />
</div>

## Model Description

This is an **FP8 quantized** version of [DoradusAI/RnJ-1-Instruct](https://huggingface.co/DoradusAI/RnJ-1-Instruct), created using [llmcompressor](https://github.com/vllm-project/llm-compressor) (Neural Magic).

**Key Benefits:**
- ~50% smaller model size (8GB vs 16GB)
- Native FP8 inference on Ada Lovelace, Hopper, and Blackwell GPUs
- **Single consumer GPU deployment** on 12GB+ cards (RTX 3060, RTX 4070, etc.)
- Native vLLM and SGLang support
- Minimal quality loss with FP8 dynamic quantization

## Key Features

RnJ-1-Instruct is a Gemma3-based instruction-following model with:

- **Strong Math Performance**: GSM8K 87.19% (5-shot)
- **Multi-Domain Knowledge**: MMLU-Pro 44.45%
- **Efficient Architecture**: Only 8B parameters, fast inference
- **32K Context**: Extended context window for documents

## Quantization Details

| Property | Value |
|----------|-------|
| Quantization Method | FP8 Dynamic (W8A8) |
| Weights Precision | FP8 E4M3 (8-bit) |
| Activations Precision | FP8 E4M3 (8-bit, dynamic) |
| Ignored Layers | `lm_head` (kept in BF16) |
| Quantization Tool | llmcompressor 0.12.2 |
| Original Model Size | ~16GB |
| Quantized Model Size | ~8GB |

### Quantization Recipe

```yaml
default_stage:
  default_modifiers:
    QuantizationModifier:
      targets: [Linear]
      ignore: [lm_head]
      scheme: FP8_DYNAMIC
```

## Quick Start with Docker

The easiest way to run this model. No setup required - just Docker with NVIDIA runtime.

### Docker Compose (Recommended)

```bash
# Download docker-compose.yml
wget https://huggingface.co/Doradus/RnJ-1-Instruct-FP8/raw/main/docker/docker-compose.yml

# Run on single GPU (12GB+ recommended)
docker compose up

# Or specify GPU
GPU_ID=0 docker compose up
```

### Docker Run

```bash
# Single GPU (12GB+ VRAM recommended)
docker run --gpus '"device=0"' -p 8000:8000 \
  -v hf_cache:/root/.cache/huggingface \
  --shm-size=4g \
  vllm/vllm-openai:v0.12.0 \
  --model Doradus/RnJ-1-Instruct-FP8 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code
```

### Test the API

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Doradus/RnJ-1-Instruct-FP8",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Usage

### vLLM (Recommended)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Doradus/RnJ-1-Instruct-FP8 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --trust-remote-code
```

### SGLang

```bash
python -m sglang.launch_server \
  --model-path Doradus/RnJ-1-Instruct-FP8 \
  --host 0.0.0.0 \
  --port 8000 \
  --tp 1
```

### Python Example

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="rnj-1-instruct-fp8",
    messages=[{"role": "user", "content": "Explain quantum computing in simple terms."}],
    max_tokens=500
)

print(response.choices[0].message.content)
```

## Architecture Details

This is a **dense transformer** model based on Gemma3 architecture:

| Property | Value |
|----------|-------|
| Total Parameters | ~8B |
| Hidden Size | 4096 |
| Attention Heads | 32 |
| KV Heads (GQA) | 8 |
| Layers | 32 |
| Intermediate Size | 16384 |
| Max Context | 32,768 tokens |
| Vocabulary | 128,256 tokens |

## Hardware Requirements

### VRAM Analysis

Model weights: **~8GB** (vs ~16GB BF16 original)

| Context Length | KV Cache (FP16) | Total VRAM | Fits Single GPU? |
|----------------|-----------------|------------|------------------|
| 4K tokens | ~0.3 GB | ~9 GB | RTX 3060 12GB |
| 8K tokens | ~0.6 GB | ~9 GB | RTX 4070 12GB |
| 16K tokens | ~1.2 GB | ~10 GB | RTX 4080 16GB |
| 32K tokens | ~2.4 GB | ~11 GB | RTX 4090 24GB |

### Recommended Configurations

| GPU Setup | Max Context | Performance | Notes |
|-----------|-------------|-------------|-------|
| 1x RTX 3060 (12GB) | ~8K tokens | ~50 tok/s | Consumer budget |
| 1x RTX 4070 (12GB) | ~8K tokens | ~80 tok/s | Consumer mid-range |
| **1x RTX 4080 (16GB)** | ~16K tokens | ~100 tok/s | **Recommended consumer** |
| 1x RTX 4090 (24GB) | ~32K tokens | ~120 tok/s | Full context |
| 1x RTX 6000 Ada (48GB) | 32K tokens | ~150 tok/s | Professional |

**Note**: FP8 inference requires CUDA compute capability 8.9+ (Ada Lovelace) or 9.0+ (Hopper/Blackwell) for optimal performance.

## Quality & Performance

### FP8 Quantized Benchmarks (lm-evaluation-harness)

| Benchmark | Score | Notes |
|-----------|-------|-------|
| **GSM8K (5-shot strict)** | **87.19%** | Math reasoning |
| **MMLU-Pro** | **44.45%** | Multi-domain knowledge |
| IFEval (prompt-strict) | TBD | Instruction following |

*Benchmarked 2025-12-07 on RTX PRO 6000 Blackwell (96GB) using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with vLLM 0.12.0*

### MMLU-Pro Category Breakdown

| Category | Score |
|----------|-------|
| Biology | 63.18% |
| Psychology | 56.64% |
| Economics | 54.98% |
| Math | 54.92% |
| Computer Science | 47.56% |
| Business | 46.89% |
| Physics | 45.11% |
| Philosophy | 41.88% |
| Health | 39.61% |
| History | 37.80% |
| Engineering | 37.67% |
| Chemistry | 37.72% |
| Law | 21.98% |

## Reproduction

To reproduce this quantization:

```python
#!/usr/bin/env python3
"""
Quantize RnJ-1-Instruct to FP8 using llmcompressor (Neural Magic)
Dynamic quantization - no calibration data needed, fast conversion
Output is vLLM-compatible FP8
"""

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
import torch

MODEL_PATH = "DoradusAI/RnJ-1-Instruct"
OUTPUT_PATH = "./RnJ-1-Instruct-FP8"

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
)

oneshot(
    model=MODEL_PATH,
    output_dir=OUTPUT_PATH,
    recipe=recipe,
    num_calibration_samples=0,
    save_compressed=True,
)
```

**Requirements:**
```
pip install llmcompressor torch transformers accelerate
```

## Original Model

This quantization is based on [DoradusAI/RnJ-1-Instruct](https://huggingface.co/DoradusAI/RnJ-1-Instruct).

RnJ-1 is DoradusAI's first instruction-tuned model based on Google's Gemma3 architecture. Key features:

- Strong math and reasoning capabilities
- Efficient 8B parameter count
- 32K context window
- Optimized for instruction following

## License

This model inherits the **Gemma License** from the original Gemma3 model.

## Acknowledgements

- [Google](https://ai.google.dev/gemma) for the Gemma3 base model
- [Neural Magic / vLLM](https://github.com/vllm-project/llm-compressor) for llmcompressor
- [DoradusAI](https://doradusonline.com) for the fine-tuning and FP8 quantization
