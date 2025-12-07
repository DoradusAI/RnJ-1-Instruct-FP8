# RnJ-1-Instruct-FP8 Inference Server
# Based on vLLM OpenAI-compatible server
#
# Build:
#   docker build -t rnj-1-instruct-fp8 .
#
# Run:
#   docker run --gpus '"device=0"' -p 8000:8000 rnj-1-instruct-fp8

FROM vllm/vllm-openai:v0.12.0

# Model will be downloaded from HuggingFace on first run
ENV MODEL_NAME="Doradus/RnJ-1-Instruct-FP8"
ENV MAX_MODEL_LEN="8192"
ENV GPU_MEMORY_UTILIZATION="0.90"

# vLLM settings
ENV VLLM_ATTENTION_BACKEND="FLASHINFER"

EXPOSE 8000

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]

CMD ["--model", "Doradus/RnJ-1-Instruct-FP8", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--tensor-parallel-size", "1", \
     "--max-model-len", "8192", \
     "--gpu-memory-utilization", "0.90", \
     "--dtype", "auto", \
     "--trust-remote-code", \
     "--served-model-name", "rnj-1-instruct-fp8", \
     "--enable-chunked-prefill", \
     "--max-num-seqs", "32"]
