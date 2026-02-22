FROM vllm/vllm-openai:nightly

# DeepSeek-OCR-2 remote code imports these at startup.
# Keep this image focused: only add required runtime deps.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    addict \
    matplotlib \
    einops \
    easydict

# GLM-OCR model type (glm_ocr) requires bleeding-edge Transformers.
RUN pip install --no-cache-dir \
    "git+https://github.com/huggingface/transformers.git"
