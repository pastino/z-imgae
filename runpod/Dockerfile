# Z-Image Turbo RunPod Serverless Dockerfile
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 2.5+ 업그레이드 (enable_gqa 지원 필요)
RUN pip install --upgrade pip && \
    pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# diffusers 최신 버전 필요 (Z-Image 지원은 최신 버전에만 있음)
RUN pip install git+https://github.com/huggingface/diffusers \
    transformers>=4.51.0 \
    safetensors \
    accelerate \
    huggingface_hub>=0.25.0 \
    loguru \
    pillow

# RunPod SDK, AWS SDK, PEFT(LoRA 지원) 설치
RUN pip install runpod boto3 requests peft

# handler 복사
COPY handler.py /app/handler.py

# requests, Pillow 추가 (img2img 레퍼런스 이미지 다운로드용)
RUN pip install requests Pillow

# 모델은 런타임에 다운로드 (RunPod 환경변수 HF_TOKEN으로 인증)
ENV HF_HOME=/app/hf_cache
RUN mkdir -p /app/hf_cache

# 실행
CMD ["python", "-u", "/app/handler.py"]
