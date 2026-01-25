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

# RunPod SDK, AWS SDK 설치
RUN pip install runpod boto3 requests

# handler 복사
COPY handler.py /app/handler.py

# 모델 사전 다운로드 (콜드 스타트 최적화)
# 이미지 빌드 시 모델을 미리 다운로드하려면 아래 주석 해제
# RUN python -c "from diffusers import ZImagePipeline; ZImagePipeline.from_pretrained('Tongyi-MAI/Z-Image-Turbo')"

# 실행
CMD ["python", "-u", "/app/handler.py"]
