"""
Z-Image Turbo RunPod Serverless Handler (S3 연동)
Version: 1.0.0

환경변수 필요:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_S3_BUCKET (기본값, input으로 덮어쓰기 가능)
- AWS_REGION (기본: ap-northeast-2)
- HF_TOKEN (HuggingFace 토큰, 선택)

API 요청 예시:
{
    "input": {
        "prompt": "A beautiful sunset over mountains",
        "height": 1024,
        "width": 1024,
        "seed": 42,
        "s3_bucket": "life-vision-dev",
        "s3_folder": "generated-images"
    }
}

응답 예시:
{
    "image_url": "https://life-vision-dev.s3.ap-northeast-2.amazonaws.com/generated-images/2025-01-25/abc123.png",
    "prompt": "A beautiful sunset over mountains",
    "seed": 42,
    "resolution": "1024x1024"
}
"""

import runpod
import torch
import boto3
import os
import uuid
from io import BytesIO
from datetime import datetime

# HuggingFace 토큰 설정
hf_token = os.environ.get('HF_TOKEN')
if hf_token:
    os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
    os.environ['HF_HUB_TOKEN'] = hf_token
    try:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        print("HuggingFace 로그인 완료")
    except Exception as e:
        print(f"HuggingFace 로그인 실패: {e}")

# 전역 모델 변수 (콜드 스타트 최적화)
pipe = None

# S3 클라이언트 초기화
s3_client = None


def get_s3_client():
    """S3 클라이언트 초기화 (싱글톤)"""
    global s3_client
    if s3_client is None:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name=os.environ.get('AWS_REGION', 'ap-northeast-2')
        )
    return s3_client


def get_model():
    """Z-Image Turbo 모델 로드 (싱글톤)"""
    global pipe
    if pipe is None:
        from diffusers import ZImagePipeline

        print("Z-Image Turbo 모델 로딩 중...")
        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        pipe.to("cuda")

        # Flash Attention 활성화 (가능한 경우)
        try:
            pipe.transformer.set_attention_backend("flash")
            print("Flash Attention 활성화됨")
        except Exception as e:
            print(f"Flash Attention 사용 불가: {e}")

        print("Z-Image Turbo 모델 로드 완료")
    return pipe


def upload_to_s3(image_buffer: BytesIO, bucket: str, folder: str = "generated-images") -> str:
    """
    이미지를 S3에 업로드하고 URL 반환

    Args:
        image_buffer: 이미지 데이터 버퍼
        bucket: S3 버킷 이름
        folder: S3 폴더 경로 (기본: generated-images)
    """
    client = get_s3_client()

    # 파일명 생성 (폴더/날짜/UUID)
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"{folder}/{today}/{uuid.uuid4().hex}.png"

    # S3 업로드
    image_buffer.seek(0)
    client.upload_fileobj(
        image_buffer,
        bucket,
        filename,
        ExtraArgs={'ContentType': 'image/png'}
    )

    # URL 생성
    region = os.environ.get('AWS_REGION', 'ap-northeast-2')
    url = f"https://{bucket}.s3.{region}.amazonaws.com/{filename}"

    print(f"S3 업로드: bucket={bucket}, folder={folder}, file={filename}")

    return url


def handler(job):
    """
    RunPod Serverless Handler

    입력:
        - prompt: 이미지 생성 프롬프트 (필수)
        - height: 이미지 높이 (기본: 1024, 16의 배수)
        - width: 이미지 너비 (기본: 1024, 16의 배수)
        - seed: 시드값 (기본: 랜덤)
        - num_inference_steps: 추론 스텝 수 (기본: 9, Turbo는 8-9 권장)
        - s3_bucket: S3 버킷명 (선택)
        - s3_folder: S3 폴더 경로 (선택)

    출력:
        - image_url: 생성된 이미지 S3 URL
        - prompt: 입력 프롬프트
        - seed: 사용된 시드
        - resolution: 해상도
    """
    try:
        job_input = job["input"]

        # 필수 파라미터
        prompt = job_input.get("prompt")
        if not prompt:
            return {"error": "prompt 파라미터가 필요합니다."}

        # 선택 파라미터
        height = job_input.get("height", 1024)
        width = job_input.get("width", 1024)
        seed = job_input.get("seed")
        num_inference_steps = job_input.get("num_inference_steps", 9)

        # 해상도 검증 (16의 배수여야 함)
        if height % 16 != 0:
            height = (height // 16) * 16
            print(f"height를 16의 배수로 조정: {height}")
        if width % 16 != 0:
            width = (width // 16) * 16
            print(f"width를 16의 배수로 조정: {width}")

        # S3 설정
        bucket = job_input.get("s3_bucket") or os.environ.get('AWS_S3_BUCKET')
        folder = job_input.get("s3_folder") or "generated-images"

        if not bucket:
            return {"error": "s3_bucket 파라미터 또는 AWS_S3_BUCKET 환경변수가 필요합니다."}

        # 시드 처리
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()

        print(f"이미지 생성 시작: prompt='{prompt[:50]}...', {width}x{height}, seed={seed}")

        # 모델 로드
        model = get_model()

        # 이미지 생성
        generator = torch.Generator("cuda").manual_seed(seed)

        image = model(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,  # Turbo 모델은 CFG 불필요
            generator=generator,
        ).images[0]

        print("이미지 생성 완료")

        # 이미지를 버퍼에 저장
        buffer = BytesIO()
        image.save(buffer, format="PNG")

        # S3에 업로드
        image_url = upload_to_s3(buffer, bucket, folder)
        print(f"S3 업로드 완료: {image_url}")

        return {
            "image_url": image_url,
            "prompt": prompt,
            "seed": seed,
            "resolution": f"{width}x{height}"
        }

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"에러 발생: {error_trace}")
        return {"error": str(e)}


# RunPod Serverless 시작
runpod.serverless.start({"handler": handler})
