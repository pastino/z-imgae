"""
Illustrious XL RunPod Serverless Handler (S3 연동)
Version: 3.0.0

애니메이션 특화 이미지 생성 (Illustrious XL v1.0)

환경변수 필요:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_S3_BUCKET (기본값, input으로 덮어쓰기 가능)
- AWS_REGION (기본: ap-northeast-2)
- HF_TOKEN (HuggingFace 토큰, 선택)

API 요청 예시 (txt2img):
{
    "input": {
        "prompt": "1girl, silver hair, blue eyes, fantasy, masterpiece",
        "negative_prompt": "worst quality, low quality, blurry",
        "height": 1024,
        "width": 768,
        "seed": 42,
        "s3_bucket": "glitch-prod",
        "s3_folder": "generated-images"
    }
}
"""

import runpod
import torch
import boto3
import os
import uuid
import requests
from io import BytesIO
from datetime import datetime
from PIL import Image

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
pipe_txt2img = None
pipe_img2img = None

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


def get_txt2img_pipe():
    """Illustrious XL txt2img 파이프라인 로드 (싱글톤)"""
    global pipe_txt2img
    if pipe_txt2img is None:
        from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

        model_id = "Liberata/illustrious-xl-v1.0"
        print(f"Illustrious XL 모델 로딩 중... ({model_id})")

        pipe_txt2img = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )

        # Euler Ancestral 스케줄러 (Illustrious 권장)
        pipe_txt2img.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe_txt2img.scheduler.config
        )

        pipe_txt2img.to("cuda")

        # xformers 메모리 최적화 (가능한 경우)
        try:
            pipe_txt2img.enable_xformers_memory_efficient_attention()
            print("xformers 메모리 최적화 활성화됨")
        except Exception as e:
            print(f"xformers 사용 불가 (정상 동작에 영향 없음): {e}")

        print(f"Illustrious XL 모델 로드 완료")
    return pipe_txt2img


def get_img2img_pipe():
    """Illustrious XL img2img 파이프라인 로드 (싱글톤, txt2img 컴포넌트 재활용)"""
    global pipe_img2img
    if pipe_img2img is None:
        from diffusers import StableDiffusionXLImg2ImgPipeline

        # txt2img 파이프라인이 이미 로드되어 있으면 컴포넌트 재활용
        txt2img = get_txt2img_pipe()

        print("Illustrious XL (img2img) 파이프라인 구성 중...")
        pipe_img2img = StableDiffusionXLImg2ImgPipeline(
            vae=txt2img.vae,
            text_encoder=txt2img.text_encoder,
            text_encoder_2=txt2img.text_encoder_2,
            tokenizer=txt2img.tokenizer,
            tokenizer_2=txt2img.tokenizer_2,
            unet=txt2img.unet,
            scheduler=txt2img.scheduler,
        )

        print("Illustrious XL (img2img) 파이프라인 준비 완료")
    return pipe_img2img


def download_image(url: str) -> Image.Image:
    """URL에서 이미지 다운로드"""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def upload_to_s3(image_buffer: BytesIO, bucket: str, folder: str = "generated-images") -> str:
    """이미지를 S3에 업로드하고 URL 반환"""
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
    RunPod Serverless Handler - Illustrious XL

    입력 (공통):
        - prompt: 이미지 생성 프롬프트 (필수)
        - negative_prompt: 네거티브 프롬프트 (기본: 품질 관련)
        - height: 이미지 높이 (기본: 1024, 8의 배수)
        - width: 이미지 너비 (기본: 768, 8의 배수)
        - seed: 시드값 (기본: 랜덤)
        - num_inference_steps: 추론 스텝 수 (기본: 28)
        - guidance_scale: CFG 스케일 (기본: 7.0)
        - s3_bucket: S3 버킷명
        - s3_folder: S3 폴더 경로

    입력 (img2img 전용):
        - image_url: 레퍼런스 이미지 URL
        - strength: 변환 강도 (기본: 0.6)
    """
    try:
        job_input = job["input"]

        # 필수 파라미터
        prompt = job_input.get("prompt")
        if not prompt:
            return {"error": "prompt 파라미터가 필요합니다."}

        # 공통 파라미터
        negative_prompt = job_input.get("negative_prompt",
            "worst quality, low quality, blurry, bad anatomy, bad hands, missing fingers, extra digits, fewer digits, watermark, signature, text")
        height = job_input.get("height", 1024)
        width = job_input.get("width", 768)
        seed = job_input.get("seed")
        num_inference_steps = job_input.get("num_inference_steps", 28)
        guidance_scale = job_input.get("guidance_scale", 7.0)

        # img2img 파라미터
        ref_image_url = job_input.get("image_url")
        strength = job_input.get("strength", 0.6)

        # 모드 결정
        mode = "img2img" if ref_image_url else "txt2img"

        # 해상도 검증 (8의 배수여야 함)
        if height % 8 != 0:
            height = (height // 8) * 8
            print(f"height를 8의 배수로 조정: {height}")
        if width % 8 != 0:
            width = (width // 8) * 8
            print(f"width를 8의 배수로 조정: {width}")

        # S3 설정
        bucket = job_input.get("s3_bucket") or os.environ.get('AWS_S3_BUCKET')
        folder = job_input.get("s3_folder") or "generated-images"

        if not bucket:
            return {"error": "s3_bucket 파라미터 또는 AWS_S3_BUCKET 환경변수가 필요합니다."}

        # 시드 처리
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()

        print(f"[{mode}] 이미지 생성 시작: prompt='{prompt[:50]}...', {width}x{height}, seed={seed}, steps={num_inference_steps}, cfg={guidance_scale}")

        # 시드 생성기
        generator = torch.Generator("cuda").manual_seed(seed)

        if mode == "img2img":
            print(f"  레퍼런스 이미지 다운로드: {ref_image_url[:80]}...")
            init_image = download_image(ref_image_url)
            init_image = init_image.resize((width, height))
            print(f"  strength: {strength}")

            model = get_img2img_pipe()
            image = model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=strength,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
        else:
            model = get_txt2img_pipe()
            image = model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]

        print(f"[{mode}] 이미지 생성 완료")

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
            "resolution": f"{width}x{height}",
            "mode": mode,
            "model": "illustrious-xl-v1.0",
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
        }

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"에러 발생: {error_trace}")
        return {"error": str(e)}


# RunPod Serverless 시작
runpod.serverless.start({"handler": handler})
