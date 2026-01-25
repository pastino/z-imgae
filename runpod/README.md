# Z-Image Turbo RunPod Serverless

Z-Image Turbo 모델을 RunPod Serverless로 배포하는 설정입니다.

## 특징

- **8 steps** 초고속 생성 (Turbo 모델)
- **16GB VRAM**에서 동작
- S3 직접 업로드 (URL 반환)
- 콜드 스타트 최적화

## 빠른 시작

### 1. Docker 이미지 빌드 & 푸시

```bash
# Docker Hub 로그인
docker login

# 이미지 빌드
docker build -t yourusername/zimage-turbo:latest .

# 푸시
docker push yourusername/zimage-turbo:latest
```

### 2. RunPod 템플릿 생성

RunPod 콘솔에서:

1. **Serverless** → **My Templates** → **New Template**
2. 설정:
   - **Container Image**: `yourusername/zimage-turbo:latest`
   - **Container Disk**: 20GB (모델 캐시용)
   - **GPU**: RTX 4090 / A100 / H100 권장

### 3. 환경변수 설정

| 변수명 | 설명 | 필수 |
|--------|------|------|
| `AWS_ACCESS_KEY_ID` | AWS 액세스 키 | ✅ |
| `AWS_SECRET_ACCESS_KEY` | AWS 시크릿 키 | ✅ |
| `AWS_S3_BUCKET` | 기본 S3 버킷 | ✅ |
| `AWS_REGION` | AWS 리전 (기본: ap-northeast-2) | |
| `HF_TOKEN` | HuggingFace 토큰 | |

### 4. 엔드포인트 생성

1. **Serverless** → **My Endpoints** → **New Endpoint**
2. 생성한 템플릿 선택
3. **Workers**: Min 0, Max 1+ 설정
4. **GPU**: 최소 16GB VRAM

## API 사용법

### 요청

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A beautiful sunset over mountains, photorealistic",
      "height": 1024,
      "width": 1024,
      "seed": 42,
      "s3_bucket": "your-bucket",
      "s3_folder": "generated-images"
    }
  }'
```

### 응답

```json
{
  "image_url": "https://your-bucket.s3.ap-northeast-2.amazonaws.com/generated-images/2025-01-25/abc123.png",
  "prompt": "A beautiful sunset over mountains, photorealistic",
  "seed": 42,
  "resolution": "1024x1024"
}
```

### 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `prompt` | string | - | 이미지 생성 프롬프트 (필수) |
| `height` | int | 1024 | 이미지 높이 (16의 배수) |
| `width` | int | 1024 | 이미지 너비 (16의 배수) |
| `seed` | int | random | 시드값 |
| `num_inference_steps` | int | 9 | 추론 스텝 (8-9 권장) |
| `s3_bucket` | string | 환경변수 | S3 버킷명 |
| `s3_folder` | string | generated-images | S3 폴더 경로 |

## 테스트

```bash
# 환경변수 설정
export RUNPOD_API_KEY="your_api_key"
export AWS_S3_BUCKET="your_bucket"

# 테스트 실행
python test_runpod.py --endpoint YOUR_ENDPOINT_ID \
  --prompt "A cat sitting on a colorful chair, studio lighting"
```

## 지원 해상도

Z-Image Turbo는 다양한 해상도를 지원합니다:

| 비율 | 해상도 |
|------|--------|
| 1:1 | 1024x1024 |
| 16:9 | 1280x720, 1920x1088 |
| 9:16 | 720x1280, 1088x1920 |
| 4:3 | 1024x768 |
| 3:4 | 768x1024 |

> 모든 해상도는 16의 배수여야 합니다.

## 콜드 스타트 최적화

모델 사전 다운로드로 콜드 스타트 시간을 단축하려면:

```dockerfile
# Dockerfile에서 주석 해제
RUN python -c "from diffusers import ZImagePipeline; ZImagePipeline.from_pretrained('Tongyi-MAI/Z-Image-Turbo')"
```

> 이미지 크기가 ~15GB 증가하지만 콜드 스타트가 빨라집니다.

## 예상 성능

| GPU | 생성 시간 (1024x1024) |
|-----|----------------------|
| H100 | ~1초 |
| A100 | ~2초 |
| RTX 4090 | ~3초 |
| RTX 3090 | ~5초 |
