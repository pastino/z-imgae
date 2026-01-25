#!/usr/bin/env python3
"""
Z-Image RunPod Serverless 테스트 스크립트

사용법:
    python test_runpod.py --endpoint YOUR_ENDPOINT_ID --prompt "A cat sitting on a chair"

환경변수 필요:
    RUNPOD_API_KEY: RunPod API 키
"""

import argparse
import os
import time
import requests


def test_zimage(endpoint_id: str, prompt: str, height: int = 1024, width: int = 1024, seed: int = None):
    """Z-Image RunPod 엔드포인트 테스트"""

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise ValueError("RUNPOD_API_KEY 환경변수가 설정되지 않았습니다.")

    # 요청 페이로드
    payload = {
        "input": {
            "prompt": prompt,
            "height": height,
            "width": width,
            "s3_bucket": os.environ.get("AWS_S3_BUCKET", "life-vision-dev"),
            "s3_folder": "test-images"
        }
    }

    if seed is not None:
        payload["input"]["seed"] = seed

    # API 호출
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    print(f"요청 URL: {url}")
    print(f"프롬프트: {prompt}")
    print(f"해상도: {width}x{height}")
    print("-" * 50)

    start_time = time.time()
    response = requests.post(url, json=payload, headers=headers, timeout=300)
    elapsed = time.time() - start_time

    print(f"응답 시간: {elapsed:.2f}초")
    print(f"상태 코드: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        if "output" in result:
            output = result["output"]
            if "error" in output:
                print(f"❌ 에러: {output['error']}")
            else:
                print(f"✅ 성공!")
                print(f"   이미지 URL: {output.get('image_url')}")
                print(f"   시드: {output.get('seed')}")
                print(f"   해상도: {output.get('resolution')}")
        else:
            print(f"응답: {result}")
    else:
        print(f"❌ 요청 실패: {response.text}")


def test_async(endpoint_id: str, prompt: str, height: int = 1024, width: int = 1024):
    """비동기 요청 테스트 (긴 작업용)"""

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise ValueError("RUNPOD_API_KEY 환경변수가 설정되지 않았습니다.")

    payload = {
        "input": {
            "prompt": prompt,
            "height": height,
            "width": width,
            "s3_bucket": os.environ.get("AWS_S3_BUCKET", "life-vision-dev"),
            "s3_folder": "test-images"
        }
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 비동기 요청 시작
    run_url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    print(f"비동기 요청 시작...")
    response = requests.post(run_url, json=payload, headers=headers)

    if response.status_code != 200:
        print(f"❌ 요청 실패: {response.text}")
        return

    result = response.json()
    job_id = result.get("id")
    print(f"Job ID: {job_id}")

    # 상태 폴링
    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    while True:
        time.sleep(2)
        status_response = requests.get(status_url, headers=headers)
        status_data = status_response.json()
        status = status_data.get("status")
        print(f"상태: {status}")

        if status == "COMPLETED":
            output = status_data.get("output", {})
            print(f"✅ 완료!")
            print(f"   이미지 URL: {output.get('image_url')}")
            break
        elif status == "FAILED":
            print(f"❌ 실패: {status_data.get('error')}")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Z-Image RunPod 테스트")
    parser.add_argument("--endpoint", required=True, help="RunPod 엔드포인트 ID")
    parser.add_argument("--prompt", default="A beautiful sunset over mountains, photorealistic, 8k",
                        help="이미지 생성 프롬프트")
    parser.add_argument("--height", type=int, default=1024, help="이미지 높이")
    parser.add_argument("--width", type=int, default=1024, help="이미지 너비")
    parser.add_argument("--seed", type=int, default=None, help="시드값")
    parser.add_argument("--async-mode", action="store_true", help="비동기 모드 사용")

    args = parser.parse_args()

    if args.async_mode:
        test_async(args.endpoint, args.prompt, args.height, args.width)
    else:
        test_zimage(args.endpoint, args.prompt, args.height, args.width, args.seed)
