import time
import sys
import os
import requests
import base64
import concurrent.futures

sys.stdout.reconfigure(encoding='utf-8')

# 后端: ollama | vllm，可通过环境变量 OCR_BACKEND 切换
OCR_BACKEND = os.environ.get("OCR_BACKEND", "ollama").lower().strip()

# Ollama 服务地址（原生 /api/chat）
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

# vLLM OpenAI 兼容服务地址
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000").rstrip("/")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")


class OCRConnectionError(Exception):
    """OCR 后端服务连接/调用失败时抛出"""
    pass


# 兼容旧名称
VLLMConnectionError = OCRConnectionError


def _preprocess_image(image_bytes):
    """压缩图像，兼顾速度与清晰度（课件文字需要可读）"""
    from PIL import Image
    import io

    preprocess_start_time = time.time()
    img = Image.open(io.BytesIO(image_bytes))
    # Lecture slides remain readable at this size, while the reduced visual
    # token count makes prefill noticeably faster.
    max_dim = 1280
    if img.width > max_dim or img.height > max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    compressed_img_bytes = io.BytesIO()
    img.save(compressed_img_bytes, format='JPEG', quality=70, optimize=True)
    compressed_img_bytes.seek(0)
    image_bytes = compressed_img_bytes.getvalue()
    print(f"DEBUG: image preprocess: {time.time() - preprocess_start_time:.2f}s")
    print(f"DEBUG: 图像压缩后大小: {len(image_bytes)/1024:.2f} KB, 分辨率: {img.width}x{img.height}")
    return image_bytes


def call_ollama_model(image_bytes, model_name, prompt_text, timeout=300):
    """调用 Ollama 原生接口 (/api/chat)，传入图片 bytes（内存中）"""
    image_bytes = _preprocess_image(image_bytes)
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    api_url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt_text,
                "images": [image_base64],
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            # Preserve dense English slide text without cutting the response short.
            "num_predict": 1536,
        },
    }

    print(f"DEBUG: 发送请求到 Ollama ({api_url}), model={model_name}")

    request_start_time = time.time()
    try:
        response = requests.post(api_url, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as e:
        error_msg = (
            f"连接 Ollama 服务失败 (连接错误): {e}\n"
            f"请确保 Ollama 正在运行，并且可以访问 {api_url}\n"
            f"例如: ollama serve && ollama pull {model_name}"
        )
        print(f"ERROR: {error_msg}")
        raise OCRConnectionError(error_msg) from e
    except requests.exceptions.Timeout as e:
        error_msg = f"连接 Ollama 服务超时: {e}\n请检查 Ollama 服务是否正常运行"
        print(f"ERROR: {error_msg}")
        raise OCRConnectionError(error_msg) from e
    except requests.exceptions.RequestException as e:
        error_msg = f"连接 Ollama 服务失败 (请求错误): {e}"
        print(f"ERROR: {error_msg}")
        raise OCRConnectionError(error_msg) from e

    if not response.ok:
        error_msg = (
            f"Ollama 服务返回错误状态码: {response.status_code}\n"
            f"响应内容: {response.text[:1000]}"
        )
        print(f"ERROR: {error_msg}")
        raise OCRConnectionError(error_msg)

    request_time = time.time() - request_start_time
    print(f"DEBUG: 请求完成，耗时: {request_time:.2f}s")

    try:
        result = response.json()
        content = result["message"]["content"]
        prompt_tokens = result.get("prompt_eval_count", 0) or 0
        completion_tokens = result.get("eval_count", 0) or 0
        return content.strip(), prompt_tokens, completion_tokens
    except (KeyError, IndexError, ValueError, TypeError) as e:
        error_msg = f"解析 Ollama 响应失败: {e}\n原始响应: {response.text[:1000]}"
        print(f"ERROR: {error_msg}")
        raise OCRConnectionError(error_msg) from e


def call_vllm_model(image_bytes, model_name, prompt_text, timeout=300):
    """调用 vLLM OpenAI 兼容接口 (/v1/chat/completions)，传入图片 bytes（内存中）"""
    image_bytes = _preprocess_image(image_bytes)
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    image_data_url = f"data:image/jpeg;base64,{image_base64}"

    api_url = f"{VLLM_BASE_URL}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {VLLM_API_KEY}",
    }
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        "temperature": 0.0,
        # Preserve dense English slide text without cutting the response short.
        "max_tokens": 1536,
    }

    print(f"DEBUG: 发送请求到 vLLM ({api_url}), model={model_name}")

    request_start_time = time.time()
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as e:
        error_msg = (
            f"连接 vLLM 服务失败 (连接错误): {e}\n"
            f"请确保 vLLM 服务正在运行，并且可以访问 {api_url}"
        )
        print(f"ERROR: {error_msg}")
        raise OCRConnectionError(error_msg) from e
    except requests.exceptions.Timeout as e:
        error_msg = f"连接 vLLM 服务超时 (超时错误): {e}\n请求超时，请检查 vLLM 服务是否正常运行"
        print(f"ERROR: {error_msg}")
        raise OCRConnectionError(error_msg) from e
    except requests.exceptions.RequestException as e:
        error_msg = f"连接 vLLM 服务失败 (请求错误): {e}"
        print(f"ERROR: {error_msg}")
        raise OCRConnectionError(error_msg) from e

    if not response.ok:
        error_msg = (
            f"vLLM 服务返回错误状态码: {response.status_code}\n"
            f"响应内容: {response.text[:1000]}"
        )
        print(f"ERROR: {error_msg}")
        raise OCRConnectionError(error_msg)

    request_time = time.time() - request_start_time
    print(f"DEBUG: 请求完成，耗时: {request_time:.2f}s")

    try:
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        prompt_tokens = result.get("usage", {}).get("prompt_tokens", 0)
        completion_tokens = result.get("usage", {}).get("completion_tokens", 0)
        return content.strip(), prompt_tokens, completion_tokens
    except (KeyError, IndexError, ValueError) as e:
        error_msg = f"解析 vLLM 响应失败: {e}\n原始响应: {response.text[:1000]}"
        print(f"ERROR: {error_msg}")
        raise OCRConnectionError(error_msg) from e


def call_vision_model(image_bytes, model_name, prompt_text, timeout=300):
    """按 OCR_BACKEND 分发到 Ollama 或 vLLM"""
    if OCR_BACKEND == "vllm":
        return call_vllm_model(image_bytes, model_name, prompt_text, timeout=timeout)
    if OCR_BACKEND == "ollama":
        return call_ollama_model(image_bytes, model_name, prompt_text, timeout=timeout)
    raise OCRConnectionError(
        f"未知 OCR_BACKEND={OCR_BACKEND!r}，请设置为 'ollama' 或 'vllm'"
    )


def process_single_image(image_bytes, model_name, prompt_text):
    """处理单张图片（内存中的 bytes），失败时异常会向上抛出"""
    result = {'content': '', 'time': 0, 'input_tokens': 0, 'output_tokens': 0}
    start_time = time.time()

    content, prompt_tokens, completion_tokens = call_vision_model(
        image_bytes, model_name, prompt_text
    )
    result['content'] = content
    result['input_tokens'] = prompt_tokens
    result['output_tokens'] = completion_tokens
    result['time'] = time.time() - start_time

    return result


def process_images_batch(image_tuples, model_name, prompt_text, max_workers=6):
    """
    批量处理图片（内存中）
    image_tuples: [(timestamp_seconds, image_bytes), ...]
    任意一张图片调用失败（含连接失败）都会中断整个批次并抛出 OCRConnectionError。
    """
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {
            executor.submit(process_single_image, img_bytes, model_name, prompt_text): ts
            for ts, img_bytes in image_tuples
        }
        for future in concurrent.futures.as_completed(future_to_image):
            ts = future_to_image[future]
            # 不吞异常：直接让 OCRConnectionError / 其他异常向上抛出
            result_data = future.result()
            results.append({
                'timestamp': ts,
                'content': result_data['content'],
                'time': result_data['time'],
                'input_tokens': result_data.get('input_tokens', 0),
                'output_tokens': result_data.get('output_tokens', 0),
            })

    results.sort(key=lambda x: x['timestamp'])
    return results


def append_results_to_file(output_file_path, results, average_ocr_time=0):
    """将 OCR 结果（带时间戳）追加写入输出文件"""
    with open(output_file_path, "a", encoding="utf-8") as f:
        for result in results:
            hours = int(result['timestamp'] // 3600)
            minutes = int((result['timestamp'] % 3600) // 60)
            seconds = int(result['timestamp'] % 60)
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            f.write(f"\n{'='*60}\n")
            f.write(
                f"时间戳: {time_str} ({result['timestamp']:.2f}s), "
                f"OCR耗时: {result['time']:.2f}s, 平均耗时: {average_ocr_time:.2f}s, "
                f"InputTokens: {result.get('input_tokens', 0)}, "
                f"OutputTokens: {result.get('output_tokens', 0)}\n"
            )
            f.write(f"{'='*60}\n")
            f.write(result['content'])
            f.write("\n")
        # 让长视频任务的最新结果立刻对其他进程/文件管理器可见，
        # 而不是等到缓冲区关闭或整个视频处理结束。
        f.flush()
        os.fsync(f.fileno())
