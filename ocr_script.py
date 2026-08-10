import os
import sys
import cv2
import io
from PIL import Image
import imagehash
import ocr_util

sys.stdout.reconfigure(encoding="utf-8")


def extract_frame_as_bytes_and_phash(video_capture, timestamp_seconds):
    """Extract a frame from video, return JPEG bytes and pHash"""
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_number = int(timestamp_seconds * fps)
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video_capture.read()
    if not ret:
        return None, None

    # Convert BGR (OpenCV) to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Compute pHash
    phash = imagehash.phash(pil_image)

    # Save to in-memory bytes as JPEG
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    return img_byte_arr.getvalue(), phash


def process_video(
    video_path,
    output_dir,
    model_name="qwen2.5vl:7b",
    interval_seconds=5,
    max_test_seconds=None,
    phash_threshold=4,
    max_ocr_workers=1,
):
    """
    Process a single video file
    - Extract frames every interval_seconds seconds
    - Use pHash to skip similar frames
    - Process new frames immediately (no buffer, sequential)
    - max_test_seconds: only process this many seconds for testing (set to None for full video)
    - phash_threshold: pHash difference threshold, below this means frames are similar (smaller = stricter)
    """
    print(f"开始处理视频: {video_path}")
    print(f"OCR 后端: {ocr_util.OCR_BACKEND}, model={model_name}")

    # Write beside the source video, using exactly the same basename.
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file_path = os.path.join(
        os.path.dirname(os.path.abspath(video_path)), f"{video_name}.txt"
    )

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: 无法打开视频文件 {video_path}")
        return

    # Create the file before OCR starts, so progress is visible immediately.
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(f"视频文件名: {video_name}\n")
        f.write("OCR 处理中；结果会按帧实时追加。\n")
        # Keep the file present but empty until the first timestamped result.
        f.seek(0)
        f.truncate(0)
        f.flush()
        os.fsync(f.fileno())
    print(f"结果文件（实时写入）: {output_file_path}")

    # Get video duration
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # Limit duration for testing
    if max_test_seconds is not None:
        duration = min(duration, max_test_seconds)
        print(f"测试模式: 只处理前 {max_test_seconds} 秒")

    # 计算预计处理的帧数
    expected_frame_count = int(duration // interval_seconds) + 1

    print(f"视频信息: {total_frames} 帧, {fps:.2f} FPS, 处理时长: {duration:.2f}s")
    print(f"预计处理帧数: {expected_frame_count} (每 {interval_seconds} 秒1帧)")
    print(f"pHash 阈值: {phash_threshold} (差异小于此值视为相同画面)")

    # Initialize variables
    current_time = 0
    frame_idx = 1  # 当前处理的总帧序号（不管是跳过还是处理）
    processed_frame_count = 0
    skipped_frame_count = 0
    extract_failed_count = 0
    ocr_failed_count = 0
    total_ocr_time = 0
    actual_ocr_count = 0
    average_ocr_time = 0
    BATCH_SIZE = 8  # Define batch size for concurrent processing
    batch_queue = []  # Initialize batch queue

    processed_frames_cache = (
        {}
    )  # Key: imagehash.phash object, Value: OCR result string.

    ocr_prompt = """
        You are an extremely precise image-to-text transcription engine.

【CORE TASK】
Transcribe (OCR) every readable piece of text, button label, code snippet, menu item, and number visible on the screen image verbatim. 

【STRICT CONSTRAINTS】
1. Transcribe Only: Read from top-to-bottom and left-to-right. Copy exactly what you see.
2. NO Summaries: Do NOT explain what the screen is for, do NOT summarize the UI layout, and do NOT outline the general meaning.
3. NO Extensions: Do NOT add background knowledge, do NOT explain terms, and do NOT offer suggestions or fixes for errors shown on screen.
4. Keep Formatting: Preserve original code blocks, tables, and indentation/line breaks as closely as possible.

【OUTPUT REQUIREMENT】
If no text is detected on the screen, reply with "No text detected." 
Otherwise, output the transcribed text directly. Do NOT include any introductory greetings (e.g., "Here is the transcription:") or closing remarks. Jump straight into the transcribed text.
        """

    def make_result(timestamp, content, ocr_time=0, input_tokens=0, output_tokens=0):
        return {
            "timestamp": timestamp,
            "content": content,
            "time": ocr_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def flush_batch():
        """处理并写入当前 batch，保证时间戳顺序。"""
        nonlocal batch_queue, processed_frame_count, ocr_failed_count
        nonlocal total_ocr_time, actual_ocr_count, average_ocr_time

        if not batch_queue:
            return

        print(
            f"DEBUG: [处理] 批量 OCR {len(batch_queue)} 帧，开始 OCR 处理"
        )
        batch_input = [(ts, img_bytes) for ts, img_bytes, _phash in batch_queue]
        results = ocr_util.process_images_batch(
            batch_input, model_name, ocr_prompt, max_workers=max_ocr_workers
        )

        # results 已按 timestamp 排序，与 batch_queue 时间顺序一致
        for i, result in enumerate(results):
            total_ocr_time += result["time"]
            actual_ocr_count += 1
            average_ocr_time = (
                total_ocr_time / actual_ocr_count if actual_ocr_count > 0 else 0
            )
            processed_frames_cache[batch_queue[i][2]] = result["content"]
            if result["content"] == ocr_util.FAILED_OCR_CONTENT:
                ocr_failed_count += 1

        processed_frame_count += len(batch_queue)
        ocr_util.append_results_to_file(output_file_path, results, average_ocr_time)
        batch_queue = []

    while current_time <= duration:
        # Extract frame and pHash
        frame_bytes, current_phash = extract_frame_as_bytes_and_phash(cap, current_time)

        if not frame_bytes or not current_phash:
            print(f"WARNING: 无法提取帧，时间: {current_time:.2f}s，写入「识别失败」")
            # 先刷掉更早的 batch，再写入当前秒，保证 txt 时间序
            flush_batch()
            extract_failed_count += 1
            ocr_util.append_results_to_file(
                output_file_path,
                [make_result(current_time, ocr_util.FAILED_OCR_CONTENT)],
                average_ocr_time,
            )
            frame_idx += 1
            current_time += interval_seconds
            continue

        # Check if frame is similar to any previously processed frame
        is_similar = False
        reused_ocr_result = None

        for stored_phash, stored_ocr_content in processed_frames_cache.items():
            diff = current_phash - stored_phash
            if diff <= phash_threshold:
                is_similar = True
                reused_ocr_result = stored_ocr_content
                break

        if is_similar:
            # 重复画面：不调模型，但仍把复用结果写入对应秒
            skipped_frame_count += 1
            print(
                f"DEBUG: [跳过] 帧 {frame_idx}/{expected_frame_count} "
                f"(相似画面 pHash: {current_phash})，复用结果并写入"
            )
            flush_batch()
            ocr_util.append_results_to_file(
                output_file_path,
                [make_result(current_time, reused_ocr_result)],
                average_ocr_time,
            )
        else:
            print(
                f"DEBUG: [排队] 帧 {frame_idx}/{expected_frame_count} "
                f"(新画面 pHash: {current_phash})"
            )
            batch_queue.append((current_time, frame_bytes, current_phash))

            if (
                len(batch_queue) >= BATCH_SIZE
                or current_time + interval_seconds > duration
            ):
                flush_batch()

        frame_idx += 1
        current_time += interval_seconds

    # Process any remaining frames in the batch_queue
    flush_batch()

    # Print summary
    total_written = processed_frame_count + skipped_frame_count + extract_failed_count
    summary = [
        "=" * 60,
        "视频处理统计",
        "=" * 60,
        f"视频文件名: {video_name}",
        f"OCR 处理帧数: {processed_frame_count}",
        f"跳过(复用)帧数: {skipped_frame_count}",
        f"抽帧失败帧数: {extract_failed_count}",
        f"OCR 识别失败帧数: {ocr_failed_count}",
        f"总写入帧数: {total_written}",
        "=" * 60,
    ]
    print("\n".join(summary))

    with open(output_file_path, "a", encoding="utf-8") as f:
        for line in summary:
            f.write(line + "\n")

    # Release video capture
    cap.release()
    print(f"结果文件: {output_file_path}")


def main():
    import time

    start_time = time.time()

    # Define paths
    data_dir = os.path.join(os.getcwd(), "data", "lectures")
    output_dir = os.path.join(os.getcwd(), "ocr_results")

    print(f"当前工作目录: {os.getcwd()}")
    print(f"数据目录: {data_dir}")
    print(f"数据目录存在: {os.path.exists(data_dir)}")

    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"ERROR: 数据目录不存在: {data_dir}")
        return

    # Find all video files
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv"]
    video_files = []
    for root, dirs, files in os.walk(data_dir):
        print(f"扫描目录: {root}, 文件: {files}")
        for file in files:
            if os.path.splitext(file.lower())[1] in video_extensions:
                video_files.append(os.path.join(root, file))

    if not video_files:
        print(f"WARNING: 在 {data_dir} 中未找到视频文件")
        return

    print(f"找到 {len(video_files)} 个视频文件: {video_files}")

    # Collect all output file paths
    output_files = []

    # Process each video
    for video_path in video_files:
        # Get output file path
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_file = os.path.join(
            os.path.dirname(os.path.abspath(video_path)), f"{video_name}.txt"
        )
        output_files.append(output_file)

        # 默认走 Ollama（OCR_BACKEND=ollama）。若用 vLLM：
        #   OCR_BACKEND=vllm VLLM_BASE_URL=http://localhost:8000
        #   model_name="RedHatAI/Qwen2.5-VL-7B-Instruct-FP8-Dynamic"
        process_video(
            video_path,
            output_dir,
            model_name=os.environ.get("OCR_MODEL", "qwen3-vl:32b"),
            interval_seconds=1,
            max_test_seconds=None,  # Process full video
            max_ocr_workers=int(os.environ.get("OCR_MAX_WORKERS", "1")),
        )

    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60

    # Print total time
    print("=" * 60)
    print(f"所有视频处理完成！总耗时: {hours:02d}h {minutes:02d}m {seconds:.2f}s")
    print("=" * 60)

    # Prepend total time summary to all output files
    total_time_summary = [
        "=" * 60,
        "总处理时间",
        "=" * 60,
        f"总耗时: {hours:02d}h {minutes:02d}m {seconds:.2f}s",
        "=" * 60,
        "\n",
    ]

    for output_file in output_files:
        # Read existing content
        with open(output_file, "r", encoding="utf-8") as f:
            existing_content = f.read()

        # Write total time summary first, then existing content
        with open(output_file, "w", encoding="utf-8") as f:
            for line in total_time_summary:
                f.write(line + "\n")
            f.write(existing_content)


if __name__ == "__main__":
    main()
