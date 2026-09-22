import os
import re
import sys
import cv2
import io
from PIL import Image
import imagehash
import ocr_util

sys.stdout.reconfigure(encoding="utf-8")

# 进行中写到 *.ocr.tmp；全部成功后才原子改名为 *.txt，避免半成品 txt
PARTIAL_SUFFIX = ".ocr.tmp"
COMPLETE_MARKER = "视频处理统计"
TIMESTAMP_LINE_RE = re.compile(
    r"^时间戳:\s*(\d{2}:\d{2}:\d{2})\s*\((\d+(?:\.\d+)?)s\)",
    re.MULTILINE,
)


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


def parse_time_to_seconds(value):
    """将秒数或 'HH:MM:SS' / 'MM:SS' 字符串转为浮点秒。"""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        parts = text.split(":")
        try:
            if len(parts) == 3:
                hours, minutes, seconds = parts
                return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
            if len(parts) == 2:
                minutes, seconds = parts
                return int(minutes) * 60 + float(seconds)
            return float(text)
        except ValueError as e:
            raise ValueError(f"无法解析时间: {value!r}") from e
    raise ValueError(f"不支持的时间类型: {type(value).__name__} ({value!r})")


def normalize_time_ranges(time_ranges):
    """
    规范化时间段数组。
    入参示例: [(0, 60), ("10:00", "12:30"), [90, 120]]
    返回按起点排序的 [(start_sec, end_sec), ...]；None/空表示不限制。
    """
    if not time_ranges:
        return None

    normalized = []
    for idx, item in enumerate(time_ranges):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                f"time_ranges[{idx}] 必须是 [start, end] 或 (start, end)，实际为: {item!r}"
            )
        start = parse_time_to_seconds(item[0])
        end = parse_time_to_seconds(item[1])
        if start < 0 or end < 0:
            raise ValueError(f"time_ranges[{idx}] 时间不能为负数: {item!r}")
        if end < start:
            start, end = end, start
        normalized.append((start, end))

    normalized.sort(key=lambda pair: pair[0])
    return normalized


def format_time_ranges(time_ranges):
    """用于日志展示。"""
    if not time_ranges:
        return "整段视频"

    def fmt(sec):
        hours = int(sec // 3600)
        minutes = int((sec % 3600) // 60)
        seconds = sec % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
        return f"{minutes:02d}:{seconds:05.2f}"

    return ", ".join(f"[{fmt(start)} ~ {fmt(end)}]" for start, end in time_ranges)


def build_timestamps(duration, interval_seconds, time_ranges=None):
    """
    生成待处理时间戳列表。
    time_ranges 为 None/空时：从 0 到 duration，按 interval 取样。
    有时间段时：只在各 [start, end] 内按 interval 取样（含端点附近对齐的点）。
    """
    if interval_seconds <= 0:
        raise ValueError(f"interval_seconds 必须 > 0，实际为: {interval_seconds}")

    if duration <= 0:
        return []

    if not time_ranges:
        timestamps = []
        t = 0.0
        while t <= duration + 1e-9:
            timestamps.append(round(t, 6))
            t += interval_seconds
        return timestamps

    timestamps = []
    seen = set()
    for start, end in time_ranges:
        start = max(0.0, start)
        end = min(duration, end)
        if start > duration or end < 0 or start > end:
            continue

        t = start
        while t <= end + 1e-9:
            key = round(t, 3)
            if key not in seen and t <= duration + 1e-9:
                seen.add(key)
                timestamps.append(round(t, 6))
            t += interval_seconds

    timestamps.sort()
    return timestamps


def output_paths_for_video(video_path):
    """返回 (最终 txt 路径, 进行中 partial 路径)。"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.dirname(os.path.abspath(video_path))
    final_path = os.path.join(out_dir, f"{video_name}.txt")
    partial_path = os.path.join(out_dir, f"{video_name}{PARTIAL_SUFFIX}")
    return final_path, partial_path


def is_complete_result_file(path):
    """最终 txt 是否包含完成标记（避免把半成品当成已完成）。"""
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 8192), os.SEEK_SET)
            tail = f.read().decode("utf-8", errors="ignore")
        return COMPLETE_MARKER in tail
    except OSError:
        return False


def parse_partial_results(partial_path):
    """
    解析 partial 中已完整写入的时间戳结果。
    若文件末尾被截断，只保留最后一个完整块之前的内容，并回写清理后的文件。
    返回 [(timestamp_seconds, content), ...]
    """
    if not os.path.isfile(partial_path):
        return []

    with open(partial_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 按时间戳块切分；块头形如 =====\n时间戳: ...
    parts = re.split(r"\n?={60}\n(?=时间戳:)", text)
    done = []
    clean_chunks = []

    for part in parts:
        part = part.strip("\n")
        if not part:
            continue
        if COMPLETE_MARKER in part and "时间戳:" not in part.split("\n", 1)[0]:
            continue

        m = TIMESTAMP_LINE_RE.search(part)
        if not m:
            continue

        lines = part.split("\n")
        ts_line_idx = None
        for i, line in enumerate(lines):
            if TIMESTAMP_LINE_RE.match(line):
                ts_line_idx = i
                break
        if ts_line_idx is None:
            continue
        if ts_line_idx + 1 >= len(lines) or not re.fullmatch(r"={60}", lines[ts_line_idx + 1]):
            # 不完整块，丢弃及其之后
            break

        ts_sec = float(m.group(2))
        content = "\n".join(lines[ts_line_idx + 2 :]).strip("\n")
        done.append((ts_sec, content))

        hours = int(ts_sec // 3600)
        minutes = int((ts_sec % 3600) // 60)
        seconds = int(ts_sec % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        block = (
            f"\n{'=' * 60}\n"
            f"时间戳: {time_str} ({ts_sec:.2f}s)\n"
            f"{'=' * 60}\n"
            f"{content}\n"
        )
        clean_chunks.append(block)

    cleaned = "".join(clean_chunks)
    with open(partial_path, "w", encoding="utf-8") as f:
        f.write(cleaned)
        f.flush()
        os.fsync(f.fileno())

    return done


def finalize_output(partial_path, final_path):
    """全部成功后原子替换为最终 txt，并确保不留下半成品。"""
    os.replace(partial_path, final_path)


def process_video(
    video_path,
    output_dir,
    model_name,
    interval_seconds=5,
    max_test_seconds=None,
    phash_threshold=4,
    max_ocr_workers=1,
    time_ranges=None,
):
    """
    Process a single video file
    - Extract frames every interval_seconds seconds
    - Use pHash to skip similar frames
    - 进行中只写 *.ocr.tmp；全部完成后才原子改名为 *.txt
    - 中断后再次运行会从 partial 已完成的时间戳继续
    - time_ranges: 只处理这些时间段内的截图；None/[] 表示整段视频
    """
    print(f"开始处理视频: {video_path}")
    print(f"OCR 后端: {ocr_util.OCR_BACKEND}, model={model_name}")

    try:
        normalized_ranges = normalize_time_ranges(time_ranges)
    except ValueError as e:
        print(f"ERROR: time_ranges 无效: {e}")
        return False

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file_path, partial_path = output_paths_for_video(video_path)

    # 已有完整结果则直接跳过
    if is_complete_result_file(output_file_path):
        if os.path.exists(partial_path):
            try:
                os.remove(partial_path)
            except OSError:
                pass
        print(f"跳过（已有完整结果）: {output_file_path}")
        return True

    # 旧版半成品 .txt（无完成标记）迁移为 partial，便于续跑
    if os.path.isfile(output_file_path) and not is_complete_result_file(output_file_path):
        print(f"发现未完成的 txt，迁移为续跑文件: {output_file_path} -> {partial_path}")
        if os.path.exists(partial_path):
            try:
                os.remove(output_file_path)
            except OSError:
                pass
        else:
            os.replace(output_file_path, partial_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: 无法打开视频文件 {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    if max_test_seconds is not None:
        duration = min(duration, max_test_seconds)
        print(f"测试模式: 只处理前 {max_test_seconds} 秒")

    all_timestamps = build_timestamps(duration, interval_seconds, normalized_ranges)
    expected_frame_count = len(all_timestamps)

    print(f"视频信息: {total_frames} 帧, {fps:.2f} FPS, 视频时长: {duration:.2f}s")
    print(f"处理时间段: {format_time_ranges(normalized_ranges)}")
    print(f"预计处理帧数: {expected_frame_count} (每 {interval_seconds} 秒1帧)")
    print(f"pHash 阈值: {phash_threshold} (差异小于此值视为相同画面)")
    print(f"进行中文件: {partial_path}")
    print(f"完成后文件: {output_file_path}")

    if expected_frame_count == 0:
        print("WARNING: 没有可处理的时间点（时间段为空或超出视频时长）")
        cap.release()
        return False

    # 加载已完成进度并清理截断尾巴
    done_pairs = parse_partial_results(partial_path) if os.path.exists(partial_path) else []
    done_ts = {round(ts, 3) for ts, _ in done_pairs}
    timestamps = [t for t in all_timestamps if round(t, 3) not in done_ts]

    if done_pairs:
        print(f"断点续跑: 已完成 {len(done_pairs)} 帧，剩余 {len(timestamps)} 帧")
    else:
        with open(partial_path, "w", encoding="utf-8") as f:
            f.write("")
            f.flush()
            os.fsync(f.fileno())

    frame_idx = len(done_pairs) + 1
    processed_frame_count = 0
    skipped_frame_count = 0
    extract_failed_count = 0
    ocr_failed_count = 0
    total_ocr_time = 0
    actual_ocr_count = 0
    average_ocr_time = 0
    BATCH_SIZE = 8
    batch_queue = []

    processed_frames_cache = {}

    # 用已完成帧重建 pHash 缓存，保证续跑后仍能复用相似结果
    if done_pairs:
        print("重建已完成帧的 pHash 缓存…")
        for ts, content in done_pairs:
            _bytes, phash = extract_frame_as_bytes_and_phash(cap, ts)
            if phash is not None:
                processed_frames_cache[phash] = content

    ocr_prompt = """
        Extract useful content from this screenshot for a downstream LLM.
        Note: The diagram shows a reduction of the factorial program using formal semantics rules.

Rules:

1. Identify the screenshot type. Use one or more of: `slide`, `code`, `document`, `webpage`, `terminal`, `desktop`, `application`, `diagram`, `table`.

   * If multiple types apply, list them together, e.g. `Type: slide, code`.
   * Do not use `mixed`.

2. Ignore irrelevant UI such as desktop background, app icons, taskbars, browser bars, Zoom/Teams controls, window chrome, and unrelated UI.

3. Extract only meaningful visible content:

   * Titles and headings
   * Text and bullet points
   * Code
   * Formulas
   * Tables
   * Labels
   * Important diagram relationships

4. Preserve the original reading order and visual structure.
   Clearly label information blocks, for example:
   `Left:`, `Right:`, `Top:`, `Bottom:`, `Code:`, `Table:`, `Diagram:`

5. Do not explain, summarize, or infer information that is not visible.
   Preserve code, formulas, and technical terms accurately.

6. Remove repetition and low-value text.

7. Keep the output concise and preferably under 600 tokens.

8. If there is no useful content, output:
   `Type: irrelevant`

Output:
Type: <one or more types>

Content: <structured extraction>
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
        nonlocal batch_queue, processed_frame_count, ocr_failed_count
        nonlocal total_ocr_time, actual_ocr_count, average_ocr_time

        if not batch_queue:
            return

        print(f"DEBUG: [处理] 批量 OCR {len(batch_queue)} 帧，开始 OCR 处理")
        batch_input = [(ts, img_bytes) for ts, img_bytes, _phash in batch_queue]
        results = ocr_util.process_images_batch(
            batch_input, model_name, ocr_prompt, max_workers=max_ocr_workers
        )

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
        # 只写入 partial，最终成功前不会出现正式 txt
        ocr_util.append_results_to_file(partial_path, results, average_ocr_time)
        batch_queue = []

    try:
        for ts_idx, current_time in enumerate(timestamps):
            is_last_timestamp = ts_idx == len(timestamps) - 1

            frame_bytes, current_phash = extract_frame_as_bytes_and_phash(
                cap, current_time
            )

            if not frame_bytes or not current_phash:
                print(
                    f"WARNING: 无法提取帧，时间: {current_time:.2f}s，写入「识别失败」"
                )
                flush_batch()
                extract_failed_count += 1
                ocr_util.append_results_to_file(
                    partial_path,
                    [make_result(current_time, ocr_util.FAILED_OCR_CONTENT)],
                    average_ocr_time,
                )
                frame_idx += 1
                continue

            is_similar = False
            reused_ocr_result = None

            for stored_phash, stored_ocr_content in processed_frames_cache.items():
                diff = current_phash - stored_phash
                if diff <= phash_threshold:
                    is_similar = True
                    reused_ocr_result = stored_ocr_content
                    break

            if is_similar:
                skipped_frame_count += 1
                print(
                    f"DEBUG: [跳过] 帧 {frame_idx}/{expected_frame_count} "
                    f"(相似画面 pHash: {current_phash})，复用结果并写入"
                )
                flush_batch()
                ocr_util.append_results_to_file(
                    partial_path,
                    [make_result(current_time, reused_ocr_result)],
                    average_ocr_time,
                )
            else:
                print(
                    f"DEBUG: [排队] 帧 {frame_idx}/{expected_frame_count} "
                    f"(新画面 pHash: {current_phash})"
                )
                batch_queue.append((current_time, frame_bytes, current_phash))

                if len(batch_queue) >= BATCH_SIZE or is_last_timestamp:
                    flush_batch()

            frame_idx += 1

        flush_batch()

        total_written = (
            len(done_pairs)
            + processed_frame_count
            + skipped_frame_count
            + extract_failed_count
        )
        summary = [
            "=" * 60,
            COMPLETE_MARKER,
            "=" * 60,
            f"视频文件名: {video_name}",
            f"处理时间段: {format_time_ranges(normalized_ranges)}",
            f"OCR 处理帧数: {processed_frame_count}",
            f"跳过(复用)帧数: {skipped_frame_count}",
            f"抽帧失败帧数: {extract_failed_count}",
            f"OCR 识别失败帧数: {ocr_failed_count}",
            f"此前已完成帧数: {len(done_pairs)}",
            f"总写入帧数: {total_written}",
            "=" * 60,
        ]
        print("\n".join(summary))

        with open(partial_path, "a", encoding="utf-8") as f:
            for line in summary:
                f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

        finalize_output(partial_path, output_file_path)
        print(f"结果文件（完整）: {output_file_path}")
        return True
    except Exception as e:
        print(f"ERROR: 处理中断，进度已保存在 {partial_path}，下次将从此处续跑: {e}")
        raise
    finally:
        cap.release()


def main():
    import time

    start_time = time.time()

    data_dir = os.path.join(os.getcwd(), "data", "lectures")
    output_dir = os.path.join(os.getcwd(), "ocr_results")

    print(f"当前工作目录: {os.getcwd()}")
    print(f"数据目录: {data_dir}")
    print(f"数据目录存在: {os.path.exists(data_dir)}")

    if not os.path.exists(data_dir):
        print(f"ERROR: 数据目录不存在: {data_dir}")
        return

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
    video_files.sort()

    queue = []
    for video_path in video_files:
        final_path, partial_path = output_paths_for_video(video_path)
        if is_complete_result_file(final_path):
            print(f"跳过（已有完整结果）: {final_path}")
            continue
        if os.path.exists(partial_path) or (
            os.path.isfile(final_path) and not is_complete_result_file(final_path)
        ):
            print(f"待续跑: {video_path}")
        queue.append((video_path, final_path))

    if not queue:
        print("没有待处理视频（全部已有完整 txt）")
        return

    print(f"排队处理 {len(queue)} 个视频（串行，一次只跑一个）:")
    for i, (vp, _) in enumerate(queue, 1):
        print(f"  [{i}/{len(queue)}] {vp}")

    output_files = []

    for i, (video_path, output_file) in enumerate(queue, 1):
        print("=" * 60)
        print(f"开始处理队列 [{i}/{len(queue)}]: {video_path}")
        print("=" * 60)

        time_ranges = [
            # ("00:00:28", "00:00:28"),
            # ("00:00:41", "00:00:41"),
            # ("00:40:27", "00:40:27"),
            # ("01:01:46", "01:01:46"),
            # ("01:16:32", "01:16:32"),
            # ("02:11:50", "02:11:50"),
            # ("03:05:22", "03:05:22"),
            # ("03:07:02", "03:07:02"),
        ]

        try:
            ok = process_video(
                video_path,
                output_dir,
                model_name=os.environ.get("OCR_MODEL", "qwen3-vl:8b-instruct"),
                interval_seconds=1,
                max_test_seconds=None,
                max_ocr_workers=int(os.environ.get("OCR_MAX_WORKERS", "1")),
                time_ranges=time_ranges,
            )
        except Exception as e:
            print(f"队列 [{i}/{len(queue)}] 中断，保留进度后续续跑: {e}")
            continue

        if ok and os.path.isfile(output_file):
            output_files.append(output_file)
            print(f"队列 [{i}/{len(queue)}] 完成: {output_file}")
        else:
            print(f"队列 [{i}/{len(queue)}] 未完成，下次将续跑")

    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60

    print("=" * 60)
    print(f"所有视频处理完成！总耗时: {hours:02d}h {minutes:02d}m {seconds:.2f}s")
    print("=" * 60)

    total_time_summary = [
        "=" * 60,
        "总处理时间",
        "=" * 60,
        f"总耗时: {hours:02d}h {minutes:02d}m {seconds:.2f}s",
        "=" * 60,
        "\n",
    ]

    for output_file in output_files:
        with open(output_file, "r", encoding="utf-8") as f:
            existing_content = f.read()

        with open(output_file, "w", encoding="utf-8") as f:
            for line in total_time_summary:
                f.write(line + "\n")
            f.write(existing_content)


if __name__ == "__main__":
    main()
