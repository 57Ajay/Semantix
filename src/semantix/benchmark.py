from __future__ import annotations

import datetime
import os
import platform
import shutil
import statistics
import sys
import sysconfig
import tempfile
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .chunking import chunk_text, document_prompt
from .config import Config
from .media import AUDIO_RATE
from .model import Embedder, local_snapshot, resolve_device, resolve_dtype
from .paths import Paths

AGREEMENT_FLOOR = 0.99
STAGES = ("text", "query", "image", "video", "audio")
E2E_VIDEO_SECONDS = 34
AUDIO_TOKENS_PER_SECOND = 24
REQUIRED_FILES = 3
STDLIB_FILES = (
    "json/decoder.py", "heapq.py", "textwrap.py", "bisect.py", "shlex.py", "fnmatch.py", "colorsys.py",
    "json/encoder.py", "glob.py", "base64.py", "gzip.py", "csv.py", "calendar.py", "fractions.py",
    "statistics.py", "tempfile.py", "ftplib.py", "smtplib.py", "configparser.py", "difflib.py",
    "random.py", "string.py", "copy.py", "queue.py", "sched.py", "getpass.py", "netrc.py", "pprint.py",
    "reprlib.py", "numbers.py", "operator.py", "keyword.py", "linecache.py", "tokenize.py", "codecs.py",
    "hashlib.py", "hmac.py", "secrets.py", "uuid.py", "zipapp.py", "tarfile.py", "shutil.py",
)
SIZES = {
    "small": {"chunks": 16, "images": 3, "frames": 6, "segments": 1, "audio": 1, "queries": 10, "chars": 40_000},
    "large": {"chunks": 384, "images": 48, "frames": 30, "segments": 3, "audio": 6, "queries": 30, "chars": 600_000},
}
SHAPES = (
    ("red_circle", "red", "circle"), ("blue_square", "blue", "square"), ("green_triangle", "green", "triangle"),
    ("yellow_circle", "gold", "circle"), ("purple_square", "purple", "square"),
)
EXPECTATIONS = (
    ("decode a JSON document from a string", "decoder.py"),
    ("priority queue that pops the smallest item", "heapq.py"),
    ("wrap a paragraph of text to a line width", "textwrap.py"),
    ("a red circle", "red_circle.png"),
    ("a blue square", "blue_square.png"),
    ("an invoice showing the total amount due", "invoice.png"),
)


def environment(device: str) -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "semantix": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
        "device": device,
        "nvidia_smi": shutil.which("nvidia-smi") is not None,
    }
    try:
        info["memory_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (ValueError, OSError, AttributeError):
        info["memory_gb"] = None
    if device == "cuda":
        properties = torch.cuda.get_device_properties(0)
        info["gpu"] = properties.name
        info["gpu_memory_gb"] = round(properties.total_memory / 1024**3, 1)
        info["gpu_capability"] = ".".join(str(part) for part in torch.cuda.get_device_capability(0))
        info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    return info


def stdlib_sources(max_chars: int) -> list[tuple[str, str]]:
    root = Path(sysconfig.get_path("stdlib") or "")
    sources: list[tuple[str, str]] = []
    total = 0
    for name in STDLIB_FILES:
        path = root / name
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if total + len(text) > max_chars and len(sources) >= REQUIRED_FILES:
            continue
        sources.append((name, text))
        total += len(text)
    if len(sources) < REQUIRED_FILES:
        package = Path(__file__).parent
        for path in sorted(package.glob("*.py")):
            sources.append((path.name, path.read_text(encoding="utf-8")))
    return sources


def shape_image(color: str, shape: str, size: tuple[int, int] = (1280, 720)):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    width, height = size
    box = (width // 2 - height // 3, height // 6, width // 2 + height // 3, height * 5 // 6)
    if shape == "circle":
        draw.ellipse(box, fill=color)
    elif shape == "square":
        draw.rectangle(box, fill=color)
    else:
        draw.polygon([(width // 2, box[1]), (box[0], box[3]), (box[2], box[3])], fill=color)
    return image


def invoice_image():
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1280, 720), "white")
    draw = ImageDraw.Draw(image)
    title = ImageFont.load_default(size=64)
    body = ImageFont.load_default(size=36)
    draw.text((80, 60), "INVOICE #1042", fill="black", font=title)
    for row, line in enumerate(("Consulting, 12 hours        $240.00", "Hosting, March               $78.00", "Total amount due            $318.00")):
        draw.text((80, 220 + row * 90), line, fill="black", font=body)
    return image


def noise_images(count: int) -> list:
    from PIL import Image

    generator = np.random.default_rng(7)
    images = []
    for index in range(count):
        base = generator.integers(0, 255, (45, 80, 3), dtype=np.uint8)
        image = Image.fromarray(base).resize((1280, 720))
        _name, color, shape = SHAPES[index % len(SHAPES)]
        overlay = shape_image(color, shape)
        images.append(Image.blend(image, overlay, 0.6))
    return images


def moving_frames(count: int, size: tuple[int, int] = (640, 360)) -> list:
    from PIL import Image, ImageDraw

    frames = []
    for index in range(count):
        frame = Image.new("RGB", size, "white")
        offset = int((size[0] - 160) * index / max(count - 1, 1))
        ImageDraw.Draw(frame).ellipse((offset, 100, offset + 160, 260), fill="red")
        frames.append(frame)
    return frames


def tone(seconds: float) -> np.ndarray:
    moments = np.arange(int(seconds * AUDIO_RATE)) / AUDIO_RATE
    sweep = 220 + 440 * moments / max(seconds, 1)
    return (0.3 * np.sin(2 * np.pi * sweep * moments)).astype(np.float32)


def reference_inputs() -> dict[str, Any]:
    return {
        "documents": [
            document_prompt("add.py", "def add(a, b):\n    return a + b"),
            document_prompt("notes.md", "The database password is rotated on the first Monday of each month."),
            document_prompt("none", "A recipe for lentil soup with cumin and lemon."),
        ],
        "query": "function that adds two numbers",
        "images": [shape_image("red", "circle"), invoice_image()],
        "frames": moving_frames(4),
        "audio": tone(5),
    }


def reference_vectors(embedder: Embedder, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
    vectors = {
        "text": embedder.embed_documents(inputs["documents"]),
        "query": np.stack(list(embedder.embed_query(inputs["query"]).values())),
    }
    if "image" in embedder.capabilities:
        vectors["image"] = embedder.embed_images(inputs["images"])
    if "video" in embedder.capabilities:
        vectors["video"] = embedder.embed_video(inputs["frames"], 1.0)[None, :]
    if "audio" in embedder.capabilities:
        vectors["audio"] = embedder.embed_audio(inputs["audio"])[None, :]
    return vectors


def agreement(candidate: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        name: round(float(np.min(np.sum(candidate[name] * reference[name], axis=1))), 5)
        for name in reference
        if name in candidate
    }


def peak_memory(embedder: Embedder) -> dict[str, float]:
    result: dict[str, float] = {}
    if embedder.device == "cuda":
        import torch

        result["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1048576)
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS"):
                    result["process_rss_mb"] = round(int(line.split()[1]) / 1024)
    except OSError:
        pass
    return result


def describe(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"[:600]


def check_limits(embedder: Embedder) -> dict[str, Any]:
    from PIL import Image

    from .config import OPTIONS
    from .model import CONTEXT_TOKENS, MAX_AUDIO_SECONDS, max_video_frames, video_segment_seconds

    result: dict[str, Any] = {"context_tokens": CONTEXT_TOKENS, "video": [], "audio": []}
    if "audio" in embedder.capabilities:
        for seconds in (30, 60, MAX_AUDIO_SECONDS):
            entry: dict[str, Any] = {"seconds": seconds}
            try:
                entry["tokens"] = embedder.audio_token_count(tone(seconds))
                entry["fits"] = entry["tokens"] <= CONTEXT_TOKENS
                entry["complete"] = entry["tokens"] >= seconds * AUDIO_TOKENS_PER_SECOND
            except Exception as error:
                entry["fits"] = entry["complete"] = False
                entry["error"] = describe(error)
            result["audio"].append(entry)
    if "video" not in embedder.capabilities:
        result["ok"] = all(entry["fits"] and entry["complete"] for entry in result["audio"])
        return result
    original = embedder.config
    try:
        for tokens in OPTIONS["video.frame_tokens"].choices:
            config = Config({**original.flat(), "video.frame_tokens": tokens})
            embedder.config = config
            config.set("video.segment_seconds", 3600)
            fps = config.get("video.fps")
            frames = max(round(video_segment_seconds(config) * fps), 1)
            entry: dict[str, Any] = {"frame_tokens": tokens, "frames": frames, "limit_frames": max_video_frames(tokens)}
            try:
                image = Image.new("RGB", (512, 512), "gray")
                entry["tokens"] = embedder.video_token_count([image] * frames, fps)
                entry["fits"] = entry["tokens"] <= CONTEXT_TOKENS
                entry["complete"] = entry["tokens"] >= frames * tokens * 0.8
            except Exception as error:
                entry["fits"] = entry["complete"] = False
                entry["error"] = describe(error)
            result["video"].append(entry)
    finally:
        embedder.config = original
    result["ok"] = all(entry["fits"] and entry["complete"] for entry in [*result["video"], *result["audio"]])
    return result


def measure(embedder: Embedder, sizes: dict[str, int], chunks: list[str], say: Callable[[str], None]) -> dict[str, Any]:
    if embedder.device == "cuda":
        import torch

        torch.cuda.reset_peak_memory_stats()
    result: dict[str, Any] = {"text_batch": embedder.text_batch_size(), "media_batch": embedder.media_batch_size()}

    def stage(name: str, work: Callable[[], dict[str, Any]]) -> None:
        say(name)
        try:
            result[name] = work()
        except Exception as error:
            result[name] = {"error": describe(error)}

    def text() -> dict[str, Any]:
        embedder.embed_documents(chunks[:2])
        batch = chunks[: sizes["chunks"]]
        started = time.perf_counter()
        embedder.embed_documents(batch)
        elapsed = time.perf_counter() - started
        characters = sum(len(item) for item in batch)
        return {
            "parts": len(batch), "characters": characters, "seconds": round(elapsed, 3),
            "characters_per_second": round(characters / elapsed), "parts_per_second": round(len(batch) / elapsed, 2),
        }

    def query() -> dict[str, Any]:
        embedder.embed_query("warm up")
        timings = []
        for index in range(sizes["queries"]):
            started = time.perf_counter()
            embedder.embed_query(f"where is the retry logic for request number {index}")
            timings.append(time.perf_counter() - started)
        return {"runs": len(timings), "median_ms": round(statistics.median(timings) * 1000, 1)}

    def image() -> dict[str, Any]:
        images = noise_images(sizes["images"])
        embedder.embed_images(images[:1])
        started = time.perf_counter()
        embedder.embed_images(images)
        elapsed = time.perf_counter() - started
        return {
            "images": len(images), "tokens": embedder.config.get("image.tokens"), "seconds": round(elapsed, 3),
            "images_per_second": round(len(images) / elapsed, 2), "seconds_per_image": round(elapsed / len(images), 3),
        }

    def video() -> dict[str, Any]:
        frames = moving_frames(sizes["frames"])
        embedder.embed_video(frames[:2], 1.0)
        started = time.perf_counter()
        for _ in range(sizes["segments"]):
            embedder.embed_video(frames, 1.0)
        elapsed = time.perf_counter() - started
        total = sizes["segments"] * len(frames)
        return {
            "segments": sizes["segments"], "frames_per_segment": len(frames),
            "tokens": embedder.config.get("video.frame_tokens"), "seconds": round(elapsed, 3),
            "frames_per_second": round(total / elapsed, 2), "seconds_per_segment": round(elapsed / sizes["segments"], 3),
        }

    def audio() -> dict[str, Any]:
        samples = tone(30)
        embedder.embed_audio(samples[: AUDIO_RATE * 2])
        started = time.perf_counter()
        for _ in range(sizes["audio"]):
            embedder.embed_audio(samples)
        elapsed = time.perf_counter() - started
        seconds = 30 * sizes["audio"]
        return {"audio_seconds": seconds, "seconds": round(elapsed, 3), "times_realtime": round(seconds / elapsed, 1)}

    stage("text", text)
    stage("query", query)
    for name, work in (("image", image), ("video", video), ("audio", audio)):
        if name in embedder.capabilities:
            stage(name, work)
    result["memory"] = peak_memory(embedder)
    result["stage_errors"] = sorted(name for name in STAGES if "error" in result.get(name, {}))
    return result


def write_video(path: Path, seconds: int = E2E_VIDEO_SECONDS) -> None:
    import av

    with av.open(str(path), "w") as container:
        try:
            stream = container.add_stream("libx264", rate=12)
        except Exception:
            stream = container.add_stream("mpeg4", rate=12)
        stream.width, stream.height, stream.pix_fmt = 640, 360, "yuv420p"
        for frame in moving_frames(seconds * 12):
            for packet in stream.encode(av.VideoFrame.from_image(frame)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def write_audio(path: Path, seconds: int = 20) -> None:
    samples = (tone(seconds) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(AUDIO_RATE)
        handle.writeframes(samples.tobytes())


def build_corpus(folder: Path, sources: list[tuple[str, str]], capabilities: tuple[str, ...]) -> None:
    for name, text in sources:
        target = folder / "stdlib" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    if "image" in capabilities:
        (folder / "pictures").mkdir(parents=True)
        for name, color, shape in SHAPES:
            shape_image(color, shape).save(folder / "pictures" / f"{name}.png")
        invoice_image().save(folder / "pictures" / "invoice.png")
    if "video" in capabilities:
        (folder / "media").mkdir(exist_ok=True)
        write_video(folder / "media" / "moving_dot.mp4")
    if "audio" in capabilities:
        (folder / "media").mkdir(exist_ok=True)
        write_audio(folder / "media" / "sweep.wav")


def end_to_end(config: Config, models_dir: Path, sources: list[tuple[str, str]], say: Callable[[str], None]) -> dict[str, Any]:
    from .engine import Engine

    workspace = Path(tempfile.mkdtemp(prefix="semantix-benchmark-"))
    try:
        corpus = workspace / "corpus"
        corpus.mkdir()
        capabilities = tuple(config.get("model.capabilities"))
        build_corpus(corpus, sources, capabilities)
        embedder = Embedder(config, models_dir)
        embedder.load()
        engine = Engine(Paths(workspace / "home"), config, embedder)
        try:
            engine.ensure_anchors()
            say("sync")
            plan = engine.plan([os.path.realpath(corpus)])
            started = time.perf_counter()
            done = list(engine.execute(plan["id"]))[-1]
            elapsed = time.perf_counter() - started
            say("search")
            checks = []
            timings = []
            for query, expected in EXPECTATIONS:
                if expected.endswith(".png") and "image" not in capabilities:
                    continue
                began = time.perf_counter()
                hits = engine.search(query, limit=3)["hits"]
                timings.append(time.perf_counter() - began)
                names = [os.path.basename(hit["path"]) for hit in hits]
                checks.append(
                    {
                        "query": query, "expected": expected, "top": names,
                        "rank": names.index(expected) + 1 if expected in names else None,
                        "score": hits[0]["score"] if hits else None,
                    }
                )
            result = {
                "files": plan["totals"]["files"], "bytes": plan["totals"]["bytes"],
                "estimated_seconds": round(plan["totals"]["cost"], 1), "seconds": round(elapsed, 2),
                "embedded": done["embedded"], "failed": done["failed"], "parts": done["chunks"],
                "search_median_ms": round(statistics.median(timings) * 1000, 1) if timings else None,
                "checks": checks,
                "top1": sum(check["rank"] == 1 for check in checks),
                "top3": sum(check["rank"] is not None for check in checks),
                "queries": len(checks),
                "memory": peak_memory(embedder),
                "errors": engine.store.error_files(),
            }
        finally:
            engine.close()
            embedder.unload()
        return result
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def candidate_dtypes(device: str, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    if device == "cuda":
        import torch

        return ["bfloat16", "float32"] if torch.cuda.is_bf16_supported() else ["float32"]
    if device == "cpu":
        return ["float32", "bfloat16"]
    return ["float32"]


def run(
    models_dir: Path,
    device_preference: str = "auto",
    dtypes: list[str] | None = None,
    workload: str = "auto",
    overrides: dict[str, Any] | None = None,
    say: Callable[[str], None] = lambda message: None,
) -> dict[str, Any]:
    device = resolve_device(device_preference)
    if workload == "auto":
        workload = "small" if device == "cpu" else "large"
    sizes = SIZES[workload]
    overrides = {key: value for key, value in (overrides or {}).items() if value}
    report: dict[str, Any] = {
        "when": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "environment": environment(device),
        "sizes": workload,
        "overrides": overrides,
        "command": " ".join(sys.argv[1:]),
    }
    base = Config({"model.device": device, **overrides})
    if local_snapshot(base.get("model.id"), models_dir) is None:
        raise RuntimeError("the model is not downloaded")
    sources = stdlib_sources(sizes["chars"])
    budget = base.get("index.chunk_chars")
    chunks = [
        document_prompt(name, chunk.text) for name, text in sources for chunk in chunk_text(text, budget)
    ]
    while len(chunks) < sizes["chunks"]:
        chunks = chunks + chunks
    inputs = reference_inputs()

    reference = None
    if device != "cpu":
        say("reference embeddings on the CPU")
        cpu = Embedder(Config({"model.device": "cpu", "model.dtype": "float32"}), models_dir)
        cpu.load()
        reference = reference_vectors(cpu, inputs)
        cpu.unload()

    report["dtypes"] = {}
    for dtype in candidate_dtypes(device, dtypes):
        entry: dict[str, Any] = {}
        report["dtypes"][dtype] = entry
        config = Config({"model.device": device, "model.dtype": dtype, **overrides})
        embedder = Embedder(config, models_dir)
        try:
            say(f"{dtype}: loading")
            started = time.perf_counter()
            embedder.load()
            entry["load_seconds"] = round(time.perf_counter() - started, 2)
            vectors = reference_vectors(embedder, inputs)
            if reference is None:
                reference = vectors
            entry["agreement_with_cpu_float32"] = agreement(vectors, reference)
            entry["agrees"] = all(value >= AGREEMENT_FLOOR for value in entry["agreement_with_cpu_float32"].values())
            if "limits" not in report:
                say("checking input limits")
                report["limits"] = check_limits(embedder)
            entry.update(measure(embedder, sizes, chunks, lambda stage, dtype=dtype: say(f"{dtype}: {stage}")))
        except Exception as error:
            entry["error"] = describe(error)
        finally:
            embedder.unload()

    usable = {
        name: entry
        for name, entry in report["dtypes"].items()
        if "error" not in entry and entry.get("agrees") and "error" not in entry.get("text", {"error": True})
    }
    report["default_dtype"] = str(resolve_dtype(device, "auto")).replace("torch.", "")
    if usable:
        fastest = max(usable, key=lambda name: usable[name]["text"]["characters_per_second"])
        report["fastest_dtype"] = fastest
        chosen = report["default_dtype"] if report["default_dtype"] in usable else fastest
        try:
            say(f"end to end with {chosen}")
            report["end_to_end"] = end_to_end(
                Config({"model.device": device, "model.dtype": chosen, **overrides}), models_dir, sources, say
            )
            report["end_to_end"]["dtype"] = chosen
        except Exception as error:
            report["end_to_end"] = {"error": describe(error)}
    run_result = report.get("end_to_end", {"error": True})
    report["ok"] = (
        bool(usable)
        and "error" not in run_result
        and run_result.get("failed") == 0
        and report.get("limits", {}).get("ok", True)
        and not any(entry.get("stage_errors") or "error" in entry for entry in report["dtypes"].values())
    )
    return report
