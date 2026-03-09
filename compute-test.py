import argparse
import io
from pathlib import Path
from typing import Any

import librosa
import torch
import torch.profiler
from datasets import Audio, load_dataset
from tqdm.auto import trange
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified accelerator smoke test for Wav2Vec2 inference"
    )
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "cuda", "xpu", "rocm"],
        default="auto",
        help="Requested accelerator backend",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "xpu", "rocm"],
        help="Deprecated alias for --accelerator",
    )
    parser.add_argument(
        "--runtime",
        choices=["any", "rocm", "cuda11", "cuda12"],
        default="any",
        help="Require a specific CUDA/ROCm runtime family",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of falling back to CPU when request is unavailable",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable torch.profiler and write Perfetto/Chrome trace JSON",
    )
    parser.add_argument(
        "--profile-dir",
        default="./log/wav2vec2_profile",
        help="Directory for profiler trace output",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit dataset samples (0 means full split)",
    )
    parser.add_argument(
        "--print-transcriptions",
        action="store_true",
        help="Print decoded text per sample",
    )
    parser.add_argument(
        "--model-id",
        default="facebook/wav2vec2-base-960h",
        help="Hugging Face model ID",
    )
    parser.add_argument(
        "--dataset-id",
        default="hf-internal-testing/librispeech_asr_dummy",
        help="Hugging Face dataset ID",
    )
    parser.add_argument(
        "--dataset-config",
        default="clean",
        help="Hugging Face dataset config",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split name",
    )
    args = parser.parse_args()
    if args.device:
        args.accelerator = args.device
    return args


def xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def detect_runtime() -> str:
    if torch.version.hip:
        return "rocm"
    if torch.version.cuda:
        if torch.version.cuda.startswith("11"):
            return "cuda11"
        if torch.version.cuda.startswith("12"):
            return "cuda12"
    return "cpu"


def resolve_device(accelerator: str, strict: bool) -> str:
    runtime = detect_runtime()

    if accelerator == "cpu":
        return "cpu"

    if accelerator == "xpu":
        if xpu_available():
            return "xpu"
        if strict:
            raise RuntimeError(
                "Requested --accelerator xpu, but torch.xpu is unavailable"
            )
        return "cpu"

    if accelerator == "cuda":
        if runtime in {"cuda11", "cuda12"} and torch.cuda.is_available():
            return "cuda"
        if strict:
            raise RuntimeError(
                "Requested --accelerator cuda, but CUDA runtime/GPU is unavailable "
                f"(runtime={runtime})"
            )
        return "cpu"

    if accelerator == "rocm":
        if runtime == "rocm" and torch.cuda.is_available():
            return "cuda"
        if strict:
            raise RuntimeError(
                "Requested --accelerator rocm, but ROCm runtime/GPU is unavailable "
                f"(runtime={runtime})"
            )
        return "cpu"

    if xpu_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def validate_runtime(runtime: str, strict: bool) -> None:
    if runtime == "any":
        return

    actual_runtime = detect_runtime()
    if runtime == actual_runtime:
        return

    message = (
        f"Requested --runtime {runtime}, but current torch runtime is {actual_runtime}. "
        f"(torch.version.cuda={torch.version.cuda}, torch.version.hip={torch.version.hip})"
    )
    if strict:
        raise RuntimeError(message)
    print(f"WARNING: {message}")


def profiler_activities_for_device(
    device: str,
) -> list[torch.profiler.ProfilerActivity]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    elif device == "xpu":
        xpu_activity = getattr(torch.profiler.ProfilerActivity, "XPU", None)
        if xpu_activity is not None:
            activities.append(xpu_activity)
        else:
            print(
                "WARNING: XPU profiler activity unavailable; profiling CPU activity only"
            )
    return activities


def extract_audio_array(sample: dict[str, Any], sample_rate: int = 16000) -> Any:
    audio = sample["audio"]

    if isinstance(audio, dict) and "array" in audio and audio["array"] is not None:
        return audio["array"]

    if not isinstance(audio, dict):
        raise RuntimeError(f"Unexpected audio sample format: {type(audio)}")

    if audio.get("bytes"):
        waveform, _ = librosa.load(
            io.BytesIO(audio["bytes"]), sr=sample_rate, mono=True
        )
        return waveform

    if audio.get("path"):
        waveform, _ = librosa.load(audio["path"], sr=sample_rate, mono=True)
        return waveform

    raise RuntimeError("Audio sample missing both 'path' and 'bytes' fields")


def run_inference(args: argparse.Namespace, device: str) -> None:
    print(f"Requested accelerator: {args.accelerator}")
    print(f"Resolved device: {device}")
    print(
        "Runtime info: "
        f"torch.version.cuda={torch.version.cuda}, torch.version.hip={torch.version.hip}"
    )

    processor = Wav2Vec2Processor.from_pretrained(args.model_id)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_id)
    model.eval()
    model_any: Any = model
    model_any.to(device)
    processor_any: Any = processor

    ds = load_dataset(args.dataset_id, args.dataset_config, split=args.split)
    ds = ds.cast_column("audio", Audio(decode=False))
    sample_count = len(ds) if args.max_samples <= 0 else min(args.max_samples, len(ds))

    transcriptions: list[str] = []

    profiler_ctx = None
    profile_dir: Path | None = None
    trace_files: list[Path] = []
    profiler_step = 0

    def on_trace_ready(prof: torch.profiler.profile) -> None:
        trace_file = profile_dir / f"trace_step_{profiler_step:05d}.json"
        prof.export_chrome_trace(str(trace_file))
        trace_files.append(trace_file)

    if args.profile:
        profile_dir = Path(args.profile_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)
        profiler_ctx = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=on_trace_ready,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            activities=profiler_activities_for_device(device),
        )
        profiler_ctx.__enter__()

    try:
        for i in trange(sample_count):
            input_values = processor_any(
                # Typed as Any because upstream transformer stubs are narrower than runtime.
                # Runtime accepts these keyword arguments for audio preprocessing.
                extract_audio_array(ds[i]),
                return_tensors="pt",
                padding="longest",
                sampling_rate=16000,
            ).input_values

            logits = model_any(input_values.to(device)).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
            transcriptions.append(transcription)

            if profiler_ctx is not None:
                profiler_step += 1
                profiler_ctx.step()
    finally:
        if profiler_ctx is not None:
            profiler_ctx.__exit__(None, None, None)

    print(f"Processed samples: {sample_count}")

    if args.profile:
        if trace_files:
            print(f"Profile traces written to: {profile_dir.resolve()}")
            print(
                f"Generated {len(trace_files)} trace file(s): "
                f"{', '.join(str(path.name) for path in trace_files)}"
            )
            print("View with Perfetto: https://ui.perfetto.dev (drag trace JSON files)")
            print("Fallback viewer: chrome://tracing")
        else:
            print(
                "No trace file generated. Increase --max-samples so profiling reaches "
                "an active window."
            )

    if args.print_transcriptions:
        print("\nTranscriptions:")
        for idx, text in enumerate(transcriptions):
            print(f"Sample {idx + 1}: {text}")


def main() -> None:
    args = parse_args()
    validate_runtime(args.runtime, args.strict)
    device = resolve_device(args.accelerator, args.strict)
    run_inference(args, device)


if __name__ == "__main__":
    main()
