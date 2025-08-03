import argparse
import glob
import os
import torch
from datasets import load_dataset
from tqdm.auto import tqdm, trange
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# --- NEW: Import torch.profiler ---
import torch.profiler
# --- END NEW ---

# Set mode in command line
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu, xpu, cuda)")
parser.add_argument("--print-transcriptions", action="store_true")
args = parser.parse_args()

if args.device == "cpu":
    DEVICE = "cpu"
elif args.device == "xpu":
    DEVICE = "xpu" if hasattr(torch, 'xpu') and torch.xpu.is_available() else "cpu"
elif args.device == "cuda":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = "cpu"

print(f"Using {DEVICE} device")

# load model and tokenizer
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()
model.to(DEVICE)

# load dummy dataset and read soundfiles
ds = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)

transcriptions = []

# --- NEW: Profiler setup ---
# Define the directory to save the trace files
LOG_DIR = "./log/wav2vec2_profile"
os.makedirs(LOG_DIR, exist_ok=True) # Ensure the directory exists

# Define activities to profile based on the device
profiler_activities = [torch.profiler.ProfilerActivity.CPU]
if DEVICE == "cuda":
    profiler_activities.append(torch.profiler.ProfilerActivity.CUDA)
elif DEVICE == "xpu":
    # For XPU, ensure you have Intel® Extension for PyTorch installed
    # The activity enum might be slightly different or specific to your XPU setup.
    # torch.profiler.ProfilerActivity.XPU is typically available with Intel's PyTorch.
    # If not, you might only get CPU activity without specific XPU events.
    profiler_activities.append(torch.profiler.ProfilerActivity.XPU)


# Wrap the loop with the profiler context manager
# This schedule profiles 3 active steps after 1 wait and 1 warmup step.
# This means it will profile iterations 3, 4, and 5 (0-indexed).
# Adjust 'wait', 'warmup', 'active' based on how many iterations you want to capture.
# For example, if you want to profile all 10 iterations of the dummy dataset:
# schedule=torch.profiler.schedule(wait=0, warmup=0, active=len(ds), repeat=1)
with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(LOG_DIR),
    record_shapes=True,
    profile_memory=True,
    with_stack=False, # Set to True for more detailed stack traces, can add overhead
    # record_device_events=True, # Essential for GPU/XPU kernel timing
    activities=profiler_activities,
) as prof:
    for i in trange(len(ds)):
        # tokenize
        input_values = processor(
            ds[i]["audio"]["array"],
            return_tensors="pt",
            padding="longest",
            sampling_rate=16000,
        ).input_values  # Batch size 1

        # retrieve logits
        logits = model(input_values.to(DEVICE)).logits

        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        transcriptions.append(transcription)

        # --- NEW: Step the profiler ---
        prof.step()
        # --- END NEW ---

# --- NEW: Print instructions to view trace ---
print(f"\nProfiling complete. Trace saved to: {os.path.abspath(LOG_DIR)}")
print("To view the trace, run the following command in your terminal:")
print(f"tensorboard --logdir {os.path.abspath(LOG_DIR)}")
print("Then open your browser to the address provided by TensorBoard (usually http://localhost:6006/).")
print("Navigate to the 'Profile' tab.")
# --- END NEW ---

if args.print_transcriptions:
    print("\nTranscriptions:")
    for i, t in enumerate(transcriptions):
        print(f"Sample {i+1}: {t}")
