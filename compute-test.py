import argparse
import glob
import os

import torch
from datasets import load_dataset
from tqdm.auto import tqdm, trange
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Set mode in command line
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu", help="Device to use")
parser.add_argument("--print-transcriptions", action="store_true")
args = parser.parse_args()

if args.device == "cpu":
    DEVICE = "cpu"
elif args.device == "xpu":
    DEVICE = "xpu" if torch.xpu.is_available() else "cpu"
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


    
