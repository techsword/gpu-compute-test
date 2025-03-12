import os 
import glob

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch
from tqdm.auto import trange, tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

# load model and tokenizer
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()
model.to(device)
    
# load dummy dataset and read soundfiles
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")


for i in trange(len(ds)):
    # tokenize
    input_values = processor(ds[i]["audio"]["array"], return_tensors="pt", padding="longest", sampling_rate=16000).input_values  # Batch size 1

    # retrieve logits
    logits = model(input_values.to(device)).logits

    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)

    transcription = processor.batch_decode(predicted_ids)
    print(transcription)
