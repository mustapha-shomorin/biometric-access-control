import face_recognition
import cv2
import os
import numpy as np
import torch
from speechbrain.inference.speaker import SpeakerRecognition
from pathlib import Path
from datetime import datetime

# Disable symlinks on HuggingFace to avoid permission issues on Windows
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

# Locate the spkrec folder relative to the current script
BASE_DIR = Path(__file__).resolve().parent  # This gives you .../biometric_access_control/app

# Paths
AUDIO_DIR = BASE_DIR.parent / "data" / "audio"  # Directory where your audio files are located
EMBEDDING_DIR = BASE_DIR.parent / "embeddings" / "audio" # Directory to save embeddings
#MODEL_PATH = os.path.join("../pretrained_models", "spkrec")  # Path to the pre-trained model
#MODEL_PATH = Path(__file__).parent/"pretrained_models"/"spkrec"

# Locate the spkrec folder relative to the current script
BASE_DIR = Path(__file__).resolve().parent  # This gives you .../biometric_access_control/app
MODEL_PATH = BASE_DIR.parent / "pretrained_models" / "spkrec"

# Ensure the embedding directory exists
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# Initialize speaker recognition model from local files
speaker_model = SpeakerRecognition.from_hparams(
    source=MODEL_PATH,
    savedir=MODEL_PATH,
    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    use_auth_token=False
)

import torchaudio

def process_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
    waveform = waveform.squeeze(0)  # Make sure shape is [time] not [1, time]
    embedding = speaker_model.encode_batch(waveform.unsqueeze(0)).squeeze().detach()
    return embedding



# Function to save audio embedding
def save_audio_embedding(name, embedding):
    os.makedirs(EMBEDDING_DIR, exist_ok=True)
    embedding_filename = os.path.join(EMBEDDING_DIR, f"{name}_voice.pt")
    torch.save(embedding, embedding_filename)  # Save the embedding
    print(f"Embedding saved for {name} as {embedding_filename}")


def main():
    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')]  # Get all .wav files
    for audio_filename in audio_files:
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        name = os.path.splitext(audio_filename)[0]  # Get the base name without extension
        print(f"Processing: {name}")

        try:
                # Extract and save embedding
                embedding = process_audio(audio_path)
                save_audio_embedding(name, embedding)
        except Exception as e:
            print(f"[!] Error processing {name}: {e}")

if __name__ == '__main__':
    main()
