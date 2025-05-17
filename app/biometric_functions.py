import os
import cv2
import torch
import time
import queue
import threading
import shutil
import sounddevice as sd
import face_recognition
import numpy as np
import streamlit as st
import scipy.io.wavfile as wav
from scipy.io.wavfile import write
import torch.nn.functional as F
import torchaudio
from pathlib import Path
from datetime import datetime
from playsound import playsound
from speechbrain.inference.speaker import SpeakerRecognition



# --- Constants ---
# Locate the spkrec folder relative to the current script
BASE_DIR = Path(__file__).resolve().parent  # This gives you .../biometric_access_control/app
FACE_EMBED_DIR = BASE_DIR.parent / "embeddings" / "video"
VOICE_EMBED_DIR = BASE_DIR.parent / "embeddings" / "audio"

#MODEL_PATH = os.path.join("../pretrained_models", "spkrec")
MODEL_PATH = BASE_DIR.parent / "pretrained_models" / "spkrec"
AUDIO_TMP_PATH = "temp_user_audio.wav"
FACE_MATCH_THRESHOLD = 0.4
VOICE_MATCH_THRESHOLD = 0.60
RECORD_DURATION = 10
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "pass123"

from utils import log_access, load_access_logs

speaker_model = SpeakerRecognition.from_hparams(
    source=MODEL_PATH,
    savedir=MODEL_PATH,
    run_opts={"device": DEVICE},
    use_auth_token=False
)


def load_face_encodings():
    known_face_encodings = []
    known_face_names = []
    encodings = {}
    for file in os.listdir(FACE_EMBED_DIR):
        if file.endswith(".npy"):
            name = os.path.splitext(file)[0].replace("_face_embeddings", "")
            encoding = np.load(os.path.join(FACE_EMBED_DIR, file))
            encodings[name] = encoding
            known_face_encodings.append(encoding)
            known_face_names.append(name)
    return known_face_encodings, known_face_names

def verify_face_live(known_face_encodings, known_face_names):
    st.info("📸 Please look into your camera for verification...")
    
    cap = cv2.VideoCapture(0)
    matched_name = None
    verified=False
    stframe = st.empty()

    with st.spinner("🔍 Processing..."):
        for _ in range(100):
            ret, frame = cap.read()
            if not ret:
                continue
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                face_matched_in_frame = False
                for known_embedding, known_name in zip(known_face_encodings, known_face_names):
                    distances = face_recognition.face_distance(known_embedding, face_encoding)
                    if np.min(distances) < FACE_MATCH_THRESHOLD:
                        verified = True
                        matched_name = known_name
                        face_matched_in_frame = True
                        break  # Break inner loop (known faces)
                if face_matched_in_frame:
                    break  # Break outer loop (detected faces in the frame)


            # Show frame in Streamlit
            stframe.image(frame, channels="BGR")
            if verified:
                break
            time.sleep(0.1)

    cap.release()
    if verified:
        st.success(f"✅ Face verified: {matched_name.title()}")
        log_access(matched_name.title(), "success", "face match")
        return matched_name
    else:
        st.error("❌ Face not recognized. Please try again")
        log_access("Unregistered user", "denied", "Face not recognized")
        return None

def load_voice_embedding_for_user(name):
    path = os.path.join(VOICE_EMBED_DIR, f"{name}_voice.pt")
    if not os.path.exists(path):
        return None
    return path

def record_voice(seconds=10, fs=16000):
    st.info("🎤 Please speak after clicking the record button")
    audio_q = queue.Queue()

    def callback(indata, frames, time, status):
        audio_q.put(indata.copy())

    with sd.InputStream(samplerate=fs, channels=1, callback=callback):
        audio_data = []
        with st.spinner("Recording..."):
            for _ in range(int(fs / 1024 * seconds)):
                audio_data.append(audio_q.get())
        audio_np = np.concatenate(audio_data, axis=0)
        wav.write(AUDIO_TMP_PATH, fs, audio_np)
    st.success("✅ Recording complete")
    return AUDIO_TMP_PATH

def verify_voice(processed_audio, stored_embedding):
    score = F.cosine_similarity(processed_audio, stored_embedding, dim=0).item()
    return score > VOICE_MATCH_THRESHOLD, score

def process_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
    waveform = waveform.squeeze(0)  # Make sure shape is [time] not [1, time]
    embedding = speaker_model.encode_batch(waveform.unsqueeze(0)).squeeze().detach()
    return embedding

def play_audio_background(path):
    threading.Thread(target=playsound, args=(path,), daemon=True).start()
