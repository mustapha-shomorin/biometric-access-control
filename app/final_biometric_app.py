import os
import cv2
import torch
import time
import queue
import threading
import shutil
import sounddevice as sd
import face_recognition
from pathlib import Path
import numpy as np
import streamlit as st
import scipy.io.wavfile as wav
from scipy.io.wavfile import write
import torch.nn.functional as F
import torchaudio
from datetime import datetime
from playsound import playsound
from speechbrain.inference.speaker import SpeakerRecognition

from utils import log_access, load_access_logs
from video_enrollment import main as run_video_enrollment
from audio_enrollment import main as run_audio_enrollment
from biometric_functions import load_face_encodings, verify_face_live, load_voice_embedding_for_user, record_voice, verify_voice, process_audio

# # --- Constants ---
MODEL_PATH = os.path.join("pretrained_models", "spkrec")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "87654321"

# --- INITIALIZE SESSION STATE ---
for key in ["face_verified", "user_name", "stored_voice_embedding", "access_granted"]:
    if key not in st.session_state:
        st.session_state[key] = None

# --- Init ---
st.set_page_config(page_title="Biometric Access", layout="centered")
st.title("🔐 Event Biometric Access Control")

speaker_model = SpeakerRecognition.from_hparams(
    source=MODEL_PATH,
    savedir=MODEL_PATH,
    run_opts={"device": DEVICE},
    use_auth_token=False
)

@st.cache_resource

def play_audio_background(path):
    threading.Thread(target=playsound, args=(path,), daemon=True).start()

# Inside your main script, after importing and loading logs
df_logs = load_access_logs()

#---------Admin Panel------------------------
with st.sidebar:
    st.markdown("## Admin Controls")

    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False

    if not st.session_state.is_admin:
        with st.expander("🔐 Admin Login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                    st.success("✅ Logged in as admin")
                    st.session_state.is_admin = True
                else:
                    st.error("❌ Invalid credentials")
    
    else:
        show_logs = st.checkbox("📊 Show Access Log Dashboard")
        if show_logs:
            st.markdown("### 🔐 Access Log Dashboard")

            df_logs = load_access_logs()

            status_filter = st.multiselect(
                "Filter by Status",
                options=df_logs["status"].unique(),
                default=df_logs["status"].unique()
            )

            user_filter = st.multiselect(
                "Filter by User",
                options=df_logs["user_name"].unique(),
                default=df_logs["user_name"].unique()
            )

            filtered_df = df_logs[
                (df_logs["status"].isin(status_filter)) &
                (df_logs["user_name"].isin(user_filter))
            ]

            st.dataframe(filtered_df.sort_values("timestamp", ascending=False), use_container_width=True)

            st.download_button(
                "📥 Download Logs as CSV",
                data=filtered_df.to_csv(index=False),
                file_name="filtered_access_logs.csv",
                mime="text/csv"
            )

            if st.button("Logout"):
                st.session_state.is_admin = False
                st.experimental_rerun()

        # if st.sidebar.button("Load New Data"):
        #     st.session_state.confirm_load = True  # Trigger confirmation step

        #     # Step 2: Confirmation prompt
        #     if st.session_state.get("confirm_load", False):
        #         st.sidebar.warning("⚠️ This will overwrite existing embeddings.")
        #         if st.sidebar.button("✅ Yes, proceed to generate new embeddings"):
        #             try:
        #                 run_video_enrollment()
        #                 st.sidebar.success("✅ Face embeddings generated successfully.")

        #                 try:
        #                     run_audio_enrollment()
        #                     st.sidebar.success("✅ Voice embeddings generated successfully.")
        #                 except Exception as e:
        #                     st.sidebar.error("❌ Error running audio_enrollment")
        #                     st.sidebar.error(str(e))

        #             except Exception as e:
        #                 st.sidebar.error("❌ Error running video_enrollment")
        #                 st.sidebar.error(str(e))

        #             # Reset confirmation state
        #             st.session_state.confirm_load = False
        #         elif st.sidebar.button("❌ Cancel"):
        #             st.sidebar.info("Operation cancelled.")
        #             st.session_state.confirm_load = False

        st.markdown("### New Data")
        st.sidebar.warning("⚠️ This will overwrite existing embeddings.")
        if st.sidebar.button("Load New Data"):
            st.session_state.confirm_load = True  # Trigger confirmation step

            # Step 2: Confirmation prompt
            if st.session_state.get("confirm_load", False):
                try:
                    run_video_enrollment()
                    st.sidebar.success("✅ Face embeddings generated successfully.")

                    try:
                        run_audio_enrollment()
                        st.sidebar.success("✅ Voice embeddings generated successfully.")
                    except Exception as e:
                        st.sidebar.error("❌ Error running audio_enrollment")
                        st.sidebar.error(str(e))

                except Exception as e:
                    st.sidebar.error("❌ Error running video_enrollment")
                    st.sidebar.error(str(e))

        st.markdown("### Erase Data")
        st.sidebar.warning("⚠️ This will delete all existing embeddings.")
        if st.sidebar.button("Erase Existing Data"):
            st.session_state.erase_data = True  # Trigger confirmation step


            # # Step 2: Confirmation prompt
            # if st.session_state.get("erase_data", False):
            #     st.sidebar.warning("⚠️ This will delete all existing embeddings.")
            #     if st.sidebar.button("✅ Yes, proceed to delete all embeddings"):
            #         BASE_DIR = Path(__file__).resolve().parent
            #         embeddings_dir = BASE_DIR.parent / "embeddings"
            #         if os.path.exists(embeddings_dir):
            #             try:
            #                 # Delete the entire embeddings folder (both audio and video folders inside)
            #                 shutil.rmtree(embeddings_dir)
            #                 st.sidebar.success("✅ All embeddings have been deleted successfully.")
            #             except Exception as e:
            #                 st.sidebar.error("❌ Error deleting embeddings folder.")
            #                 st.sidebar.error(str(e))
            #         else:
            #             st.sidebar.warning("⚠️ No embeddings folder found.")

            #         # Reset confirmation state
            #         st.session_state.erase_data = False
            #     elif st.sidebar.button("❌ Cancel"):
            #         st.sidebar.info("Operation cancelled.")
            #         st.session_state.erase_data = False
    
            # Step 2: Confirmation prompt
            if st.session_state.get("erase_data", False):
                BASE_DIR = Path(__file__).resolve().parent
                embeddings_dir = BASE_DIR.parent / "embeddings"
                if os.path.exists(embeddings_dir):
                    try:
                        # Delete the entire embeddings folder (both audio and video folders inside)
                        shutil.rmtree(embeddings_dir)
                        st.sidebar.success("✅ All embeddings have been deleted successfully.")
                    except Exception as e:
                        st.sidebar.error("❌ Error deleting embeddings folder.")
                        st.sidebar.error(str(e))
                else:
                    st.sidebar.warning("⚠️ No embeddings folder found.")

                    # Reset confirmation state
                    st.session_state.erase_data = False
        

# --- Initialize session state ---
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'face_verification'
if 'face_verified' not in st.session_state:
    st.session_state.face_verified = None
if 'access_granted' not in st.session_state:
    st.session_state.access_granted = False

# --- MAIN APP UI ---
known_face_encodings, known_face_names = load_face_encodings() 

# --- Step 1: Face Verification ---
if st.session_state.current_step == 'face_verification':
    st.markdown("### Step 1: Face Verification")
    if st.button("Start Face Verification"):
        user = verify_face_live(known_face_encodings, known_face_names)
        if user:
            st.session_state.face_verified = True
            st.session_state.user_name = user
            voice_path = load_voice_embedding_for_user(user)
            if voice_path:
                st.session_state.stored_voice_embedding = voice_path
                st.session_state.current_step = 'voice_verification'
                st.rerun()
            else:
                st.warning("⚠️ Face recognized but no voice data found.")
                log_access(user.title(), "denied", "Missing voice embedding")
                st.session_state.face_verified = None
        
# --- Step 2: Voice Verification ---
elif st.session_state.current_step == 'voice_verification':
    st.markdown("### Step 2: Voice Verification")
    if st.button("Record Voice"):
        user_voice = record_voice()
        processed_audio = process_audio(user_voice)
        voice_tensor = torch.load(st.session_state.stored_voice_embedding)
        verified, score = verify_voice(processed_audio, voice_tensor)

        if verified:
            st.session_state.access_granted = True
            st.session_state.current_step = 'access_granted'
            log_access(st.session_state.user_name.title(), "access granted", "Voice matches face")
            st.rerun()
        else:
            st.error(f"❌ Voice mismatch. Please try again.")
            log_access(st.session_state.user_name.title(), "denied", "Voice mismatch")

# --- Access Granted Block ---
elif st.session_state.current_step == 'access_granted':
    st.success("🎉 Access Granted!")
    st.markdown(f"**Welcome, {st.session_state.user_name.title()}**")
    st.markdown(f"🕒 **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("🏷️ **Event:** Tech Innovators Conference 2025")
    play_audio_background("welcome.wav")  # Play welcome message silently
    st.balloons()

    if st.button("Start New Verification"):
        for key in ["user_name", "stored_voice_embedding", "face_verified", "access_granted"]:
            st.session_state[key] = None
        st.session_state.current_step = 'face_verification'
        st.rerun()


