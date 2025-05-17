import face_recognition
import os
import numpy as np
import cv2
from pathlib import Path


# Disable symlinks on HuggingFace to avoid permission issues on Windows
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

# Locate the spkrec folder relative to the current script
BASE_DIR = Path(__file__).resolve().parent  # This gives you .../biometric_access_control/app

# Folder containing videos
video_folder = BASE_DIR.parent / "data" / "video"

# Directory to save embeddings
embeddings_folder = BASE_DIR.parent / "embeddings" / "video"
os.makedirs(embeddings_folder, exist_ok=True)


def extract_face_embeddings_from_video(video_path, video_name):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    embeddings = []
    frames_to_extract = 10
    frame_indices = np.linspace(0, total_frames - 1, frames_to_extract, dtype=int)

    while True:
        success, frame = cap.read()
        if not success or frame is None:
            break

        if frame_count in frame_indices:
            # Convert frame to RGB directly in memory
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect face locations and embeddings
            face_locations = face_recognition.face_locations(rgb_frame)
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                if face_encodings:
                    embeddings.append(face_encodings[0])
            else:
                print(f"No face detected at frame {frame_count}. Skipping...")

        frame_count += 1
        if len(embeddings) >= frames_to_extract:
            break

    cap.release()
    print(f"Extracted {len(embeddings)} embeddings from {video_name}.")

    # Save embeddings if any were found
    if embeddings:
        embeddings_array = np.array(embeddings)
        os.makedirs(embeddings_folder, exist_ok=True)
        np.save(os.path.join(embeddings_folder, f"{video_name}_face_embeddings.npy"), embeddings_array)
        print(f"Embeddings saved for {video_name}.")
    else:
        print(f"[!] No embeddings found for {video_name}.")


def main():
    for filename in os.listdir(video_folder):
        if filename.endswith('.mp4'):
            video_path = os.path.join(video_folder, filename)
            video_name = os.path.splitext(filename)[0]
            print(f"Processing video: {video_name}")
            extract_face_embeddings_from_video(video_path, video_name)

if __name__ == '__main__':
    main()
