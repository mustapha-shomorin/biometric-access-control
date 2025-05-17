# Biometric Access Control System

This project is a multi-modal biometric access control system that uses both **facial recognition** and **voice recognition** to verify user identity. Designed for **invite-only events**, it ensures secure access by matching a user's face and voice with pre-enrolled data.

---

## 🔧 Features

- 🎥 **Face Recognition** from video uploads
- 🎙️ **Voice Recognition** from audio recordings
- 🧠 **Embedding Generator** with pre-trained models
- 🧑‍💼 **Admin Dashboard** for data management
- ⚙️ **Streamlit UI** for interactive use
- 🔐 **Multi-modal verification**

---

## 📁 Project Structure

```
biometric_access_control/
├── app/
│   ├── final_biometric_app.py       # Main Streamlit app
│   ├── audio_enrollment.py          # Voice embedding logic
│   ├── video_enrollment.py          # Face embedding logic
│   ├── utils.py                     # Shared helper functions
├── data/
│   ├── video/                       # Uploaded videos for face data
│   ├── audio/                       # Uploaded audios for voice data
├── embeddings/
│   ├── video/                       # Face embeddings (npy files)
│   ├── audio/                       # Voice embeddings (npy files)
├── pretrained_models/
│   ├── spkrec/                      # Pretrained speaker recognition model
├── requirements.txt                # Project dependencies
├── .streamlit/
│   └── config.toml                 # Streamlit layout customization
├── README.md                       # Project overview
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/biometric-access-control.git
cd biometric_access_control
```

### 2. Create & Activate Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app/final_biometric_app.py
```

---

## 🧪 Admin Controls

Admins can:
- Load new user data (video/audio) (admin only)
- Automatically run face & voice embedding generation
- Delete all saved embeddings (admin only)
- View verification logs (admin only)

---

## 🧠 Models Used

- **Face Recognition:** `face_recognition` library
- **Voice Recognition:** `speechbrain` pretrained speaker verification model

---

## ⚠️ Notes

- The `pretrained_models/spkrec` folder must contain the speaker verification model from SpeechBrain.
- Avoid using special characters or non-standard paths, especially on Windows.
- Tested on Python 3.10+

---

## 📜 License

MIT License

---

## 🙋‍♂️ Author

Developed by Mustapha Shomorin.  
Feel free to connect on (https://www.linkedin.com/in/mustaphashomorin) or explore more at (https://mustaphashomorin.wixsite.com/mustapha-shomorin).
