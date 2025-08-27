import streamlit as st
import numpy as np
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav
import os
import pickle
import speech_recognition as sr
import time
import tempfile

# ------------------- Setup -------------------
st.set_page_config(page_title="Voice Identification App", page_icon="🎙️", layout="centered")

encoder = VoiceEncoder()
recognizer = sr.Recognizer()

# Database file
DB_FILE = "Known_Audio_DB/voice_db.pkl"
os.makedirs("Known_Audio_DB", exist_ok=True)

if os.path.exists(DB_FILE):
    with open(DB_FILE, "rb") as f:
        db = pickle.load(f)
else:
    db = {}

# ------------------- Functions -------------------
def record_audio():
    """Record audio using SpeechRecognition and save temp file"""
    with sr.Microphone() as source:
        st.info("🎤 Listening... please speak")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=20)

    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with open(tmpfile.name, "wb") as f:
        f.write(audio.get_wav_data())

    st.success("✅ Recording complete")
    return tmpfile.name


def register_speaker(file_path, speaker_name):
    wav = preprocess_wav(Path(file_path))
    embedding = encoder.embed_utterance(wav)
    db[speaker_name] = embedding
    with open(DB_FILE, "wb") as f:
        pickle.dump(db, f)
    return f"✅ Speaker **{speaker_name}** registered successfully!"


def identify_speaker(file_path, threshold=0.75):
    if not db:
        return "[ERROR] No speakers in database. Register first!", None, None

    wav = preprocess_wav(Path(file_path))
    embedding = encoder.embed_utterance(wav)

    best_match, best_score = None, -1
    for name, ref_embedding in db.items():
        score = np.dot(embedding, ref_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(ref_embedding))
        if score > best_score:
            best_match, best_score = name, score

    if best_score > threshold:
        return f"🎯 Identified as: **{best_match}** (Score: {best_score:.2f})", best_match, best_score
    else:
        return f"❌ Unknown speaker (Closest: {best_match}, Score: {best_score:.2f})", "Unknown", best_score


# ------------------- Streamlit UI -------------------
st.title("🎙️ Voice Identification App")
st.write("Register speakers and identify them using voice embeddings (Resemblyzer).")

menu = ["Register Speaker", "Identify Speaker", "Database"]
choice = st.sidebar.radio("Navigation", menu)

# ----- Register -----
if choice == "Register Speaker":
    st.subheader("Register a New Speaker")
    name = st.text_input("Enter Speaker Name")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎤 Record Live"):
            if name:
                filepath = record_audio()
                msg = register_speaker(filepath, name)
                st.success(msg)
            else:
                st.warning("⚠️ Please enter a name before recording")

    with col2:
        file_upload = st.file_uploader("Upload Audio File", type=["wav", "mp3"])
        if file_upload and name:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(file_upload.read())
                tmp_path = tmp.name
            msg = register_speaker(tmp_path, name)
            st.success(msg)
        elif file_upload and not name:
            st.warning("⚠️ Please enter a name before uploading")

# ----- Identify -----
elif choice == "Identify Speaker":
    st.subheader("Identify a Speaker")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎤 Record Live"):
            filepath = record_audio()
            result, match, score = identify_speaker(filepath)
            st.info(result)

    with col2:
        file_upload = st.file_uploader("Upload Audio File", type=["wav", "mp3"])
        if file_upload:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(file_upload.read())
                tmp_path = tmp.name
            result, match, score = identify_speaker(tmp_path)
            st.info(result)

# ----- Database -----
elif choice == "Database":
    st.subheader("Registered Speakers Database")
    if db:
        st.write("### Speakers in Database:")
        for name in db.keys():
            st.write(f"- {name}")
    else:
        st.warning("⚠️ No speakers registered yet.")
