import streamlit as st
import yt_dlp
from pathlib import Path
import os
from transformers import pipeline
from faster_whisper import WhisperModel

# Load models only once
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    whisper_model = WhisperModel("base", compute_type="int8")
    return summarizer, whisper_model

summarizer, whisper_model = load_models()

# Download audio using yt-dlp
def download_audio(url):
    output_path = "temp_audio/audio.wav"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'temp_audio/audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'quiet': True,
    }

    try:
        os.makedirs("temp_audio", exist_ok=True)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path
    except Exception as e:
        st.error(f"Download error: {e}")
        return None

# Transcribe audio using faster-whisper
def transcribe_audio(path):
    try:
        segments, _ = whisper_model.transcribe(path)
        transcript = " ".join([segment.text for segment in segments])
        return transcript
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None

# Summarize text
def summarize(text):
    try:
        if len(text) > 4000:
            text = text[:4000]  # limit input size
        summary = summarizer(text, max_length=250, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        st.error(f"Summarization error: {e}")
        return None

# UI
st.set_page_config(page_title="Free YouTube Summarizer", layout="wide")
st.title("📼 Free AI-Powered YouTube Summarizer")

url = st.text_input("Enter a YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")
if st.button("Generate Summary"):
    if url:
        with st.spinner("Downloading and processing..."):
            audio_file = download_audio(url)

            if audio_file and Path(audio_file).exists():
                transcript = transcribe_audio(audio_file)
                os.remove(audio_file)

                if transcript:
                    st.subheader("📝 Transcript")
                    st.expander("View Transcript").write(transcript)

                    st.subheader("📌 Summary")
                    summary = summarize(transcript)
                    if summary:
                        st.write(summary)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button("Download Summary", summary, file_name="summary.txt")
                        with col2:
                            st.download_button("Download Transcript", transcript, file_name="transcript.txt")
                    else:
                        st.error("Summarization failed.")
                else:
                    st.error("Transcription failed.")
            else:
                st.error("Download failed. Try a different URL.")
    else:
        st.warning("Please enter a valid YouTube URL.")

st.markdown("---")
st.markdown("💡 100% free and API-less using `faster-whisper` + `Hugging Face`")
