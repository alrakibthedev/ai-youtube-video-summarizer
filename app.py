import streamlit as st
import yt_dlp
import os
from pathlib import Path
import tempfile
from transformers import pipeline
from faster_whisper import WhisperModel

# Summarizer pipeline (lightweight)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Whisper model (faster, CPU-only)
@st.cache_resource
def load_whisper():
    return WhisperModel("base", compute_type="int8")

# Download audio from YouTube
def download_audio(url):
    try:
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "%(title)s.%(ext)s")

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            mp3_path = Path(filename).with_suffix(".mp3")
            return mp3_path
    except Exception as e:
        st.error(f"Download error: {e}")
        return None

# Transcribe using faster-whisper
def transcribe_audio(audio_path):
    model = load_whisper()
    segments, _ = model.transcribe(str(audio_path), beam_size=5)
    transcript = " ".join([segment.text for segment in segments])
    return transcript

# Summarize transcript
def summarize_text(text):
    summarizer = load_summarizer()
    if len(text) < 400:
        return "Text too short to summarize."
    return summarizer(text, max_length=300, min_length=60, do_sample=False)[0]["summary_text"]

# Streamlit UI
st.set_page_config(page_title="Free YouTube Summarizer", layout="wide")
st.title("🎥 Free YouTube Video Summarizer (No API Keys)")

url = st.text_input("Enter YouTube video URL:")

if st.button("Summarize Video"):
    if not url:
        st.warning("Please enter a valid YouTube URL.")
    else:
        with st.spinner("Downloading audio..."):
            audio_path = download_audio(url)

        if audio_path and audio_path.exists():
            with st.spinner("Transcribing audio..."):
                transcript = transcribe_audio(audio_path)

            st.subheader("📝 Full Transcript")
            st.expander("Click to view transcript").write(transcript)

            with st.spinner("Summarizing transcript..."):
                summary = summarize_text(transcript)

            st.subheader("📌 Summary")
            st.write(summary)

            # Downloads
            st.download_button("Download Transcript", transcript, file_name="transcript.txt")
            st.download_button("Download Summary", summary, file_name="summary.txt")
        else:
            st.error("Failed to process video.")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Whisper, and Hugging Face.")
