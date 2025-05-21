import streamlit as st
from transformers import pipeline
import yt_dlp
import os
from pathlib import Path
from faster_whisper import WhisperModel

# Setup
st.set_page_config(page_title="YouTube Summarizer", layout="wide")
st.title("🎬 YouTube Video Summarizer (Free & Local)")

FFMPEG_PATH = "ffmpeg/ffmpeg"
FFPROBE_PATH = "ffmpeg/ffprobe"

def download_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'temp_audio/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'ffmpeg_location': FFMPEG_PATH,
        'ffprobe_location': FFPROBE_PATH,
        'quiet': True
    }

    try:
        os.makedirs("temp_audio", exist_ok=True)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return Path(ydl.prepare_filename(info)).with_suffix('.mp3')
    except Exception as e:
        st.error(f"Download error: {e}")
        return None

def transcribe_audio(audio_path):
    try:
        model = WhisperModel("base", compute_type="int8")
        segments, _ = model.transcribe(str(audio_path), beam_size=5)
        transcript = " ".join([segment.text for segment in segments])
        return transcript
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None

@st.cache_resource
def get_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_text(text):
    try:
        summarizer = get_summarizer()
        if len(text) > 3000:
            text = text[:3000]
        summary = summarizer(text, max_length=180, min_length=30, do_sample=False)[0]["summary_text"]
        return summary
    except Exception as e:
        st.error(f"Summarization error: {e}")
        return None

url = st.text_input("Enter YouTube URL")

if st.button("Generate Summary"):
    if not url:
        st.warning("Please provide a valid YouTube URL.")
    else:
        with st.spinner("Downloading audio..."):
            audio_file = download_audio(url)

        if audio_file and audio_file.exists():
            with st.spinner("Transcribing audio..."):
                transcript = transcribe_audio(audio_file)
            os.remove(audio_file)

            if transcript:
                st.subheader("📝 Transcript")
                st.expander("View Full Transcript").write(transcript)

                with st.spinner("Generating summary..."):
                    summary = summarize_text(transcript)

                if summary:
                    st.subheader("📌 Summary")
                    st.write(summary)
                    st.download_button("Download Summary", summary, file_name="summary.txt")
                    st.download_button("Download Transcript", transcript, file_name="transcript.txt")
        else:
            st.error("Audio download failed.")
