import streamlit as st
import yt_dlp
from pathlib import Path
import os
from transformers import pipeline
from faster_whisper import WhisperModel

# Load models once (cached for performance)
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    whisper_model = WhisperModel("base", compute_type="int8")  # Use 'int8' for low-RAM environments
    return summarizer, whisper_model

summarizer, whisper_model = load_models()

def download_audio(url):
    output_dir = "temp_audio"
    output_path = f"{output_dir}/audio.wav"
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_dir}/audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'quiet': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path
    except Exception as e:
        st.error(f"Download error: {e}")
        return None

def transcribe_audio(path):
    try:
        segments, _ = whisper_model.transcribe(path)
        transcript = " ".join([segment.text for segment in segments])
        return transcript
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None

def summarize(text):
    try:
        # Hugging Face models have input token limits, so truncate large texts
        if len(text) > 4000:
            text = text[:4000]
        summary = summarizer(text, max_length=250, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        st.error(f"Summarization error: {e}")
        return None

# UI
st.set_page_config(page_title="🎥 Free YouTube Summarizer", layout="wide")
st.title("🎬 Free AI-Powered YouTube Summarizer (No API Required)")

url = st.text_input("Enter a YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Generate Summary"):
    if url:
        with st.spinner("Downloading and processing video..."):
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
                st.error("Download failed. Try another video.")
    else:
        st.warning("Please enter a valid YouTube URL.")

st.markdown("---")
st.markdown("🔓 Built with `yt-dlp`, `faster-whisper`, `transformers`, and ❤️ by the open source community.")
