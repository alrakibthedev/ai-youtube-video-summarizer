import streamlit as st
import openai
import yt_dlp
import os
from pathlib import Path
import tiktoken

# Set up OpenAI client using the new SDK
from openai import OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Configure paths for FFmpeg and FFprobe
FFMPEG_PATH = 'ffmpeg/ffmpeg'
FFPROBE_PATH = 'ffmpeg/ffprobe'

def download_audio(url):
    """Download audio using yt-dlp with FFmpeg/FFprobe for Streamlit Cloud"""
    if not Path(FFMPEG_PATH).exists() or not Path(FFPROBE_PATH).exists():
        st.error("FFmpeg or FFprobe not found. Please upload them to 'ffmpeg/' directory.")
        return None

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
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return Path(ydl.prepare_filename(info)).with_suffix('.mp3')
    except Exception as e:
        st.error(f"Download error: {e}")
        return None

def transcribe_audio(file_path):
    """Transcribe MP3 audio using OpenAI Whisper (v1 SDK)"""
    try:
        with open(file_path, "rb") as audio_file:
            transcript_response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        return transcript_response
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None

def count_tokens(text):
    """Estimate token count for GPT-3.5 input handling"""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

def generate_summary(text, max_tokens=3000):
    """Summarize transcript using GPT-3.5 via ChatCompletion"""
    try:
        if count_tokens(text) > max_tokens:
            # Truncate by characters (approximate, since 1 token ≈ 4 chars)
            text = text[:int(max_tokens * 4)]

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes YouTube video transcripts."},
                {"role": "user", "content": f"Create a detailed summary with key points in bullet format:\n\n{text}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Summarization error: {e}")
        return None

# ---------------------- Streamlit UI ------------------------

st.set_page_config(page_title="YouTube Video Summarizer", layout="wide")
st.title("🎥 AI-Powered YouTube Video Summarizer")

url = st.text_input("Enter YouTube Video URL:", placeholder="https://youtube.com/...")

if st.button("Generate Summary"):
    if url:
        with st.spinner("Processing video..."):
            audio_path = download_audio(url)
            
            if audio_path and audio_path.exists():
                transcript = transcribe_audio(audio_path)
                os.remove(audio_path)  # Clean up after processing
                
                if transcript:
                    st.subheader("📝 Full Transcript")
                    st.expander("View Transcript").write(transcript)
                    
                    st.subheader("📌 AI-Generated Summary")
                    summary = generate_summary(transcript)
                    if summary:
                        st.write(summary)

                        # Download buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button("Download Summary", summary, file_name="summary.txt")
                        with col2:
                            st.download_button("Download Transcript", transcript, file_name="transcript.txt")
                    else:
                        st.error("Summary generation failed.")
                else:
                    st.error("Transcription failed. Try a different video.")
            else:
                st.error("Audio download failed. Check the video URL.")
    else:
        st.warning("Please enter a valid YouTube URL.")

st.markdown("---")
st.markdown("Built with ❤️ using OpenAI, Streamlit, and Python")
