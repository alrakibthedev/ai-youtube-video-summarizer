import streamlit as st
import openai
import yt_dlp
import os
from pathlib import Path
import tiktoken
import ffmpeg

# Set up OpenAI API
openai.api_key = st.secrets["OPENAI_API_KEY"]

def download_audio(url):
    """Download audio from YouTube using yt-dlp with FFmpeg and FFprobe path specification"""
    ffmpeg_path = 'ffmpeg/ffmpeg'
    ffprobe_path = 'ffmpeg/ffprobe'

    if not Path(ffmpeg_path).exists() or not Path(ffprobe_path).exists():
        st.error("FFmpeg or FFprobe not found at the specified path. Please ensure they are installed.")
        return None

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'temp_audio/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'ffmpeg_location': ffmpeg_path,
        'ffprobe_location': ffprobe_path,  # Add this line
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
    """Transcribe audio using OpenAI Whisper"""
    try:
        with open(file_path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create()
        return transcript
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None

def count_tokens(text):
    """Count tokens for GPT-3.5 input limit"""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

def generate_summary(text, max_tokens=3000):
    """Generate summary using GPT-3.5 with token limit handling"""
    try:
        if count_tokens(text) > max_tokens:
            text = text[:int(max_tokens * 3.5)]  # Approximate character limit
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes video transcripts."},
                {"role": "user", "content": f"Create a detailed summary with key points in bullet format:\n\n{text}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message['content']
    except Exception as e:
        st.error(f"Summarization error: {e}")
        return None

# Streamlit UI
st.set_page_config(page_title="YouTube Video Summarizer", layout="wide")
st.title("🎥 AI-Powered YouTube Video Summarizer")

url = st.text_input("Enter YouTube Video URL:", placeholder="https://youtube.com/...")

if st.button("Generate Summary"):
    if url:
        with st.spinner("Processing video..."):
            # Download audio
            audio_path = download_audio(url)
            
            if audio_path and audio_path.exists():
                # Transcribe audio
                transcript = transcribe_audio(audio_path)
                os.remove(audio_path)  # Cleanup audio file
                
                if transcript:
                    # Display transcript
                    st.subheader("📝 Full Transcript")
                    st.expander("View Transcript").write(transcript)
                    
                    # Generate and display summary
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
                        st.error("Failed to generate summary")
                else:
                    st.error("Transcription failed. Please try another video.")
            else:
                st.error("Could not download video. Please check URL or try another video.")
    else:
        st.warning("Please enter a valid YouTube URL")

st.markdown("---")
st.markdown("Built with ❤️ using OpenAI, Streamlit, and Python")
