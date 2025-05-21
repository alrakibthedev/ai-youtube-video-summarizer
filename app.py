import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from pytube import YouTube
import os
import librosa
import torch

# Set page config
st.set_page_config(
    page_title="YouTube Video Summarizer",
    page_icon="▶️",
    layout="wide"
)

# Initialize models
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
    return summarizer, asr

summarizer, asr = load_models()

def get_transcript(video_id):
    try:
        # Try to get existing subtitles
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry['text'] for entry in transcript])
        return text, "subtitles"
    except:
        return None, None

def download_audio(url):
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_path = audio_stream.download()
    return audio_path

def transcribe_audio(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = asr(audio, return_timestamps=False)
    return inputs['text']

def summarize_text(text):
    chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
    summaries = []
    for chunk in chunks:
        summaries.append(summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text'])
    return " ".join(summaries)

# UI Components
st.title("YouTube Video Summarizer")
url = st.text_input("Enter YouTube Video URL:")

if url:
    video_id = url.split("v=")[1].split("&")[0]
    transcript, source = get_transcript(video_id)
    
    if transcript:
        st.write("Using existing subtitles")
        with st.spinner("Summarizing..."):
            summary = summarize_text(transcript)
    else:
        st.warning("No subtitles found. Transcribing audio (This may take a few minutes)")
        audio_path = download_audio(url)
        transcript = transcribe_audio(audio_path)
        os.remove(audio_path)
        with st.spinner("Summarizing..."):
            summary = summarize_text(transcript)
    
    st.subheader("Summary:")
    st.write(summary)
    
    with st.expander("Show Transcript"):
        st.write(transcript)