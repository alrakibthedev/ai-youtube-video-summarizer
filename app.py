import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from transformers import pipeline
import re
import torch

# Set page config FIRST
st.set_page_config(page_title="AI YouTube Summarizer", layout="wide")

# --- Configuration & Model Loading ---
@st.cache_resource
def load_summarizer():
    """Loads the Hugging Face summarization pipeline."""
    try:
        summarizer = pipeline("summarization", 
                            model="sshleifer/distilbart-cnn-12-6",
                            device=-1)  # Force CPU
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    return summarizer

summarizer_pipeline = load_summarizer()

# --- Helper Functions ---
def extract_video_id(youtube_url):
    """Extracts YouTube video ID from URL."""
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    return None

def get_video_transcript(video_id):
    """Fetches transcript with error handling."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([item['text'] for item in transcript_list])
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        return f"Error: {str(e)}"
    except Exception as e:
        if "IP" in str(e):
            return "Error: YouTube is blocking requests. Try a different video or use VPN."
        return f"Error: {str(e)}"

def chunk_text(text, max_chars=2800):
    """Splits text into chunks."""
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def summarize_text(text):
    """Summarizes text in chunks."""
    if not text or not summarizer_pipeline:
        return "Error: No text to summarize"
    
    chunks = chunk_text(text)
    summaries = []
    
    progress_bar = st.progress(0)
    for i, chunk in enumerate(chunks):
        try:
            summary = summarizer_pipeline(chunk, 
                                        max_length=150, 
                                        min_length=30,
                                        do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            st.warning(f"Chunk {i+1} failed: {str(e)}")
        progress_bar.progress((i+1)/len(chunks))
    
    return " ".join(summaries)

# --- UI Components ---
st.title("🎥 AI YouTube Video Summarizer")
st.markdown("""
This tool generates summaries of YouTube videos using AI.  
Enter a YouTube URL below to get started.
""")

# Input Section
url = st.text_input("YouTube Video URL:", 
                   placeholder="https://youtube.com/watch?v=...")

if st.button("Generate Summary"):
    if not url:
        st.warning("Please enter a YouTube URL")
    else:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("Invalid YouTube URL")
        else:
            with st.spinner("Fetching transcript..."):
                transcript = get_video_transcript(video_id)
            
            if transcript.startswith("Error"):
                st.error(transcript)
            else:
                with st.spinner("Generating summary (this may take a minute)..."):
                    summary = summarize_text(transcript)
                
                # Display Results
                st.subheader("Summary")
                st.markdown(f"```\n{summary}\n```")
                
                # Download Buttons
                st.download_button("Download Summary",
                                  data=summary,
                                  file_name="summary.txt")
                
                with st.expander("View Full Transcript"):
                    st.text(transcript)

# Footer
st.markdown("---")
st.markdown("**Note:** Some videos may not work due to YouTube restrictions")
st.markdown("Built with Streamlit 🤖 and Hugging Face Transformers")