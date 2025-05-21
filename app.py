import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from transformers import pipeline
import re
import torch # Required by transformers, ensure it's installed

# --- Configuration & Model Loading ---

# Cache the summarization pipeline for efficiency
@st.cache_resource
def load_summarizer():
    """Loads the Hugging Face summarization pipeline."""
    # Using a distilled BART model for a balance of speed and quality
    # device=0 for GPU if available, device=-1 for CPU.
    # Let transformers pipeline auto-detect or default to CPU if GPU not configured/available.
    # For Streamlit sharing, CPU is more common for free tier.
    try:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

    except Exception as e:
        st.error(f"Error loading summarization model: {e}. Make sure you have an internet connection for the first download.")
        summarizer = None
    return summarizer

summarizer_pipeline = load_summarizer()

# --- Helper Functions ---

def extract_video_id(youtube_url):
    """
    Extracts the video ID from various YouTube URL formats.
    Returns video ID string or None if not found.
    """
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([a-zA-Z0-9_-]{11})',
        r'(?:https?:\/\/)?m\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    return None

def get_video_transcript(video_id):
    """
    Fetches the transcript for a given YouTube video ID.
    Returns the transcript text or an error message string.
    """
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = " ".join([item['text'] for item in transcript_list])
        return full_transcript
    except TranscriptsDisabled:
        return "Error: Transcripts are disabled for this video."
    except NoTranscriptFound:
        return "Error: No transcript found for this video. It might be that the video has no subtitles or they are not available in a processable format."
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

def chunk_text_by_sentences(text, max_chars_per_chunk=2800):
    """
    Splits text into chunks based on sentences, aiming for a max character length per chunk.
    This is a heuristic to prepare text for models with token limits.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' ')) # Split by sentences
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # If adding the new sentence exceeds the max_chars_per_chunk,
        # store the current_chunk and start a new one.
        if len(current_chunk) + len(sentence) + 1 > max_chars_per_chunk and current_chunk:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
                
    if current_chunk: # Add the last remaining chunk
        chunks.append(current_chunk)
        
    if not chunks and text.strip(): # Handle very short texts that might not form chunks
        chunks.append(text)
        
    return chunks

def summarize_text_chunks(text_to_summarize):
    """
    Summarizes text by breaking it into manageable chunks first.
    Uses the pre-loaded summarizer_pipeline.
    """
    if not summarizer_pipeline:
        return "Error: Summarization model not loaded."
    if not text_to_summarize or not text_to_summarize.strip():
        return "Error: Input text is empty."

    # Chunk the text
    # The model 'sshleifer/distilbart-cnn-12-6' has a max input token limit of 1024.
    # We chunk by characters as an approximation. 2800 chars ~ 700-900 tokens.
    chunks = chunk_text_by_sentences(text_to_summarize, max_chars_per_chunk=2800)

    if not chunks:
         return "Error: Could not split text into processable chunks."

    all_summaries = []
    st.write(f"Total chunks to summarize: {len(chunks)}")

    progress_bar = st.progress(0)
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        try:
            # Adjust min/max length for summary based on chunk length
            # These are output length constraints for the summary of EACH chunk
            num_words_in_chunk = len(chunk.split())
            min_len = max(15, int(num_words_in_chunk * 0.1))  # Min 10% of chunk words, or 15
            max_len = max(40, int(num_words_in_chunk * 0.4))  # Max 40% of chunk words, or 40
            
            # Cap max_len to something reasonable for this model (default max is 142)
            max_len = min(max_len, 140)
            if min_len >= max_len: # Ensure min_len is less than max_len
                min_len = max(10, max_len - 20)


            summary_part = summarizer_pipeline(chunk, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
            all_summaries.append(summary_part)
        except Exception as e:
            st.warning(f"Could not summarize chunk {i+1}: {e}. Skipping this chunk.")
            # Fallback: append a placeholder or a snippet of the chunk
            # all_summaries.append(f"[Error summarizing part: {chunk[:50]}...]") 
        progress_bar.progress((i + 1) / len(chunks))
        
    final_summary = " ".join(all_summaries)
    
    if not final_summary.strip() and text_to_summarize.strip():
        return "Could not generate a summary. The content might be too short or not suitable for summarization with the current settings."
    return final_summary

# --- Streamlit UI ---
st.set_page_config(page_title="AI YouTube Summarizer", layout="wide")
st.title("📺 AI-Powered YouTube Video Summarizer")
st.markdown("""
This tool uses AI to generate a concise summary of a YouTube video. 
Enter a YouTube video URL below, and the system will attempt to fetch its transcript and summarize it.
This uses free tools: `youtube-transcript-api` for transcripts and a Hugging Face model for summarization.
""")

# Initialize session state variables
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'video_title_for_download' not in st.session_state:
    st.session_state.video_title_for_download = "video"


youtube_url = st.text_input("Enter YouTube Video URL:", placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ")

if st.button("✨ Generate Summary"):
    if not youtube_url:
        st.warning("Please enter a YouTube URL.")
    elif not summarizer_pipeline:
        st.error("Summarization service is not available. Please check model loading status.")
    else:
        video_id = extract_video_id(youtube_url)
        if video_id:
            st.session_state.video_title_for_download = video_id # Use video ID for filename
            st.info(f"Extracted Video ID: {video_id}")
            with st.spinner("Fetching transcript... 📄"):
                transcript_result = get_video_transcript(video_id)
            
            if transcript_result.startswith("Error:"):
                st.error(transcript_result)
                st.session_state.transcript = ""
                st.session_state.summary = ""
            else:
                st.session_state.transcript = transcript_result
                st.success("Transcript fetched successfully!")
                
                with st.spinner("Summarizing text... 🧠 This may take a few moments for longer videos."):
                    summary_result = summarize_text_chunks(st.session_state.transcript)
                
                if summary_result.startswith("Error:"):
                    st.error(summary_result)
                    st.session_state.summary = ""
                else:
                    st.session_state.summary = summary_result
                    st.success("Summary generated successfully! 🎉")
        else:
            st.error("Invalid YouTube URL. Please check the URL and try again.")
            st.session_state.transcript = ""
            st.session_state.summary = ""

# Display transcript and summary if available
if st.session_state.transcript:
    st.subheader("📜 Video Transcript")
    with st.expander("View/Hide Transcript", expanded=False):
        st.markdown(f"<div style='height: 300px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; border-radius: 5px;'>{st.session_state.transcript}</div>", unsafe_allow_html=True)
    
    st.download_button(
        label="📥 Download Transcript (.txt)",
        data=st.session_state.transcript,
        file_name=f"{st.session_state.video_title_for_download}_transcript.txt",
        mime="text/plain"
    )

if st.session_state.summary:
    st.subheader("📝 Generated Summary")
    st.markdown(f"<div style='background-color:#f0f2f6; padding: 15px; border-radius: 5px;'>{st.session_state.summary}</div>", unsafe_allow_html=True)
    st.download_button(
        label="📥 Download Summary (.txt)",
        data=st.session_state.summary,
        file_name=f"{st.session_state.video_title_for_download}_summary.txt",
        mime="text/plain"
    )

st.markdown("---")
st.markdown("Developed as part of a Software Development Project.")
st.markdown("Using: Streamlit, youtube-transcript-api, Hugging Face Transformers.")