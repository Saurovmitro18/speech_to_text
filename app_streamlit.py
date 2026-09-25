import streamlit as st
import requests

# Set page configuration
st.set_page_config(
    page_title="Multimodal Meeting & Lecture Copilot",
    page_icon="🎙️",
    layout="wide"
)

API_BASE_URL = "http://127.0.0.1:8000"

st.title("🎙️ Multimodal Audio Intelligence Copilot")
st.caption("Powered by Whisper, ChromaDB Vector Search, FastAPI & LLMs")

# Create two primary UI layout columns
col1, col2 = st.columns([1, 1])

# -------------------------------------------------------------
# COLUMN 1: Audio Upload & Interactive Transcript
# -------------------------------------------------------------
with col1:
    st.header("1. Upload & Transcribe")
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a", "ogg"])

    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        if st.button("Process Audio Pipeline", type="primary"):
            with st.spinner("Transcribing audio and indexing into Vector DB..."):
                try:
                    # Send audio file to FastAPI backend
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_BASE_URL}/upload-audio/", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state["transcript"] = data["transcript"]
                        st.success(f"Successfully processed {data['segments_count']} transcript segments!")
                    else:
                        st.error(f"Error processing audio: {response.json().get('detail')}")
                except Exception as e:
                    st.error(f"Failed to connect to FastAPI backend: {str(e)}")

    # Render timestamped transcript if available
    if "transcript" in st.session_state:
        st.subheader("Timestamped Transcript")
        for block in st.session_state["transcript"]:
            with st.expander(f"⏱️ [{block['start']}s - {block['end']}s] {block['speaker']}"):
                st.write(block["text"])

# -------------------------------------------------------------
# COLUMN 2: Vector Search & LLM Summaries (RAG)
# -------------------------------------------------------------
with col2:
    st.header("2. Search & Summarize (RAG)")
    
    # RAG Search Section
    st.subheader("🔎 Semantic Vector Search")
    search_query = st.text_input("Search for specific topics, decisions, or quotes:")
    
    if st.button("Search Transcript"):
        if search_query.strip():
            with st.spinner("Searching vector database..."):
                try:
                    res = requests.post(
                        f"{API_BASE_URL}/search-transcript/",
                        json={"query": search_query}
                    )
                    if res.status_code == 200:
                        results = res.json().get("results", [])
                        if results:
                            for match in results:
                                st.info(f"**[{match['timestamp']}] {match['speaker']}**:\n\n{match['text']}")
                        else:
                            st.warning("No relevant matches found.")
                    else:
                        st.error(f"Search failed: {res.json().get('detail')}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
        else:
            st.warning("Please enter a search query.")

    st.divider()

    # LLM Summary Generation
    st.subheader("🤖 AI Meeting Summary & Action Items")
    llm_provider = st.selectbox("Select LLM Provider", ["groq", "ollama"])
    
    if st.button("Generate Summary"):
        with st.spinner("Generating summary via LLM..."):
            try:
                res = requests.post(
                    f"{API_BASE_URL}/generate-summary/",
                    json={"llm_provider": llm_provider}
                )
                if res.status_code == 200:
                    summary_text = res.json().get("summary", "No summary generated.")
                    st.markdown("### Executive Summary")
                    st.write(summary_text)
                else:
                    st.error(f"Summary generation failed: {res.json().get('detail')}")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")