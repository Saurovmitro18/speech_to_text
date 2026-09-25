import os
import nltk
from nltk.corpus import wordnet
import gradio as gr
from faster_whisper import WhisperModel

# Download NLTK data required for dictionary/synonyms
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# -------------------------------------------------------------
# 1. Load Local Speech-to-Text Model (Faster-Whisper)
# -------------------------------------------------------------
# Uses 'base' model on CPU (Lightweight & fast). 
# Options: 'tiny', 'base', 'small', 'medium', 'large-v3'
print("Loading Speech Recognition Model...")
stt_model = WhisperModel("base", device="cpu", compute_type="int8")
print("Model loaded successfully!")

# -------------------------------------------------------------
# 2. NLP Logic: Synonym & Simplification Engine
# -------------------------------------------------------------
def get_synonyms(word: str) -> list:
    """Finds alternative synonyms for a given word using WordNet."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            clean_word = lemma.name().replace('_', ' ')
            if clean_word.lower() != word.lower():
                synonyms.add(clean_word)
    return list(synonyms)[:4] # Return top 4 distinct synonyms

def simplify_and_explain(text: str):
    """Processes transcript to extract vocabulary synonyms and simplified text."""
    if not text.strip():
        return "No text detected.", ""

    words = text.split()
    simplified_words = []
    synonym_map = []

    for word in words:
        clean_word = "".join(filter(str.isalnum, word))
        if len(clean_word) > 4: # Focus on non-trivial words
            syns = get_synonyms(clean_word)
            if syns:
                synonym_map.append(f"**{clean_word}**: {', '.join(syns)}")
                # Replace with first simpler/shorter synonym if available
                shorter_syn = min(syns, key=len)
                if len(shorter_syn) < len(clean_word):
                    simplified_words.append(shorter_syn)
                else:
                    simplified_words.append(word)
            else:
                simplified_words.append(word)
        else:
            simplified_words.append(word)

    simplified_text = " ".join(simplified_words)
    synonyms_output = "\n".join(synonym_map) if synonym_map else "No major complex words found."
    
    return simplified_text, synonyms_output

# -------------------------------------------------------------
# 3. Audio Processing Function
# -------------------------------------------------------------
def process_audio(audio_path):
    if not audio_path:
        return "Please record or upload an audio file.", "", ""

    # Transcribe audio file
    segments, info = stt_model.transcribe(audio_path, beam_size=5)
    transcript = " ".join([segment.text for segment in segments]).strip()

    # Get simplification & synonyms
    simplified_text, synonyms_info = simplify_and_explain(transcript)

    return transcript, simplified_text, synonyms_info

# -------------------------------------------------------------
# 4. Gradio Web Interface Setup
# -------------------------------------------------------------
custom_css = """
footer {visibility: hidden}
"""

with gr.Blocks(title="AI Speech Simplifier", css=custom_css) as app:
    gr.Markdown("# 🎙️ AI Speech-to-Text & Easy English Simplifier")
    gr.Markdown("Speak into your microphone or upload an audio clip. The system will transcribe your voice, simplify complex terms, and display synonyms.")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio Input")
            btn = gr.Button("Process Audio", variant="primary")

        with gr.Column():
            output_transcript = gr.Textbox(label="Original Transcription", lines=3)
            output_simplified = gr.Textbox(label="Simplified English Output", lines=3)
            output_synonyms = gr.Markdown(label="Synonyms Dictionary")

    btn.click(
        fn=process_audio,
        inputs=[audio_input],
        outputs=[output_transcript, output_simplified, output_synonyms]
    )

# -------------------------------------------------------------
# 5. Launch Application
# -------------------------------------------------------------
if __name__ == "__main__":
    app.launch(inbrowser=True)