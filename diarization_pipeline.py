import torch
from faster_whisper import WhisperModel

# Load lightweight Whisper model (runs fast on CPU)
print("Loading Whisper STT model with timestamps...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

def process_audio_with_timestamps(audio_path: str):
    """
    Transcribes audio and extracts precise start/end timestamps for each segment.
    """
    if not audio_path:
        return []

    # Transcribe audio with word/segment level timestamps
    segments, info = whisper_model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    
    formatted_transcript = []
    
    for segment in segments:
        formatted_transcript.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip(),
            "speaker": "Speaker 1"  # Default speaker tag (can be dynamically extended)
        })

    return formatted_transcript

# Quick local test if run directly
if __name__ == "__main__":
    test_audio = "sample.wav"  # Optional test file path
    print("Diarization module initialized successfully.")