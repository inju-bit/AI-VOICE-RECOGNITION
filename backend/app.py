
# ==========================================
# AI VOICE ASSISTANT - FASTAPI BACKEND
# ==========================================
#
# Speech-to-Text: Faster-Whisper
# AI Brain: Ollama (Llama 3.2 1B)
# Text-to-Speech: Browser JavaScript
# Backend: FastAPI
#
# ==========================================

import io
import json
import wave
import urllib.request
import urllib.error

import numpy as np

from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from faster_whisper import WhisperModel


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

WHISPER_MODEL = "base.en"
OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.003


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="AI Voice Assistant",
    description="Local voice assistant using Whisper and Ollama",
    version="1.0.0"
)

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "running",
        "whisper_model": WHISPER_MODEL,
        "ollama_model": OLLAMA_MODEL
    }


# ==========================================
# LOAD WHISPER
# ==========================================

print("🧠 Loading Whisper speech recognition model...")

model = WhisperModel(
    WHISPER_MODEL,
    device="cpu",
    compute_type="int8"
)

print("✅ Whisper model loaded!")


# ==========================================
# AUDIO DECODING
# ==========================================

def decode_wav(data: bytes):

    with wave.open(io.BytesIO(data), "rb") as wav:

        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise ValueError("Expected 16-bit PCM WAV audio.")

    if channels not in (1, 2):
        raise ValueError("Expected mono or stereo audio.")

    if sample_rate != SAMPLE_RATE:
        raise ValueError("Expected 16000 Hz audio.")

    samples = np.frombuffer(
        frames,
        dtype=np.int16
    ).astype(np.float32) / 32768.0

    if channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)

    return samples


# ==========================================
# SPEECH TO TEXT - WHISPER
# ==========================================

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):

    try:
        data = await audio.read()

        if not data:
            raise ValueError("Empty recording.")

        samples = decode_wav(data)

        if samples.size == 0:
            return {
                "text": "",
                "message": "No audio detected."
            }

        volume = float(
            np.sqrt(np.mean(samples ** 2))
        )

        print(f"🔊 Audio volume: {volume:.5f}")

        if volume < SILENCE_THRESHOLD:
            return {
                "text": "",
                "message": "Audio too quiet."
            }

        print("🎤 Transcribing audio...")

        # Direct NumPy array avoids PyAV WAV
        # decoding compatibility problems.
        segments, info = model.transcribe(
            samples,
            language="en",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

        print(f"📝 Recognised: {text}")

        return {
            "text": text,
            "message": (
                "Success" if text
                else "No speech recognised."
            )
        }

    except (ValueError, wave.Error) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        print(f"❌ Transcription error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Audio transcription failed."
        )


# ==========================================
# AI REQUEST MODEL
# ==========================================

class ChatRequest(BaseModel):
    message: str


# ==========================================
# AI CHAT - OLLAMA
# ==========================================

def generate_ai_response(question: str):

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful and friendly "
                    "AI voice assistant. "
                    "Speak naturally like a human. "
                    "Keep your answers concise, "
                    "usually one to three sentences. "
                    "Avoid markdown, bullet points, "
                    "and unnecessary technical jargon. "
                    "If you do not know something, "
                    "say so clearly."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.7
        }
    }

    encoded_data = json.dumps(
        payload
    ).encode("utf-8")

    request = urllib.request.Request(
        OLLAMA_URL,
        data=encoded_data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=120
        ) as response:

            result = json.load(response)

        answer = result["message"]["content"].strip()

        if not answer:
            raise ValueError("AI returned an empty answer.")

        return answer

    except urllib.error.HTTPError as error:
        details = error.read().decode(
            "utf-8", errors="replace"
        )

        print(f"❌ Ollama HTTP error: {details}")

        raise HTTPException(
            status_code=502,
            detail=(
                "Ollama could not generate a response. "
                "Check that llama3.2:1b is installed."
            )
        )

    except urllib.error.URLError as error:
        print(f"❌ Ollama connection error: {error}")

        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama is not running. "
                "Please open Ollama on your Mac."
            )
        )

    except (KeyError, ValueError) as error:
        print(f"❌ Invalid Ollama response: {error}")

        raise HTTPException(
            status_code=502,
            detail="Invalid AI model response."
        )


# ==========================================
# CHAT ENDPOINT
# ==========================================

@app.post("/chat")
def chat_with_ai(request: ChatRequest):

    question = request.message.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please ask a question."
        )

    print(f"👤 User: {question}")

    answer = generate_ai_response(question)

    print(f"🤖 Assistant: {answer}")

    return {
        "question": question,
        "answer": answer,
        "model": OLLAMA_MODEL
    }


# ==========================================
# SERVER STARTUP
# ==========================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )
