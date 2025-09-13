# single.py
# Single-file FastAPI app that serves an HTML/JS frontend (Liquid Glass UI) and a /api/tts endpoint
# Requirements: fastapi, uvicorn, python-dotenv, google-genai, aiofiles
# Install: pip install fastapi uvicorn python-dotenv google-genai aiofiles

import os
import io
import wave
import base64
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# --- Optional: Google GenAI imports ---
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

# Load environment
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

app = FastAPI(title="AI TTS — FastAPI + GSAP frontend")

# Allow local testing from the frontend served by this app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Voices list (with styles)
VOICES = [
    {"name": "Zephyr", "style": "Bright"},
    {"name": "Puck", "style": "Upbeat"},
    {"name": "Charon", "style": "Informative"},
    {"name": "Kore", "style": "Firm"},
    {"name": "Fenrir", "style": "Excitable"},
    {"name": "Leda", "style": "Youthful"},
    {"name": "Orus", "style": "Firm"},
    {"name": "Aoede", "style": "Breezy"},
    {"name": "Callirrhoe", "style": "Easy-going"},
    {"name": "Autonoe", "style": "Bright"},
    {"name": "Enceladus", "style": "Breathy"},
    {"name": "Iapetus", "style": "Clear"},
    {"name": "Umbriel", "style": "Easy-going"},
    {"name": "Algieba", "style": "Smooth"},
    {"name": "Despina", "style": "Smooth"},
    {"name": "Erinome", "style": "Clear"},
    {"name": "Algenib", "style": "Gravelly"},
    {"name": "Rasalgethi", "style": "Informative"},
    {"name": "Laomedeia", "style": "Upbeat"},
    {"name": "Achernar", "style": "Soft"},
    {"name": "Alnilam", "style": "Firm"},
    {"name": "Schedar", "style": "Even"},
    {"name": "Gacrux", "style": "Mature"},
    {"name": "Pulcherrima", "style": "Forward"},
    {"name": "Achird", "style": "Friendly"},
    {"name": "Zubenelgenubi", "style": "Casual"},
    {"name": "Vindemiatrix", "style": "Gentle"},
    {"name": "Sadachbia", "style": "Lively"},
    {"name": "Sadaltager", "style": "Knowledgeable"},
    {"name": "Sulafat", "style": "Warm"},
]

# Helper to write wav file from raw pcm bytes (assuming 16-bit PCM)
def write_wave_bytes(pcm_bytes: bytes, filepath: str, channels: int = 1, rate: int = 24000, sample_width: int = 2):
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_bytes)


# Pydantic schema for the POST body
class TTSRequest(BaseModel):
    voice: str
    text: str


@app.get("/", response_class=HTMLResponse)
async def index():
    # Read the HTML file and inject voices data
    import json
    
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    
    # Replace the voices placeholder with actual data
    html_content = html_content.replace(
        'const voices = [{"name": "Zephyr", "style": "Bright"}',
        f'const voices = {json.dumps(VOICES)}'
    )
    
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/favicon.ico")
async def favicon():
    return Response(content=b"", media_type="image/x-icon")


@app.post("/api/tts")
async def api_tts(req: TTSRequest):
    if not req.text or not req.voice:
        raise HTTPException(status_code=400, detail="voice and text are required")
    if not any(v["name"] == req.voice for v in VOICES):
        raise HTTPException(status_code=400, detail="unknown voice")

    if genai is None or types is None:
        raise HTTPException(status_code=500, detail="google-genai SDK not installed in server environment")
    if not API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not set in environment")

    client = genai.Client(api_key=API_KEY)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=req.text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=req.voice
                        )
                    )
                ),
            ),
        )

        candidate = response.candidates[0]
        part = candidate.content.parts[0]
        data = getattr(part.inline_data, 'data', None)
        if data is None:
            raise HTTPException(status_code=500, detail="no audio data in response")

        if isinstance(data, str):
            try:
                pcm_bytes = base64.b64decode(data)
            except Exception:
                pcm_bytes = data.encode('latin-1')
        else:
            pcm_bytes = bytes(data)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        tmp_path = tmp.name
        tmp.close()
        write_wave_bytes(pcm_bytes, tmp_path, channels=1, rate=24000, sample_width=2)

        return FileResponse(tmp_path, filename='tts.wav', media_type='audio/wav')

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


if __name__ == '__main__':
    uvicorn.run("single:app", host="0.0.0.0", port=8000, reload=True)
