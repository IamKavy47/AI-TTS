import os
import wave
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Create client
client = genai.Client(api_key=api_key)

# Voice list
voices = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
    "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
    "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat"
]

# Helper: save wav
def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)

# --- Interactive selection ---
print("\nAvailable Voices:")
for i, v in enumerate(voices, start=1):
    print(f"{i}. {v}")

choice = int(input("\nChoose a voice (number): "))
voice_name = voices[choice - 1]

prompt = input("Enter the text you want to speak: ")

# Generate TTS
response = client.models.generate_content(
    model="gemini-2.5-flash-preview-tts",
    contents=prompt,
    config=types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_name,
                )
            )
        ),
    ),
)

pcm = response.candidates[0].content.parts[0].inline_data.data

# Save file
file_name = f"tts_{voice_name}.wav"
wave_file(file_name, pcm)

print(f"\n✅ Saved audio as {file_name} with voice '{voice_name}'")
