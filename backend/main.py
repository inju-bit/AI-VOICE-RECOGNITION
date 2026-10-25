
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# ==================================
# SETTINGS
# ==================================

SAMPLE_RATE = 16000
DURATION = 5
MODEL_SIZE = "base.en"

# ==================================
# LOAD WHISPER MODEL
# ==================================

print("🧠 Loading Whisper AI model...")

model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8"
)

print("✅ Whisper AI loaded successfully!")


# ==================================
# RECORD AUDIO
# ==================================

def record_audio():
    print("\n🎤 Speak now! Recording for 5 seconds...")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    print("✅ Recording completed!")

    return audio.flatten()


# ==================================
# RECOGNISE SPEECH
# ==================================

def recognize_speech(audio):
    print("🧠 Recognising speech...")

    # Check microphone volume
    volume = np.sqrt(np.mean(audio ** 2))

    print(f"🔊 Audio level: {volume:.5f}")

    if volume < 0.003:
        print("⚠️ Audio is too quiet. No speech detected.")
        return ""

    # Send NumPy audio directly to Whisper
    # Avoids the PyAV metadata_errors issue
    segments, info = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    ).strip()

    if not text:
        print("⚠️ No speech recognised.")
        return ""

    print("\n" + "=" * 45)
    print("📝 YOU SAID:")
    print(text)
    print("=" * 45)

    return text


# ==================================
# MAIN APPLICATION
# ==================================

def main():
    print("\n🎙️ AI VOICE RECOGNITION")
    print("Press ENTER to start recording.")
    print("Type 'q' to quit.")

    while True:
        command = input("\nPress ENTER to speak (q to quit): ")

        if command.strip().lower() == "q":
            print("👋 Goodbye!")
            break

        try:
            audio = record_audio()
            recognize_speech(audio)

        except Exception as error:
            print(f"❌ Error: {error}")


if __name__ == "__main__":
    main()
