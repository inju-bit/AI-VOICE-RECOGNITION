# AI Voice Recognition — NOVA AI Voice Assistant

A local AI-powered voice assistant built using **Python, FastAPI, Faster-Whisper, and Ollama**. The application converts spoken language into text, processes questions using a locally running large language model, and delivers spoken responses through a modern web interface.

## Features

- **Speech Recognition:** Converts English speech into text using Faster-Whisper.
- **AI Conversation:** Generates intelligent responses using Ollama with Llama 3.2 1B.
- **Text-to-Speech:** Reads AI responses aloud using browser speech synthesis.
- **Interactive Web Interface:** Responsive voice assistant dashboard.
- **Local AI Processing:** Processes speech and AI requests locally.
- **Microphone Integration:** Records audio directly from the browser.
- **REST API:** FastAPI backend for transcription and AI chat.
- **Terminal Mode:** Standalone Python application for microphone-based speech recognition.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Backend | FastAPI |
| Speech Recognition | Faster-Whisper |
| Large Language Model | Llama 3.2 1B |
| Local LLM Runtime | Ollama |
| Frontend | HTML, CSS, JavaScript |
| Audio Processing | NumPy, Web Audio API |
| Text-to-Speech | Browser Speech Synthesis |
| Server | Uvicorn |

## Project Structure

```text
AI-VOICE-RECOGNITION/
├── app.py
├── main.py
├── requirements.txt
├── .gitignore
├── README.md
└── static/
    ├── index.html
    ├── script.js
    └── style.css
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-VOICE-RECOGNITION.git
cd AI-VOICE-RECOGNITION
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and Configure Ollama

Download Ollama from https://ollama.com/.

Install the language model:

```bash
ollama pull llama3.2:1b
```

Ensure Ollama is running. If necessary, start it using:

```bash
ollama serve
```

## Running the Application

### Web-Based AI Voice Assistant

Start the FastAPI backend:

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open the application:

http://127.0.0.1:8000

Allow microphone permissions in your browser.

You can now record speech, view recognised text, ask AI questions and listen to spoken responses.

### Terminal-Based Speech Recognition

Run:

```bash
python main.py
```

Press Enter to start a five-second recording.

The recognised speech appears in the terminal.

Type `q` to exit.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Loads the web interface |
| GET | `/health` | Returns application status |
| POST | `/transcribe` | Converts recorded WAV audio into text |
| POST | `/chat` | Generates AI responses using Ollama |

## How It Works

1. The user records speech using the microphone.
2. The application captures and prepares the audio.
3. Faster-Whisper converts speech into text.
4. The transcription is displayed in the conversation interface.
5. The text is sent to the local Ollama language model.
6. Llama 3.2 generates a response.
7. The frontend displays the AI response and can read it aloud using browser speech synthesis.

## System Requirements

- Python 3.10 or later, with compatible dependencies
- Microphone
- Modern web browser
- Ollama installed locally for AI chat
- Internet connection for initial model downloads

The application uses CPU inference, so a dedicated NVIDIA GPU is not required.

## Limitations

- The speech recognition model is configured for English.
- Model inference speed depends on the computer's CPU.
- Ollama must be running for AI conversation features.
- Voice output quality depends on browser and operating-system voices.

## Future Improvements

- Multilingual speech recognition
- Conversation memory
- Wake-word activation
- Improved background-noise handling
- Streaming speech recognition
- Custom text-to-speech voices
- Support for additional local and cloud-based language models
- Real-time voice conversation

## Author

Developed as a practical AI engineering project exploring speech recognition, local large language models, REST API development and voice-based human-computer interaction.

## License

No license has been specified for this project.