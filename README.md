# Shuttu - AI Voice Assistant

A sophisticated AI voice assistant built with Vosk speech recognition and LangChain, featuring speaker identification and tool integration.

## Features

- 🎤 **Speech Recognition** - Real-time voice input using Vosk
- 🔊 **Text-to-Speech** - Natural language output with pyttsx3
- 🧠 **LLM Integration** - Powered by Ollama with phi3 model
- 🎯 **Speaker Identification** - Voice fingerprinting for secure access
- 🛠️ **Tool Integration** - Extensible tools (web browsing, etc.)
- 💬 **Conversation Memory** - Maintains chat history

## Prerequisites

- Python 3.8+
- Ollama installed and running
- Vosk model files (see setup below)

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd Blender
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Vosk Model (Required)

**⚠️ Important**: You need to download the Vosk speech recognition model before running the application.

**Recommended**: Download the **1.8 GB English model** (best accuracy for English):

- **Website**: https://alphacephei.com/vosk/models
- **Model**: `vosk-model-en-us-0.42-gigaspeech` (~1.8 GB)

**Steps**:

1. Download the model from the link above
2. Extract the ZIP file
3. Rename the extracted folder to `model`
4. Place it in the project root directory:
   ```
   Blender/
   ├── main.py
   ├── model.py
   ├── model/          ← Extract here
   └── ...
   ```

**Alternative smaller models** (if disk space is limited):

- `vosk-model-small-en-us-0.15` (~40 MB) - Lower accuracy, faster
- `vosk-model-en-us-0.22` (~350 MB) - Medium accuracy

### 4. Ensure Ollama is running

```bash
# Start Ollama service
ollama serve
```

In another terminal, pull the phi3 model:

```bash
ollama pull phi3
```

## Usage

```bash
python main.py
```

## Project Structure

```
Blender/
├── main.py           - Main application entry point
├── model.py          - Model utilities
├── model/            - Vosk speech recognition models (not tracked in git)
├── README.md         - This file
├── .gitignore        - Git ignore rules
└── requirements.txt  - Python dependencies
```

## Configuration

- **Vosk Model Path**: `model/` (in project root)
- **Speech Recognition**: English (en-us)
- **LLM Model**: phi3 (via Ollama)
- **TTS Engine**: pyttsx3
- **Voice Sample Rate**: 16000 Hz

## Important Notes

⚠️ **Model files are NOT tracked in git** - You need to download them locally after cloning.

The `.gitignore` file excludes:

- Model directories and files
- Python cache and virtual environments
- Generated voice fingerprints

## Features in Detail

### Speaker Identification

- Voice enrollment (captures 5 samples)
- Creates voice fingerprint saved as `boss_voice.npy`
- Verifies speaker identity on subsequent uses

### Tools

- Browser integration for web access
- Extensible tool framework via LangChain
- Context-aware tool usage with memory

## Troubleshooting

**"Vosk model not found" error**:

- Ensure you've downloaded and extracted the model to `model/` folder
- Verify the model path is correct

**"Model initialisation failed" error**:

- Ensure Ollama is running: `ollama serve`
- Verify phi3 model is installed: `ollama list`

**Audio capture issues**:

- Check microphone permissions
- Ensure sounddevice is properly installed

## Requirements

See `requirements.txt` for complete dependencies:

- langchain
- langchain-ollama
- langchain-community
- vosk
- pyttsx3
- sounddevice
- numpy
- scipy

## Author

Created by Sruthindev

## License

[Specify your license here]
