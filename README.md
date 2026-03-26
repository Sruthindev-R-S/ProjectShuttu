# Shuttu - AI Voice Assistant

A sophisticated AI voice assistant built with Vosk speech recognition and LangChain, featuring speaker identification and advanced tool integration.

## Overview

Shuttu is an intelligent voice-controlled assistant that combines state-of-the-art speech recognition with large language models to create natural, context-aware conversations. It can identify different speakers for personalized responses and integrate with various external tools.

## Features

- 🎤 **Speech Recognition** - Real-time voice input using Vosk (offline)
- 🔊 **Text-to-Speech** - Natural language output with pyttsx3
- 🧠 **LLM Integration** - Powered by Ollama with phi3 model
- 🎯 **Speaker Identification** - Voice fingerprinting for secure, personalized access
- 🛠️ **Tool Integration** - Extensible architecture for web browsing and custom tools
- 💬 **Conversation Memory** - Maintains chat history for context-aware responses
- 🔒 **Privacy-First** - Runs locally without external API dependencies

## Prerequisites

- Python 3.13+ (recommended) or 3.8+
- Ollama installed and running locally
- Microphone for voice input
- Vosk model files (detailed setup below)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Sruthindev-R-S/ProjectShuttu.git
cd ProjectShuttu
```

### 2. Create and activate virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Vosk Model (Required)

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
   ProjectShuttu/
   ├── main.py
   ├── model.py
   ├── model/          ← Extract here
   └── ...
   ```

**Alternative smaller models** (if disk space is limited):

- `vosk-model-small-en-us-0.15` (~40 MB) - Lower accuracy, faster
- `vosk-model-en-us-0.22` (~350 MB) - Medium accuracy

### 5. Ensure Ollama is running

```bash
# Start Ollama service
ollama serve
```

In another terminal, pull the phi3 model:

```bash
ollama pull phi3
```

## Quick Start

### 1. Activate virtual environment (if not already activated)

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Ensure Ollama is running

```bash
# In a separate terminal, start Ollama
ollama serve
```

### 3. Run the application

```bash
python main.py
```

## Usage

Once running, speak naturally to the assistant:

```
You: "What's the weather?"
Shuttu: [Processes voice input and responds with weather information]
```

Press `Ctrl+C` to exit the application.

## Project Structure

```
ProjectShuttu/
├── main.py              - Main application entry point
├── model.py             - Model utilities and core logic
├── model/               - Vosk speech recognition models (not tracked in git)
├── model-spk/           - Speaker identification models
├── boss_voice.npy       - Voice sample for speaker identification
├── requirements.txt     - Python dependencies
├── pyvenv.cfg           - Virtual environment configuration
├── README.md            - This file
├── WAKEWORD_SETUP.md    - Wakeword configuration guide
├── .gitignore           - Git ignore rules
└── venv/                - Virtual environment (not tracked in git)
```

## Configuration

### Default Settings

- **Vosk Model Path**: `model/` (in project root)
- **Speaker Model Path**: `model-spk/` (for voice ID)
- **Speech Recognition**: English (en-US)
- **Sample Rate**: 16000 Hz
- **LLM Model**: phi3 (via Ollama)
- **TTS Engine**: pyttsx3

### Environment Variables

Create a `.env` file in the project root for custom configuration:

```bash
OLLAMA_HOST=127.0.0.1
OLLAMA_PORT=11434
VOSK_MODEL_PATH=./model
VOICE_SAMPLE_RATE=16000
```

## Troubleshooting

### Issue: Vosk model not found

**Solution**: Ensure the model folder is properly extracted in the project root and named exactly `model/`.

### Issue: Ollama connection failed

**Solution**:

```bash
# Start Ollama service
ollama serve

# Verify phi3 is installed
ollama list
```

### Issue: Microphone not detected

**Solution**:

- Check system audio input settings
- Verify microphone is not being used by another application
- Try restarting the application

### Issue: Speaker identification not working

**Solution**: Ensure `boss_voice.npy` and `model-spk/` folder are present in the project root.

## Dependencies

Key packages used:

- **vosk** - Speech recognition
- **langchain** - LLM orchestration
- **pyttsx3** - Text-to-speech
- **sounddevice** - Audio I/O
- **numpy** - Numerical computing
- **scipy** - Scientific computing

For complete list, see [requirements.txt](requirements.txt).

## Features in Detail

### Speaker Identification

- Voice enrollment (captures samples for fingerprinting)
- Creates voice fingerprint saved as `boss_voice.npy`
- Verifies speaker identity on subsequent uses
- Uses model files in `model-spk/` directory

### Tools

- Browser integration for web access
- Extensible tool framework via LangChain
- Context-aware tool usage with memory

## Important Notes

⚠️ **Model files are NOT tracked in git** - You need to download them locally after cloning.

The `.gitignore` file excludes:

- Model directories and files (`model/`, `model-spk/`)
- Python cache and virtual environments
- Generated voice fingerprints (`boss_voice.npy`)
- IDE and OS-specific files

## Additional Resources

- [Vosk Models](https://alphacephei.com/vosk/models/) - Download speech recognition models
- [Ollama](https://ollama.ai/) - Run LLMs locally
- [LangChain Docs](https://python.langchain.com/) - LLM integration framework
- [WAKEWORD_SETUP.md](WAKEWORD_SETUP.md) - Advanced wakeword configuration

## License

[Add your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues or questions, please open an issue on the [GitHub repository](https://github.com/Sruthindev-R-S/ProjectShuttu/issues).

- **requests**: HTTP library

## Author

Created by Sruthindev

## License

MIT License - See LICENSE file for details
