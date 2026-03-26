"""
Configuration constants for Shuttu AI Assistant
"""

# Audio Configuration
SAMPLE_RATE = 16000
AUDIO_BLOCKSIZE = 8000
COMMAND_BLOCKSIZE = 1280
TTS_RATE = 150  # Words per minute

# Vosk Configuration
VOSK_MODEL_PATH = "model"
VOSK_SPK_MODEL_PATH = "model-spk"
VOSK_WAKE_WORDS = '["shuttu", "[unk]"]'

# LLM Configuration
LLM_MODEL = "phi3"
LLM_TEMPERATURE = 0.8
LLM_NUM_PREDICT = 50

# OpenWakeword Configuration
OWW_CUSTOM_MODEL = "shuttu.onnx"
OWW_FALLBACK_MODEL = "alexa"
OWW_BUFFER_SIZE = 1280  # Samples needed for prediction

# Wake-word Detection
OWW_CONFIDENCE_THRESHOLD = 0.5
OWW_DEBUG_THRESHOLD = 0.3

# Speaker Verification
VOICE_FILE = "boss_voice.npy"
BOSS_DISTANCE_THRESHOLD = 0.25

# Timeouts
COMMAND_TIMEOUT = 5  # seconds (80 * 60ms blocks)
COMMAND_SILENCE_TIMEOUT = 80  # blocks of audio

# Agent Configuration
SYSTEM_PROMPT = """
You are Shuttu, a highly intelligent, sassy, and slightly naughty AI assistant.
You were created by Sruthindev.
Keep responses concise and natural. Be helpful but with personality!
"""

# UI Messages
MESSAGE_SLEEPING = "\n=== SHUTTU is sleeping ===\nSay 'Shuttu' to wake her up...\nSay 'Shuttu, go to sleep' to exit\n"
MESSAGE_LISTENING = "🎤 Listening for 'shuttu'...\n"
MESSAGE_WITHOUT_OWW = "⚠️  Using Vosk-only wake-word detection mode\n"
MESSAGE_LISTENING_CMD = "[LISTENING] Waiting for command..."
MESSAGE_STRANGER = "I only talk to Sruthindev. Who are you?"
