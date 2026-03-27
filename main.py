"""
Shuttu - AI Voice Assistant
All code consolidated into a single file for easier understanding
"""

import os
import sys
import json
import queue
import sounddevice
import pyttsx3
import webbrowser
import vosk
import numpy as np
from scipy.spatial.distance import cosine
from langchain_ollama import OllamaLLM
from langchain.agents import create_agent
from langchain_core.tools import tool
from openwakeword.model import Model as OWWModel

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

SAMPLE_RATE = 16000
AUDIO_BLOCKSIZE = 8000
COMMAND_BLOCKSIZE = 1280
TTS_RATE = 150

# Enrollment configuration
ENROLLMENT_PHRASES = [
    "Hello, I am Sruthindev",
    "Shuttu, can you help me?",
    "Good morning, Shuttu",
    "Open the browser for me",
    "This is my voice signature"
]

VOICE_FILE = "boss_voice.npy"
BOSS_DISTANCE_THRESHOLD = 0.25

# Wake-word detection thresholds
OWW_CONFIDENCE_THRESHOLD = 0.5
OWW_DEBUG_THRESHOLD = 0.3
OWW_BUFFER_SIZE = 1280

# Agent configuration
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

# ============================================================================
# GLOBAL STATE
# ============================================================================

audio_queue = queue.Queue()
is_speaking = False
chat_history = []

# Initialize TTS engine
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', TTS_RATE)
except Exception as e:
    print(f"⚠ pyttsx3 initialisation failed: {e}")
    engine = None


# ============================================================================
# AUDIO FUNCTIONS
# ============================================================================

def audio_callback(indata, frames, time, status):
    """Callback for audio stream processing"""
    global is_speaking
    
    if status:
        print(f"⚠ Audio stream error: {status}")
    
    if not is_speaking:
        audio_queue.put(bytes(indata))


def speak(text):
    """Text-to-speech output"""
    global is_speaking, engine
    
    if engine is None:
        print(f"[TTS] {text}")
        return
    
    try:
        is_speaking = True
        engine.say(text)
        engine.runAndWait()
    finally:
        is_speaking = False


def clear_audio_queue():
    """Clear all audio from queue"""
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break


def is_queue_empty():
    """Check if audio queue is empty"""
    return audio_queue.empty()


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

def init_vosk_models():
    """Initialize Vosk speech recognition and speaker identification models"""
    try:
        model = vosk.Model("model")
        recogniser = vosk.KaldiRecognizer(model, SAMPLE_RATE, '["shuttu", "[unk]"]')
        spk_model = vosk.SpkModel("model-spk")
        
        print("✓ Vosk models loaded successfully")
        return model, spk_model, recogniser
        
    except FileNotFoundError as e:
        print(f"❌ Vosk model not found: {e}")
        print("   Please download models from https://alphacephei.com/vosk/models")
        raise


def init_llm():
    """Initialize Ollama LLM"""
    try:
        llm = OllamaLLM(
            model="phi3",
            temperature=0.8,
            num_predict=50,
            stop=["User:", "\n"]
        )
        print("✓ Ollama LLM (phi3) initialized successfully")
        return llm
        
    except Exception as e:
        print(f"❌ LLM initialisation failed: {e}")
        print("   Make sure Ollama is running: ollama serve")
        sys.exit(1)


def init_openwakeword():
    """Initialize OpenWakeword model for acoustic wake-word detection"""
    try:
        if os.path.exists("shuttu.onnx"):
            oww_model = OWWModel(wakeword_models=["shuttu.onnx"])
            print("✓ Custom OpenWakeword model (shuttu.onnx) loaded successfully")
        else:
            oww_model = OWWModel(wakeword_models=["alexa"])
            print("✓ OpenWakeword model (alexa) loaded as fallback")
            print("  Note: Using 'alexa' model. For 'shuttu' specificity, create shuttu.onnx")
            
        return oww_model
        
    except Exception as e:
        print(f"⚠ Warning: OpenWakeword failed to load: {e}")
        print("  Falling back to Vosk-only wake-word detection")
        return None


def init_model():
    """Master initialization function - loads all models"""
    print("\n" + "="*60)
    print("🚀 INITIALIZING SHUTTU MODELS")
    print("="*60)
    
    model, spk_model, recogniser = init_vosk_models()
    llm = init_llm()
    oww_model = init_openwakeword()
    
    print("="*60)
    print("✓ All models initialized successfully!\n")
    
    return model, spk_model, recogniser, llm, oww_model


# ============================================================================
# VOICE ENROLLMENT & SPEAKER VERIFICATION
# ============================================================================

def enroll_boss(model, spk_model):
    """Voice enrollment process - captures and stores user's voice fingerprint"""
    p = pyttsx3.init()
    p.setProperty('rate', TTS_RATE)
    
    print("\n" + "="*60)
    print("🎤 VOICE ENROLLMENT FOR SECURITY")
    print("="*60)
    print("\nI need to learn your voice to verify commands.")
    print("I will ask you to say 5 different phrases.\n")
    
    p.say("Setup your voice for shuttu")
    p.runAndWait()
    
    recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetSpkModel(spk_model)
    enrollment_data = []
    
    with sounddevice.RawInputStream(
        samplerate=SAMPLE_RATE, 
        blocksize=AUDIO_BLOCKSIZE, 
        dtype='int16',
        channels=1, 
        callback=audio_callback
    ):
        while len(enrollment_data) < len(ENROLLMENT_PHRASES):
            phrase_idx = len(enrollment_data)
            phrase = ENROLLMENT_PHRASES[phrase_idx]
            
            print(f"\n[Sample {phrase_idx + 1}/{len(ENROLLMENT_PHRASES)}] Please say: {phrase}")
            
            p.say(phrase)
            p.runAndWait()
            
            print("   ⏺️  Listening... (speak now)")
            
            data = audio_queue.get()
            
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                
                if 'spk' in result:
                    enrollment_data.append(result['spk'])
                    text = result.get('text', '(recognized)')
                    print(f"   ✓ Captured! (Heard: '{text}')")
                else:
                    print("   ✗ No voice detected, try again...")
            else:
                print("   ⏳ Processing...")
    
    boss_voice_print = np.mean(enrollment_data, axis=0)
    np.save(VOICE_FILE, boss_voice_print)
    
    print("\n" + "="*60)
    print("✅ ENROLLMENT COMPLETE!")
    print("="*60)
    p.say("Enrollment successful. I have learned your voice.")
    p.runAndWait()
    print("Your voice signature has been saved.\n")


def is_it_boss(current_spk_vector):
    """Verify if speaker is the enrolled boss"""
    try:
        master_vector = np.load(VOICE_FILE)
        distance = cosine(master_vector, current_spk_vector)
        
        if distance < BOSS_DISTANCE_THRESHOLD:
            return True
        return False
        
    except FileNotFoundError:
        print("⚠ No voice enrollment found - allowing access")
        return True


def ensure_boss_enrolled(model, spk_model):
    """Check if boss is enrolled, if not perform enrollment"""
    if not os.path.exists(VOICE_FILE):
        print("\n🔐 First time setup: Enrolling your voice for security...")
        enroll_boss(model, spk_model)


def load_boss_voice():
    """Load the boss voice fingerprint"""
    try:
        return np.load(VOICE_FILE)
    except FileNotFoundError:
        print("ERROR: boss_voice.npy not found! Enrollment failed.")
        return None


def verify_boss(spk_vector):
    """Verify if speaker is the boss"""
    if not spk_vector:
        return False, "No speaker data"
    
    if is_it_boss(spk_vector):
        return True, "BOSS VERIFIED"
    
    return False, "STRANGER"


# ============================================================================
# TOOLS FOR AI AGENT
# ============================================================================

@tool
def open_browser(url: str):
    """Open a URL in the default web browser"""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    
    try:
        speak(f"opening {url}")
        webbrowser.open(url)
        speak(f"I have opened {url}")
        return f"I have opened {url} for you, Boss"
        
    except Exception as e:
        error_msg = f"Failed to open {url}: {str(e)}"
        speak(f"I couldn't open that URL. {error_msg}")
        return error_msg


@tool
def go_to_sleep():
    """Exit the program - used to stop Shuttu"""
    speak("Going to sleep. Goodbye!")
    sys.exit(0)


TOOLS = [open_browser, go_to_sleep]


# ============================================================================
# WAKE-WORD DETECTION & UTILITY FUNCTIONS
# ============================================================================

def process_vosk_result(recognizer, audio_data):
    """Process audio through Vosk recognizer"""
    if recognizer.AcceptWaveform(audio_data):
        result = json.loads(recognizer.Result())
        return result, True
    else:
        try:
            partial = json.loads(recognizer.PartialResult())
            return partial, False
        except:
            return None, False


def check_wakeword_vosk(recognizer, audio_data):
    """Check for wake-word in Vosk recognition"""
    result, is_final = process_vosk_result(recognizer, audio_data)
    
    if not result:
        return False, "", None
    
    if is_final:
        text = result.get("text", "").strip().lower()
        if text and "shuttu" in text:
            return True, text, result
    else:
        text = result.get("partial", "").strip().lower()
        if text and "shuttu" in text:
            return True, text, result
    
    return False, "", result if is_final else None


def check_wakeword_oww(oww_model, audio_input):
    """Check for wake-word using OpenWakeword model"""
    try:
        prediction = oww_model.predict(audio_input)
        
        if prediction.get("shuttu", 0) > OWW_CONFIDENCE_THRESHOLD:
            return True, prediction["shuttu"], "shuttu"
        
        if prediction.get("alexa", 0) > OWW_CONFIDENCE_THRESHOLD:
            return True, prediction["alexa"], "alexa"
        
        return False, 0, None
        
    except Exception as e:
        print(f"[ERROR] OpenWakeword prediction failed: {e}")
        return False, 0, None


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def start_shuttu():
    """Main entry point for Shuttu voice assistant"""
    
    print("\n" + "="*60)
    print("🚀 STARTING SHUTTU")
    print("="*60)
    
    # Initialize all models
    model, spk_model, recogniser, llm, oww_model = init_model()
    
    # Ensure boss is enrolled
    ensure_boss_enrolled(model, spk_model)
    
    # Create agent with tools
    agent = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT
    )
    
    # Show startup message
    print(MESSAGE_SLEEPING)
    print(f"OpenWakeword available: {oww_model is not None}")
    if not oww_model:
        print(MESSAGE_WITHOUT_OWW)
    
    print(MESSAGE_LISTENING)
    
    # Pre-load the master voice for verification
    master_vector = load_boss_voice()
    if not master_vector:
        return
    
    try:
        # Main listening loop
        audio_buffer = []
        
        with sounddevice.RawInputStream(
            samplerate=SAMPLE_RATE, 
            blocksize=COMMAND_BLOCKSIZE,
            dtype='int16',
            channels=1, 
            callback=audio_callback
        ):
            while True:
                # Get audio from queue
                data = audio_queue.get()
                
                # Convert to numpy array
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                audio_buffer.extend(audio_chunk)
                
                # ========== WAKE-WORD DETECTION ==========
                wake_word_detected = False
                
                # Try OpenWakeword first (more accurate)
                if oww_model and len(audio_buffer) >= OWW_BUFFER_SIZE:
                    audio_input = np.array(audio_buffer[-OWW_BUFFER_SIZE:], dtype=np.int16)
                    detected, confidence, model_name = check_wakeword_oww(oww_model, audio_input)
                    
                    if confidence > OWW_DEBUG_THRESHOLD:
                        print(f"[DEBUG] {model_name}: {confidence:.2%}")
                    
                    if detected:
                        print(f"\n[✓ WAKE WORD] {model_name} - Confidence: {confidence:.2%}")
                        wake_word_detected = True
                else:
                    # Fallback to Vosk wake-word detection
                    detected, text, result = check_wakeword_vosk(recogniser, data)
                    
                    if text:
                        print(f"[Vosk] Heard: '{text}'")
                    
                    if detected:
                        print(f"\n✓ [WAKE WORD DETECTED] {text}")
                        wake_word_detected = True
                
                # ========== COMMAND LISTENING (if wake-word detected) ==========
                if wake_word_detected:
                    speak("I'm listening")
                    
                    # Create new recognizer for command
                    cmd_recogniser = vosk.KaldiRecognizer(model, SAMPLE_RATE, '["shuttu", "[unk]"]')
                    cmd_recogniser.SetSpkModel(spk_model)
                    
                    print(MESSAGE_LISTENING_CMD)
                    
                    # Listen for command with timeout
                    silence_count = 0
                    
                    while True:
                        if is_queue_empty():
                            silence_count += 1
                            if silence_count > 80:  # ~5 seconds of silence
                                break
                            continue
                        
                        silence_count = 0
                        cmd_data = audio_queue.get()
                        
                        # Process with Vosk recognizer
                        if cmd_recogniser.AcceptWaveform(cmd_data):
                            result = json.loads(cmd_recogniser.Result())
                            user_text = result.get("text", "").strip().lower()
                            
                            if user_text:
                                print(f"[COMMAND] {user_text}")
                                
                                # ========== SPEAKER VERIFICATION ==========
                                spk_vector = result.get("spk")
                                is_boss, verification_msg = verify_boss(spk_vector)
                                
                                print(f"[{verification_msg}]")
                                
                                if is_boss:
                                    # ========== PROCESS WITH AGENT ==========
                                    try:
                                        # Build message history
                                        messages = chat_history + [
                                            {"role": "user", "content": user_text}
                                        ]
                                        
                                        # Get response from agent
                                        response = agent.invoke({"messages": messages})
                                        
                                        ans = response["messages"][-1]["content"]
                                        
                                        # Add to chat history
                                        chat_history.append({"role": "user", "content": user_text})
                                        chat_history.append({"role": "assistant", "content": ans})
                                        
                                        print(f"Shuttu: {ans}")
                                        speak(ans)
                                        
                                    except Exception as e:
                                        print(f"[ERROR] Agent processing failed: {e}")
                                        speak("Sorry, I encountered an error processing that.")
                                
                                else:
                                    # Unknown speaker - ignore
                                    speak(MESSAGE_STRANGER)
                                
                                # Clear audio queue and buffers
                                clear_audio_queue()
                                audio_buffer = []
                                break
                    
                    # Reset for next wake-word detection
                    audio_buffer = []
                    
                    # Reset recognizer if not using OWW
                    if not oww_model:
                        recogniser = vosk.KaldiRecognizer(
                            model, SAMPLE_RATE, '["shuttu", "[unk]"]'
                        )
    
    except KeyboardInterrupt:
        print("\n\n🛌 Shutting down gracefully...")
        speak("Goodbye!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    try:
        start_shuttu()
    except Exception as e:
        print(f"Fatal error: {e}")
        exit(1)

   



    
    
        
        
      

