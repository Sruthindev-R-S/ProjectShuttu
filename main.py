import os
import sys
import json
import queue
import threading
import sounddevice
import pyttsx3
import webbrowser
import vosk
from vosk import Model, KaldiRecognizer
from langchain_ollama import OllamaLLM
from langchain.agents import create_agent
from langchain_core.tools import tool
import numpy as np
from scipy.spatial.distance import cosine
from openwakeword.model import Model as OWWModel

audio_queue = queue.Queue()
is_speaking = False  # Global flag to prevent audio feedback loop
chat_history = []  # Manual chat history for LangChain 1.x

try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
except Exception as e:
    print(f"pyttxs3 initialisation failed by {e}")
    

def init_model():
    try:
        model = vosk.Model("model")
        recogniser = vosk.KaldiRecognizer(model, 16000, '["shuttu", "[unk]"]')
        spk_model=vosk.SpkModel("model-spk")
    except FileNotFoundError:
        print("Vosk model not found")
    
    try:
        llm = OllamaLLM(model="phi3",
                        temperature=0.8,
                        num_predict=50,
                        stop=["User:", "\n"])
    except Exception as e:
        print(f"Model initialisation failed by {e}")
        sys.exit()
    
    # Initialize OpenWakeword model for acoustic wake-word detection
    oww_model = None
    try:
        # Try to load custom shuttu.onnx file first
        if os.path.exists("shuttu.onnx"):
            oww_model = OWWModel(wakeword_models=["shuttu.onnx"])
            print("✓ Custom OpenWakeword model (shuttu.onnx) loaded successfully")
        else:
            # Fallback to universal model (detects common wake words)
            oww_model = OWWModel(wakeword_models=["alexa"])
            print("✓ OpenWakeword model (alexa) loaded successfully")
            print("  Note: Using 'alexa' model. For 'shuttu' specificity, create shuttu.onnx")
    except Exception as e:
        print(f"⚠ Warning: OpenWakeword failed to load: {e}")
        print("  Falling back to Vosk-only wake-word detection")
        oww_model = None
   
    return model, spk_model, recogniser, llm, oww_model
def audio_callback(indata, frames, time, status):
    global is_speaking
    if status:
        print(f"Error for Audio call back: {status}")
    # Don't listen when Shuttu is speaking (prevents feedback loop)
    if not is_speaking:
        audio_queue.put(bytes(indata))

    
    
def enroll_boss(model,spk_model):
    p=pyttsx3.init()
    p.setProperty('rate',150)
    
    print("\n" + "="*60)
    print("🎤 VOICE ENROLLMENT FOR SECURITY")
    print("="*60)
    print("\nI need to learn your voice to verify commands.")
    print("I will ask you to say 5 different phrases.\n")
    
    p.say("Setup your voice for shuttu")
    p.runAndWait()
    
    # Sample phrases for enrollment
    enrollment_phrases = [
        "Please say: Hello, I am Sruthindev",
        "Please say: Shuttu, can you help me?",
        "Please say: Good morning, Shuttu",
        "Please say: Open the browser for me",
        "Please say: This is my voice signature"
    ]
    
    recognizer = vosk.KaldiRecognizer(model, 16000)
    recognizer.SetSpkModel(spk_model)
    enrollment_data=[]
    
    with sounddevice.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', 
                                    channels=1, callback=audio_callback):
        while len(enrollment_data) < 5:
            # Show the phrase to say
            phrase = enrollment_phrases[len(enrollment_data)]
            print(f"\n[Sample {len(enrollment_data)+1}/5] {phrase}")
            
            # Also speak it
            p.say(phrase.replace("Please say: ", ""))
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
    
    boss_voice_print=np.mean(enrollment_data,axis=0)
    np.save("boss_voice.npy",boss_voice_print)
    
    print("\n" + "="*60)
    print("✅ ENROLLMENT COMPLETE!")
    print("="*60)
    p.say("Enrollment successful. I have learned your voice.")
    p.runAndWait()
    print("Your voice signature has been saved.\n")
def is_it_boss(current_spk_vector):
    try:
        master_vector=np.load("boss_voice.npy")
        distance=cosine(master_vector,current_spk_vector)
        if distance<0.25:
            return True
        return False
    except FileNotFoundError:
        print("No audio file for boss")
        return True
@tool
def open_browser(url: str):
    """Open a URL in the browser"""
    global is_speaking
    
    # Add http/https if missing
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    
    try:
        is_speaking = True
        engine.say(f"opening {url}")
        engine.runAndWait()
        webbrowser.open(url)
        engine.say(f"I have opened {url}")
        engine.runAndWait()
        is_speaking = False
        return f"I have opened {url} for you, Boss"
    except Exception as e:
        is_speaking = False
        return f"Failed to open {url}: {str(e)}"

@tool
def go_to_sleep():
    """Exit the program - used to stop Shuttu"""
    global is_speaking
    is_speaking = True
    engine.say("Going to sleep. Goodbye!")
    engine.runAndWait()
    sys.exit(0)

tools = [open_browser, go_to_sleep]
model, spk_model, recogniser, llm, oww_model = init_model()
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""
You are Shuttu, a highly intelligent, sassy, and slightly naughty AI assistant.
You were created by Sruthindev.
"""
)

def start_shuttu():
    """Main entry point for Shuttu voice assistant"""
    global is_speaking, recogniser
    
    # Ensure boss is enrolled
    if not os.path.exists("boss_voice.npy"):
        print("\n🔐 First time setup: Enrolling your voice for security...")
        enroll_boss(model, spk_model)

    print("\n=== SHUTTU is sleeping ===")
    print("Say 'Shuttu' to wake her up...")
    print("Say 'Shuttu, go to sleep' to exit\n")
    
    # Pre-load the master voice for speed
    try:
        master_vector = np.load("boss_voice.npy")
    except FileNotFoundError:
        print("ERROR: boss_voice.npy not found! Enrollment failed.")
        return

    try:
        # Buffer for OpenWakeword processing (needs 1280 samples at 16kHz = 80ms)
        audio_buffer = []
        buffer_size = 1280
        
        print(f"OpenWakeword available: {oww_model is not None}")
        if not oww_model:
            print("⚠️  Using Vosk-only wake-word detection mode")
            print("🎤 Listening for 'shuttu'...\n")
        
        with sounddevice.RawInputStream(samplerate=16000, blocksize=1280, dtype='int16', 
                                        channels=1, callback=audio_callback):
            while True:
                # 1. Get audio from the queue
                data = audio_queue.get()
                
                # Convert to numpy array for processing
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                audio_buffer.extend(audio_chunk)
                
                # 2. Check for wake-word using OpenWakeword (acoustic matching)
                if oww_model and len(audio_buffer) >= buffer_size:
                    # Get the latest buffer_size samples
                    audio_input = np.array(audio_buffer[-buffer_size:], dtype=np.int16)
                    
                    try:
                        # Run OpenWakeword prediction
                        prediction = oww_model.predict(audio_input)
                        
                        # Debug: Print top predictions
                        top_pred = max(prediction.items(), key=lambda x: x[1])
                        if top_pred[1] > 0.3:  # Debug threshold
                            print(f"[DEBUG] Model detected: {top_pred[0]} ({top_pred[1]:.2%})")
                        
                        # If the acoustic properties match 'shuttu' > 50%
                        if prediction.get("shuttu", 0) > 0.5:
                            print(f"\n[WAKE WORD DETECTED] Confidence: {prediction['shuttu']:.2%}")
                            wake_word_triggered = True
                        elif prediction.get("alexa", 0) > 0.5 and not os.path.exists("shuttu.onnx"):
                            # If using alexa model fallback and it detects alexa, treat as wake word
                            print(f"\n[WAKE WORD DETECTED] Using alexa model - Confidence: {prediction['alexa']:.2%}")
                            wake_word_triggered = True
                        else:
                            wake_word_triggered = False
                    except Exception as e:
                        print(f"[DEBUG] OpenWakeword prediction error: {e}")
                        wake_word_triggered = False
                else:
                    # Fallback to Vosk-only wake word detection
                    wake_word_triggered = False
                    
                    if recogniser.AcceptWaveform(data):
                        result = json.loads(recogniser.Result())
                        user_text = result.get("text", "").strip().lower()
                        
                        if user_text:
                            print(f"[Vosk] Heard: '{user_text}'")
                        
                        if user_text and "shuttu" in user_text:
                            print(f"\n✓ [WAKE WORD DETECTED] {user_text}")
                            wake_word_triggered = True
                    else:
                        # Also check partial results for faster wake-word detection
                        try:
                            partial = json.loads(recogniser.PartialResult())
                            partial_text = partial.get("partial", "").strip().lower()
                            
                            if partial_text and "shuttu" in partial_text:
                                print(f"\n✓ [PARTIAL WAKE WORD] {partial_text}")
                                wake_word_triggered = True
                        except:
                            pass
                
                if wake_word_triggered:
                    is_speaking = True
                    
                    # Play wake-up sound and wait for command
                    engine.say("I'm listening")
                    engine.runAndWait()
                    is_speaking = False
                    
                    # Now listen for the actual command with Vosk
                    command_buffer = []
                    vosk_recogniser = vosk.KaldiRecognizer(model, 16000, '["shuttu", "[unk]"]')
                    vosk_recogniser.SetSpkModel(spk_model)
                    
                    print("[LISTENING] Waiting for command...")
                    
                    # Listen for up to 5 seconds for a command
                    timeout_buffer = []
                    
                    while True:
                        if audio_queue.empty():
                            # Wait a bit for more audio
                            if len(timeout_buffer) > 80:  # ~5 seconds of silence
                                break
                            timeout_buffer.append(None)
                            continue
                        
                        timeout_buffer = []  # Reset timeout on new audio
                        cmd_data = audio_queue.get()
                        command_buffer.append(cmd_data)
                        
                        # Process with Vosk
                        if vosk_recogniser.AcceptWaveform(cmd_data):
                            result = json.loads(vosk_recogniser.Result())
                            user_text = result.get("text", "").strip().lower()
                            
                            if user_text:
                                print(f"[COMMAND] {user_text}")
                                
                                # 3. Security Check (Is it the Boss?)
                                current_spk = result.get("spk")
                                
                                if current_spk and is_it_boss(current_spk):
                                    print(f"[BOSS VERIFIED] Processing: {user_text}")
                                    
                                    # 4. Execute the Sassy Agent
                                    try:
                                        is_speaking = True
                                        # Build message list with chat history
                                        messages = chat_history + [{"role": "user", "content": user_text}]
                                        response = agent.invoke({
                                            "messages": messages
                                        })
                                        
                                        ans = response["messages"][-1]["content"]
                                        
                                        # Add to chat history
                                        chat_history.append({"role": "user", "content": user_text})
                                        chat_history.append({"role": "assistant", "content": ans})
                                        
                                        print(f"Shuttu: {ans}")
                                        engine.say(ans)
                                        engine.runAndWait()
                                    except Exception as e:
                                        print(f"Agent error: {e}")
                                    finally:
                                        is_speaking = False
                                else:
                                    is_speaking = True
                                    print("[STRANGER] Ignoring unknown speaker")
                                    engine.say("I only talk to Sruthindev. Who are you?")
                                    engine.runAndWait()
                                    is_speaking = False
                                
                                # Clear the queue to prevent echo
                                while not audio_queue.empty():
                                    try:
                                        audio_queue.get_nowait()
                                    except:
                                        break
                                audio_buffer = []
                                break
                    
                    # Clear buffers and return to listening mode
                    audio_buffer = []
                    
                    # Reset the global recognizer for next wake-word detection
                    if not oww_model:
                        recogniser = vosk.KaldiRecognizer(model, 16000, '["shuttu", "[unk]"]')
                        
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)


# Entry point
if __name__ == "__main__":
    try:
        start_shuttu()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

   



    
    
        
        
      

