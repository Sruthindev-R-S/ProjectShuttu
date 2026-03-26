"""
Shuttu - AI Voice Assistant
Main entry point
"""

import os
import json
import sounddevice
import vosk
import numpy as np
from langchain.agents import create_agent

# Import modules
from model import init_model
from audio import audio_callback, audio_queue, speak, is_queue_empty, clear_audio_queue
from enrollment import ensure_boss_enrolled
from tools import get_tools
from config import (
    SAMPLE_RATE, COMMAND_BLOCKSIZE, 
    OWW_BUFFER_SIZE, OWW_CONFIDENCE_THRESHOLD, OWW_DEBUG_THRESHOLD,
    MESSAGE_SLEEPING, MESSAGE_LISTENING, MESSAGE_WITHOUT_OWW, 
    MESSAGE_LISTENING_CMD, MESSAGE_STRANGER, SYSTEM_PROMPT
)
from utils import load_boss_voice, check_wakeword_vosk, check_wakeword_oww, verify_boss


# Global state
chat_history = []


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
        tools=get_tools(),
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

   



    
    
        
        
      

