"""
Voice enrollment and speaker identification for Shuttu AI Assistant
Handles voice fingerprinting and speaker verification
"""

import os
import json
import sounddevice
import vosk
import pyttsx3
import numpy as np
from scipy.spatial.distance import cosine
from audio import audio_callback, audio_queue


# Configuration
ENROLLMENT_PHRASES = [
    "Hello, I am Sruthindev",
    "Shuttu, can you help me?",
    "Good morning, Shuttu",
    "Open the browser for me",
    "This is my voice signature"
]

VOICE_FILE = "boss_voice.npy"
BOSS_DISTANCE_THRESHOLD = 0.25  # Cosine distance threshold for speaker match


def enroll_boss(model, spk_model):
    """
    Voice enrollment process - captures and stores user's voice fingerprint
    
    Args:
        model: Vosk speech model
        spk_model: Vosk speaker identification model
    """
    p = pyttsx3.init()
    p.setProperty('rate', 150)
    
    print("\n" + "="*60)
    print("🎤 VOICE ENROLLMENT FOR SECURITY")
    print("="*60)
    print("\nI need to learn your voice to verify commands.")
    print("I will ask you to say 5 different phrases.\n")
    
    p.say("Setup your voice for shuttu")
    p.runAndWait()
    
    # Setup recognizer
    recognizer = vosk.KaldiRecognizer(model, 16000)
    recognizer.SetSpkModel(spk_model)
    enrollment_data = []
    
    # Record enrollment samples
    with sounddevice.RawInputStream(
        samplerate=16000, 
        blocksize=8000, 
        dtype='int16',
        channels=1, 
        callback=audio_callback
    ):
        while len(enrollment_data) < len(ENROLLMENT_PHRASES):
            phrase_idx = len(enrollment_data)
            phrase = ENROLLMENT_PHRASES[phrase_idx]
            
            print(f"\n[Sample {phrase_idx + 1}/{len(ENROLLMENT_PHRASES)}] Please say: {phrase}")
            
            # Speak the phrase
            p.say(phrase)
            p.runAndWait()
            
            print("   ⏺️  Listening... (speak now)")
            
            # Get audio from queue
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
    
    # Calculate and save voice fingerprint
    boss_voice_print = np.mean(enrollment_data, axis=0)
    np.save(VOICE_FILE, boss_voice_print)
    
    print("\n" + "="*60)
    print("✅ ENROLLMENT COMPLETE!")
    print("="*60)
    p.say("Enrollment successful. I have learned your voice.")
    p.runAndWait()
    print("Your voice signature has been saved.\n")


def is_it_boss(current_spk_vector):
    """
    Verify if speaker is the enrolled boss
    
    Args:
        current_spk_vector: Speaker vector from current recognition
        
    Returns:
        bool: True if speaker is verified as boss, False otherwise
    """
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
    """
    Check if boss is enrolled, if not perform enrollment
    
    Args:
        model: Vosk speech model
        spk_model: Vosk speaker identification model
    """
    if not os.path.exists(VOICE_FILE):
        print("\n🔐 First time setup: Enrolling your voice for security...")
        enroll_boss(model, spk_model)
