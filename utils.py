"""
Utility functions for Shuttu AI Assistant
"""

import json
import numpy as np
from enrollment import is_it_boss, VOICE_FILE


def load_boss_voice():
    """
    Load the boss voice fingerprint
    
    Returns:
        np.ndarray: Voice fingerprint or None if not found
    """
    try:
        return np.load(VOICE_FILE)
    except FileNotFoundError:
        print("ERROR: boss_voice.npy not found! Enrollment failed.")
        return None


def process_vosk_result(recognizer, audio_data):
    """
    Process audio through Vosk recognizer
    
    Args:
        recognizer: Vosk KaldiRecognizer instance
        audio_data: Audio data bytes
        
    Returns:
        tuple: (result_dict, is_final) or (None, is_final) if no result
    """
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
    """
    Check for wake-word in Vosk recognition
    
    Args:
        recognizer: Vosk recognizer instance
        audio_data: Audio data bytes
        
    Returns:
        tuple: (detected, text, result_dict)
    """
    result, is_final = process_vosk_result(recognizer, audio_data)
    
    if not result:
        return False, "", None
    
    # Check final results
    if is_final:
        text = result.get("text", "").strip().lower()
        if text and "shuttu" in text:
            return True, text, result
    else:
        # Check partial results for faster detection
        text = result.get("partial", "").strip().lower()
        if text and "shuttu" in text:
            return True, text, result
    
    return False, "", result if is_final else None


def check_wakeword_oww(oww_model, audio_input):
    """
    Check for wake-word using OpenWakeword model
    
    Args:
        oww_model: OpenWakeword model instance
        audio_input: Audio input as numpy array
        
    Returns:
        tuple: (detected, confidence, model_name)
    """
    try:
        prediction = oww_model.predict(audio_input)
        
        # Check for 'shuttu' keyword (primary)
        if prediction.get("shuttu", 0) > 0.5:
            return True, prediction["shuttu"], "shuttu"
        
        # Fallback to 'alexa' if no shuttu.onnx
        if prediction.get("alexa", 0) > 0.5:
            return True, prediction["alexa"], "alexa"
        
        return False, 0, None
        
    except Exception as e:
        print(f"[ERROR] OpenWakeword prediction failed: {e}")
        return False, 0, None


def verify_boss(spk_vector):
    """
    Verify if speaker is the boss
    
    Args:
        spk_vector: Speaker identification vector from Vosk
        
    Returns:
        tuple: (is_boss, confidence_message)
    """
    if not spk_vector:
        return False, "No speaker data"
    
    if is_it_boss(spk_vector):
        return True, "BOSS VERIFIED"
    
    return False, "STRANGER"
