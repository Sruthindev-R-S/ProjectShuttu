"""
Audio processing and callbacks for Shuttu AI Assistant
Handles audio input stream and queue management
"""

import queue
import pyttsx3

# Global audio queue and state
audio_queue = queue.Queue()
is_speaking = False  # Prevents audio feedback loop

# Initialize TTS engine
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
except Exception as e:
    print(f"⚠ pyttsx3 initialisation failed: {e}")
    engine = None


def audio_callback(indata, frames, time, status):
    """
    Callback for audio stream processing
    
    Args:
        indata: Audio data from the stream
        frames: Number of frames
        time: Time info
        status: Stream status
    """
    global is_speaking
    
    if status:
        print(f"⚠ Audio stream error: {status}")
    
    # Don't listen when Shuttu is speaking (prevents feedback loop)
    if not is_speaking:
        audio_queue.put(bytes(indata))


def speak(text):
    """
    Text-to-speech output
    
    Args:
        text (str): Text to speak
    """
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


def get_audio_from_queue(timeout=None):
    """
    Get audio data from queue (non-blocking)
    
    Args:
        timeout (float): Timeout in seconds
        
    Returns:
        bytes: Audio data or None if queue empty
    """
    try:
        return audio_queue.get(timeout=timeout)
    except queue.Empty:
        return None


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
