"""
Model initialization and management for Shuttu AI Assistant
Handles Vosk, Ollama LLM, and OpenWakeword model loading
"""

import os
import sys
import vosk
from vosk import Model, KaldiRecognizer, SpkModel
from langchain_ollama import OllamaLLM
from openwakeword.model import Model as OWWModel


def init_vosk_models():
    """
    Initialize Vosk speech recognition and speaker identification models
    
    Returns:
        tuple: (model, spk_model, recogniser)
            - model: Vosk speech recognition model
            - spk_model: Speaker identification model
            - recogniser: Vosk KaldiRecognizer instance
    
    Raises:
        FileNotFoundError: If model files are not found
    """
    try:
        model = vosk.Model("model")
        recogniser = vosk.KaldiRecognizer(model, 16000, '["shuttu", "[unk]"]')
        spk_model = vosk.SpkModel("model-spk")
        
        print("✓ Vosk models loaded successfully")
        return model, spk_model, recogniser
        
    except FileNotFoundError as e:
        print(f"❌ Vosk model not found: {e}")
        print("   Please download models from https://alphacephei.com/vosk/models")
        raise


def init_llm():
    """
    Initialize Ollama LLM (Large Language Model)
    
    Returns:
        OllamaLLM: Configured Ollama instance for phi3 model
    
    Raises:
        Exception: If Ollama is not running or model initialization fails
    """
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
    """
    Initialize OpenWakeword model for acoustic wake-word detection
    
    Returns:
        OWWModel or None: OpenWakeword model if available, None if failed
    """
    oww_model = None
    try:
        # Try to load custom shuttu.onnx file first
        if os.path.exists("shuttu.onnx"):
            oww_model = OWWModel(wakeword_models=["shuttu.onnx"])
            print("✓ Custom OpenWakeword model (shuttu.onnx) loaded successfully")
        else:
            # Fallback to universal model (detects common wake words)
            oww_model = OWWModel(wakeword_models=["alexa"])
            print("✓ OpenWakeword model (alexa) loaded as fallback")
            print("  Note: Using 'alexa' model. For 'shuttu' specificity, create shuttu.onnx")
            
        return oww_model
        
    except Exception as e:
        print(f"⚠ Warning: OpenWakeword failed to load: {e}")
        print("  Falling back to Vosk-only wake-word detection")
        return None


def init_model():
    """
    Master initialization function - loads all models
    
    Returns:
        tuple: (model, spk_model, recogniser, llm, oww_model)
            - model: Vosk speech model
            - spk_model: Vosk speaker model
            - recogniser: Vosk KaldiRecognizer
            - llm: Ollama LLM instance
            - oww_model: OpenWakeword model (or None)
    """
    print("\n" + "="*60)
    print("🚀 INITIALIZING SHUTTU MODELS")
    print("="*60)
    
    # Initialize Vosk models
    model, spk_model, recogniser = init_vosk_models()
    
    # Initialize LLM
    llm = init_llm()
    
    # Initialize OpenWakeword (optional)
    oww_model = init_openwakeword()
    
    print("="*60)
    print("✓ All models initialized successfully!\n")
    
    return model, spk_model, recogniser, llm, oww_model
