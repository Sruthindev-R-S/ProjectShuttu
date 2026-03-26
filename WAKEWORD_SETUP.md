# Shuttu Wake-Word Setup Guide

## Current Status

✅ **Shuttu is now working!** It uses a fallback `alexa` OpenWakeword model for acoustic detection.

---

## How It Works Now (3-Stage Detection)

```
1. OpenWakeword (Acoustic) → Detects "Shuttu" sound acoustically
2. Vosk (Speech Recognition) → Listens for voice command
3. Speaker Verification → Confirms it's you (boss_voice.npy)
```

### Try It Now

```bash
python main.py
```

- Say "**shuttu**" or similar wake word to trigger
- System will say "I'm listening"
- Give a command like "open google.com"

---

## Option: Create Your Own 'Shuttu' Model

For better accuracy with "Shuttu" specifically, train a custom model:

### 1. Collect Audio Samples

```python
# You already have this! (boss_voice.npy or saved samples)
# Need: 10-20 audio clips of you saying "Shuttu"
```

### 2. Use OpenWakeword to Train

```python
from openwakeword.model import Model
import numpy as np

# Assuming you have audio files of "shuttu"
model_constructor = Model.build_custom_model(
    model_name="shuttu",
    training_config="path/to/your/samples",
)

model_constructor.model.save_onnx("shuttu.onnx")
```

### 3. Place in Project Root

```
projectShuttu/
├── main.py
├── shuttu.onnx          ← Place model here
├── boss_voice.npy
└── ...
```

The system will auto-detect and use `shuttu.onnx` instead of the fallback.

---

## Current Fallback Models Available

If you want to use a different pre-trained wake word:

Edit `main.py` line ~48:

```python
oww_model = OWWModel(wakeword_models=["alexa"])  # Change to your choice
```

Common options:

- `"alexa"` - Default
- `"hey_google"` - Google Assistant wake word
- `"ok_google"` - Alternative Google
- `"hey_siri"` - Apple Siri

---

## Debug Output

When Shuttu starts, you'll see:

```
OpenWakeword available: True
✓ OpenWakeword model (alexa) loaded successfully
  Note: Using 'alexa' model. For 'shuttu' specificity, create shuttu.onnx
```

If something goes wrong:

- Check console for `[DEBUG]` messages
- Verify microphone is working: `python -m sounddevice`
- Check `boss_voice.npy` exists (run enrollment again if needed)

---

## Troubleshooting

| Issue                         | Solution                                       |
| ----------------------------- | ---------------------------------------------- |
| "Not waking up"               | Try saying the exact model word louder/clearer |
| "OpenWakeword failed to load" | Check scipy, numpy, onnxruntime installed      |
| "Stranger message"            | Run enrollment again: `python main.py`         |
| No audio recognition          | Test: `python -m sounddevice`                  |

---

## Architecture Overview

```
Input Audio (16kHz, 1280 samples)
    ↓
[OpenWakeword Model] → Acoustic match? (>50% confidence)
    ↓ YES
[Vosk + Grammar] → Listen for "shuttu" command
    ↓
[Speaker Verification] → Is it the boss?
    ↓ YES
[LangChain + Ollama] → Generate sassy response
    ↓
[Text-to-Speech] → Say response
```
