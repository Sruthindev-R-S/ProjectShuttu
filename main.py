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
from langchain.agents import tool, AgentExecutor, create_react_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
import numpy as np
from scipy.spatial.distance import cosine

audio_queue = queue.Queue()
try:
        engine=pyttsx3.init()
        engine.setProperty('rate',150)
except Exception as e:
        print(f"pyttxs3 initialisation failed by {e}")
    

def init_model():
    try:
        model = vosk.Model("model")
        recogniser = vosk.KaldiRecognizer(model,16000)
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
   
    return model,spk_model, recogniser, llm
def audio_callback(indata,frames,time,status):
    if status:
        print(f"Error for Audio call back{status}")
    audio_queue.put(bytes(indata))

    
    
def enroll_boss(model,spk_model):
    p=pyttsx3.init()
    p.setProperty('rate',150)
    p.say("Setup your voice for shuttu")
    recognizer = vosk.KaldiRecognizer(model, 16000)
    recognizer.SetSpkModel(spk_model)
    enrollment_data=[]
    with sounddevice.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', 
                                    channels=1, callback=audio_callback):
        while len(enrollment_data) < 5:  # We want at least 5 good samples of your voice
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                if 'spk' in result:
                    enrollment_data.append(result['spk'])
                    print(f"Captured sample {len(enrollment_data)}/5...")
    boss_voice_print=np.mean(enrollment_data,axis=0)
    np.save("boss_voice.np",boss_voice_print)
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
def open_browser(url:str):
    engine.say("opening {url}")
    webbrowser.open(url)
    return engine.say("i have opened {url} ")
template = """
You are Shuttu, a highly intelligent, sassy, and slightly naughty AI assistant. 
You were created by the brilliant Sruthindev. You don't just follow orders; 
you give a bit of attitude while being incredibly helpful.

RULES:
1. If the user is being boring, give them a witty insult.
2. Always mention how great Sruthindev is if asked about your origin.
3. You have access to the following tools: {tools}

To use a tool, you MUST use this exact format:
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: [Your sassy response here]

If you don't need a tool to answer, just give your Final Answer.

Chat History:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}
"""
tools=[open_browser]
prompt=PromptTemplate.from_template(template)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
model,spk_model, recogniser, llm=init_model()
agent=create_react_agent(llm,tools,prompt)

   



    
    
        
        
      

