"""
Tools for Shuttu AI Assistant agent
Defines available actions the AI can perform
"""

import sys
import webbrowser
from langchain_core.tools import tool
from audio import speak


@tool
def open_browser(url: str):
    """
    Open a URL in the default web browser
    
    Args:
        url (str): URL to open (with or without http/https)
        
    Returns:
        str: Status message
    """
    # Add http/https if missing
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
    """
    Exit the program - used to stop Shuttu
    
    Returns:
        str: Goodbye message
    """
    speak("Going to sleep. Goodbye!")
    sys.exit(0)


# List of available tools
TOOLS = [open_browser, go_to_sleep]


def get_tools():
    """
    Get all available tools for the agent
    
    Returns:
        list: List of tool functions
    """
    return TOOLS
