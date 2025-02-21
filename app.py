import streamlit as st
import subprocess
import sys
import os
from pathlib import Path
import google.generativeai as genai
from time import sleep
import signal

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyDjlibHigHJ23MNR3A9QHsgpxnmVsSqeeQ"

# Global variable to store the gesture process
gesture_process = None

def configure_genai():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel('gemini-pro')

def get_gesture_info():
    return {
        "cursor_control": {
            "name": "Cursor Control",
            "description": "Move your index finger in the air to control the cursor",
            "gesture": "Open palm, move hand in desired direction",
            "image": "👆"
        },
        "left_click": {
            "name": "Left Click",
            "description": "Perform a left mouse click",
            "gesture": "Close index finger and thumb while keeping other fingers open",
            "image": "👆"
        },
        
        "scroll": {
            "name": "Scroll",
            "description": "Scroll up or down",
            "gesture": "Make a openhand and move up/down",
            "image": "🖐️"
        },
        "volume_control": {
            "name": "Volume Control",
            "description": "Adjust system volume",
            "gesture": "Thumb up/down gesture",
            "image": "👍"
        },
        "task_view": {
            "name": "Task View",
            "description": "Open Windows Task View",
            "gesture": "Three fingers up (index, middle, ring)",
            "image": "🖐️"
        }
    }

def start_gesture_control():
    global gesture_process
    if gesture_process is None:
        try:
            # Get the absolute path to g9-1.py
            gesture_path = os.path.abspath('g9-1.py')
            if not os.path.exists(gesture_path):
                st.error("g9-1.py not found!")
                return False
                
            # Start g9-1.py as a subprocess without new console
            gesture_process = subprocess.Popen(
                [sys.executable, gesture_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            # Check if process started successfully
            if gesture_process.poll() is None:
                st.success("Gesture control started successfully!")
                return True
            else:
                error = gesture_process.stderr.read().decode()
                st.error(f"Failed to start gesture control: {error}")
                gesture_process = None
                return False
                
        except Exception as e:
            st.error(f"Error starting gesture control: {str(e)}")
            gesture_process = None
            return False
    return False

def stop_gesture_control():
    global gesture_process
    if gesture_process is not None:
        try:
            # Try graceful termination first
            gesture_process.terminate()
            
            # Wait up to 2 seconds for process to terminate
            try:
                gesture_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # If process doesn't terminate, force kill it
                gesture_process.kill()
                gesture_process.wait()
            
            # Release any remaining resources
            if gesture_process.stdout:
                gesture_process.stdout.close()
            if gesture_process.stderr:
                gesture_process.stderr.close()
            
            gesture_process = None
            st.success("Gesture control stopped successfully")
            return True
            
        except Exception as e:
            st.error(f"Error stopping gesture control: {str(e)}")
            # Even if there's an error, try to clean up
            gesture_process = None
            return False
    return False

def main():
    st.set_page_config(
        page_title="HCI Gesture Control",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add session state to track gesture control status
    if 'gesture_control_active' not in st.session_state:
        st.session_state.gesture_control_active = False
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            background-color: #1E1E1E;
            color: white;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 15px 32px;
            border: none;
            border-radius: 10px;
            font-size: 20px;
            font-weight: bold;
            width: 100%;
            margin: 10px 0;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        }
        .stButton>button:hover {
            background-color: #45a049;
            transform: scale(1.02);
            transition: all 0.2s ease;
        }
        .stop-button>button {
            background-color: #dc3545 !important;
        }
        .stop-button>button:hover {
            background-color: #c82333 !important;
        }
        .gesture-card {
            background-color: #2D2D2D;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.4);
            margin: 15px 0;
            border: 2px solid #4CAF50;
        }
        .gesture-emoji {
            font-size: 48px;
            margin-bottom: 15px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .gesture-title {
            color: #4CAF50;
            font-size: 28px;
            font-weight: bold;
            margin: 12px 0;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        }
        .gesture-description {
            color: #FFFFFF;
            font-size: 18px;
            line-height: 1.6;
            margin: 10px 0;
            text-shadow: 1px 1px 1px rgba(0,0,0,0.2);
        }
        .gesture-instruction {
            color: #B9F6CA;
            font-size: 18px;
            font-weight: 500;
            margin: 10px 0;
            text-shadow: 1px 1px 1px rgba(0,0,0,0.2);
        }
        .stTextInput input {
            font-size: 18px;
            padding: 12px;
            border-radius: 8px;
            border: 2px solid #4CAF50;
            background-color: #2D2D2D;
            color: white;
        }
        .stTextInput input:focus {
            border-color: #69F0AE;
            box-shadow: 0 0 8px rgba(105, 240, 174, 0.5);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Main title with custom styling
    st.markdown("<h1 style='text-align: center; color: #2E7D32;'>🖐️ HCI Gesture Control</h1>", unsafe_allow_html=True)
    
    # Add control buttons in the main area
    col1, col2 = st.columns(2)
    
    with col1:
        if not st.session_state.gesture_control_active:
            if st.button("▶️ Start Gesture Control", key="start_btn"):
                if start_gesture_control():
                    st.session_state.gesture_control_active = True
                    st.success("Gesture control started successfully!")
                    st.rerun()
    
    with col2:
        if st.session_state.gesture_control_active:
            if st.button("⏹️ Stop Gesture Control", key="stop_btn"):
                if stop_gesture_control():
                    st.session_state.gesture_control_active = False
                    st.success("Gesture control stopped successfully!")
                    st.rerun()
    
    # Status indicator
    if st.session_state.gesture_control_active:
        st.markdown("""
            <div style='text-align: center; padding: 10px; background-color: #4CAF50; border-radius: 10px; margin: 10px 0;'>
                <h3 style='color: white; margin: 0;'>✨ Gesture Control is Active ✨</h3>
            </div>
        """, unsafe_allow_html=True)
    
    # Sidebar for Gemini API
    with st.sidebar:
        st.markdown("<h2 style='color: white;'>🤖 Project Assistant</h2>", unsafe_allow_html=True)
        model = configure_genai()
        user_question = st.text_input("Ask about gestures (e.g., 'How do I click?')")
        
        if user_question:
            try:
                # Enhanced context with detailed gesture information
                gesture_info = get_gesture_info()
                context = f"""
                I am an expert in the HCI Gesture Control project. This project uses computer vision to detect hand gestures
                and control the computer. Here are the available gestures and their functions:
                
                {str(gesture_info)}
                
                When answering questions about gestures, I'll provide specific details about how to perform the gesture
                and include the gesture emoji for visualization.
                
                Question: {user_question}
                
                Please provide a detailed, step-by-step answer about how to perform this gesture, including:
                1. The exact hand position and movement required
                2. What the gesture controls
                3. Any tips for better recognition
                """
                
                response = model.generate_content(context)
                st.markdown(f"<div class='gesture-card'>{response.text}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Create two columns for the layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<h3 style='color: white;'>Available Gestures</h3>", unsafe_allow_html=True)
        
        # Display gesture information in cards
        gesture_info = get_gesture_info()
        for gesture in gesture_info.values():
            with st.container():
                st.markdown(f"""
                    <div class='gesture-card'>
                        <div class='gesture-emoji'>{gesture['image']}</div>
                        <div class='gesture-title'>{gesture['name']}</div>
                        <div class='gesture-description'>{gesture['description']}</div>
                        <div class='gesture-instruction'>Gesture: {gesture['gesture']}</div>
                    </div>
                """, unsafe_allow_html=True)

    with col2:
        st.markdown("<h3 style='color: white;'>Control System</h3>", unsafe_allow_html=True)
        
        # Display status
        if st.session_state.gesture_control_active:
            st.markdown("<p style='color: #4CAF50;'>✅ System is running</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: #dc3545;'>⏹️ System is stopped</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
