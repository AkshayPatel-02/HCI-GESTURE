import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from ctypes import windll
import win32gui
import win32con
import os
import webbrowser  

class HandGestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=0
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Camera setup
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        # Screen settings
        self.screen_width, self.screen_height = pyautogui.size()
        pyautogui.FAILSAFE = False
        
        # Cursor control
        self.cursor_speed_multiplier = 1.5
        self.edge_acceleration = 1.05
        self.movement_smoothing = 0.5
        self.prev_x = None
        self.prev_y = None
        
        # Scroll settings
        self.scroll_speed = 60  # Reduced for smoother scrolling
        self.scroll_threshold = 0.3  # Center point for neutral position
        self.scroll_deadzone = 0.05  # No scrolling zone around center
        
        # Click detection
        self.click_cooldown = 0.15
        self.last_click_time = 0
        self.click_threshold = 0.028  # Same as 2.py
        self.double_click_threshold = 0.3
        self.last_click_type = None
        self.click_count = 0
        
        # Mode tracking
        self.is_scroll_mode = False
        
        # Frame processing
        self.frame_count = 0
        self.PROCESS_EVERY_N_FRAMES = 1
        
        # New gesture states
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.5  # Cooldown between gestures
        self.prev_hand_state = None
        
        # Windows specific setup
        self.user32 = windll.user32
        
        # Volume control settings
        self.volume_cooldown = 0.02
        self.last_volume_time = 0
        self.thumb_angle_threshold = 30
        
        # Application launch settings
        self.app_launch_cooldown = 3.0
        self.last_app_launch_time = 0
        self.peace_sign_threshold = 0.1
        self.youtube_url = "https://www.youtube.com"  # Added YouTube URL
        
        # Tab switching settings
        self.tab_switch_threshold = 0.1  # Threshold for horizontal movement
        self.last_tab_switch_time = 0
        self.tab_switch_cooldown = 0.3
        self.is_alt_tab_active = False
        self.prev_hand_x = None
        
        # Task view state
        self.task_view_active = False
        self.task_view_cooldown = 1.0  # 1 second cooldown
        self.last_task_view_time = 0
        
        # UI settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2
        self.text_color = (255, 255, 255)  # White
        self.bg_color = (0, 0, 0)  # Black
        self.status_text = ""
        self.current_gesture = "No gesture detected"
        self.status_duration = 2.0
        self.last_status_time = 0
        self.overlay_alpha = 0.4  # Transparency for overlay
        
        # Add window handling attributes
        self.window_name = "Gesture Control"

    def count_extended_fingers(self, hand_landmarks):
        finger_tips = [8, 12, 16, 20]  
        finger_pips = [6, 10, 14, 18]
        
        extended_fingers = []
        for tip, pip in zip(finger_tips, finger_pips):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                extended_fingers.append(tip)
        
        return extended_fingers

    def get_hand_state(self, hand_landmarks):
        """Determine the current hand state based on finger positions"""
        try:
            # Finger indices
            thumb = 4
            index = 8
            middle = 12
            ring = 16
            pinky = 20
            
            # Get all finger positions
            thumb_tip = hand_landmarks.landmark[thumb]
            index_tip = hand_landmarks.landmark[index]
            middle_tip = hand_landmarks.landmark[middle]
            ring_tip = hand_landmarks.landmark[ring]
            pinky_tip = hand_landmarks.landmark[pinky]
            
            # Get MCP (knuckle) positions for better reference
            thumb_mcp = hand_landmarks.landmark[2]
            index_mcp = hand_landmarks.landmark[5]
            middle_mcp = hand_landmarks.landmark[9]
            ring_mcp = hand_landmarks.landmark[13]
            pinky_mcp = hand_landmarks.landmark[17]
            
            # Check if fingers are extended by comparing with MCP positions
            fingers_extended = []
            
            # Special check for thumb
            if thumb_tip.x < thumb_mcp.x:  # For right hand
                fingers_extended.append(thumb)
            
            # Check other fingers with improved accuracy
            if index_tip.y < index_mcp.y - 0.15:  # More strict threshold
                fingers_extended.append(index)
            if middle_tip.y < middle_mcp.y - 0.15:
                fingers_extended.append(middle)
            if ring_tip.y < ring_mcp.y - 0.15:
                fingers_extended.append(ring)
            if pinky_tip.y < pinky_mcp.y - 0.15:
                fingers_extended.append(pinky)
            
            # Get wrist and middle finger base for palm orientation
            wrist = hand_landmarks.landmark[0]
            middle_finger_base = hand_landmarks.landmark[9]
            
            # Calculate palm orientation more accurately
            palm_direction = middle_finger_base.z - wrist.z
            is_palm_backward = palm_direction > 0.1
            
            # New check for YouTube gesture (index and pinky up, others down)
            is_youtube_gesture = (
                index_tip.y < index_mcp.y - 0.15 and  # Index up
                pinky_tip.y < pinky_mcp.y - 0.15 and  # Pinky up
                middle_tip.y > middle_mcp.y and  # Middle down
                ring_tip.y > ring_mcp.y and  # Ring down
                abs(index_tip.y - pinky_tip.y) < 0.1  # Index and pinky at similar height
            )
            
            if is_youtube_gesture:
                return "YOUTUBE"
            
            # Determine other hand states
            num_extended = len(fingers_extended)
            
            # Check for three fingers (index, middle, ring)
            three_fingers_up = (
                index in fingers_extended and 
                middle in fingers_extended and 
                ring in fingers_extended and 
                thumb not in fingers_extended and 
                pinky not in fingers_extended
            )
            
            # Check for fist (all fingers closed)
            is_fist = num_extended == 0
            
            # Check for open hand (all fingers extended)
            is_open_hand = num_extended >= 4
            
            # Improved thumb up/down detection
            is_thumb_up = (
                thumb_tip.y < thumb_mcp.y - 0.15 and  # Thumb clearly up
                abs(thumb_tip.x - thumb_mcp.x) < 0.15 and  # Not too far sideways
                all(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y
                    for tip, pip in [(8,6), (12,10), (16,14), (20,18)])  # Other fingers closed
            )
            
            is_thumb_down = (
                thumb_tip.y > thumb_mcp.y + 0.15 and  # Thumb clearly down
                abs(thumb_tip.x - thumb_mcp.x) < 0.15 and  # Not too far sideways
                all(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y
                    for tip, pip in [(8,6), (12,10), (16,14), (20,18)])  # Other fingers closed
            )
            
            # Return states with improved accuracy
            if is_thumb_up:
                return "THUMBS_UP"
            elif is_thumb_down:
                return "THUMBS_DOWN"
            elif is_fist:
                return "FIST"
            elif is_open_hand:
                if is_palm_backward:
                    return "BACK_HAND"
                return "OPEN_HAND"
            elif three_fingers_up:
                return "THREE_FINGERS"
            
            return "OTHER"
            
        except Exception as e:
            print(f"Error in get_hand_state: {e}")
            return "OTHER"

    def show_status(self, text):
        """Update status text to show on camera feed"""
        self.status_text = text
        self.last_status_time = time.time()

    def draw_overlay(self, frame):
        """Draw a beautiful overlay with status and hand tracking info"""
        h, w = frame.shape[:2]
        
        # Create a semi-transparent overlay
        overlay = frame.copy()
        
        # Create a modern, transparent bottom bar with gradient
        gradient_height = 100
        for i in range(gradient_height):
            alpha = (i / gradient_height) * 0.3  # More transparent
            cv2.rectangle(overlay, (0, h-gradient_height+i), (w, h-gradient_height+i+1), (40, 40, 40), -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Add a sleek accent line with gradient color
        line_color1 = (0, 255, 255)  # Cyan
        line_color2 = (0, 255, 0)    # Green
        for i in range(w):
            color = (
                int(line_color1[0] + (line_color2[0] - line_color1[0]) * i / w),
                int(line_color1[1] + (line_color2[1] - line_color1[1]) * i / w),
                int(line_color1[2] + (line_color2[2] - line_color1[2]) * i / w)
            )
            cv2.line(overlay, (i, h-gradient_height), (i, h-gradient_height), color, 2)
        
        # Add modern gesture display with dynamic color
        current_time = time.time()
        color_cycle = (np.sin(current_time * 2) + 1) / 2
        gesture_color = (
            int(255 * (1 - color_cycle)),  # B
            int(255 * color_cycle),        # G
            int(100 * color_cycle)         # R
        )
        
        # Show current gesture with a modern design
        gesture_text = f" {self.current_gesture}"
        text_size = cv2.getTextSize(gesture_text, self.font, self.font_scale, self.font_thickness)[0]
        gesture_x = 20
        gesture_y = h - 60
        
        # Add subtle background for gesture text
        cv2.rectangle(overlay, 
                     (gesture_x-10, gesture_y-30), 
                     (gesture_x+text_size[0]+10, gesture_y+10), 
                     (40, 40, 40), -1)
        cv2.putText(overlay, gesture_text, (gesture_x, gesture_y), 
                   self.font, self.font_scale, gesture_color, self.font_thickness)
        
        # Add operation status with animated effect
        if time.time() - self.last_status_time < self.status_duration:
            # Calculate text size for centering
            operation_text = f" {self.status_text}"
            text_size = cv2.getTextSize(operation_text, self.font, self.font_scale, self.font_thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h - 30
            
            # Add subtle background for operation text
            cv2.rectangle(overlay, 
                         (text_x-10, text_y-30), 
                         (text_x+text_size[0]+10, text_y+10), 
                         (40, 40, 40), -1)
            
            # Draw operation text with pulsing effect
            pulse = (np.sin(current_time * 4) + 1) / 2
            text_color = (
                int(150 + 105 * pulse),  # B
                int(150 + 105 * pulse),  # G
                int(150 + 105 * pulse)   # R
            )
            cv2.putText(overlay, operation_text, (text_x, text_y), 
                       self.font, self.font_scale, text_color, self.font_thickness)
        
        # Add minimalist FPS counter with modern font
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        fps_text = f"FPS: {fps}"
        cv2.putText(overlay, fps_text, (w-100, 30), 
                   self.font, 0.6, (0, 255, 255), 1)
        
        # Blend overlay with original frame
        cv2.addWeighted(overlay, self.overlay_alpha, frame, 1 - self.overlay_alpha, 0, frame)
        
        return frame

    def handle_gestures(self, hand_state, hand_landmarks=None):
        """Handle different gesture states and perform corresponding actions"""
        current_time = time.time()
        
        try:
            # YouTube gesture with new animation
            if hand_state == "YOUTUBE" and current_time - self.last_app_launch_time > self.app_launch_cooldown:
                self.show_status("Opening YouTube")
                webbrowser.open(self.youtube_url)
                self.last_app_launch_time = current_time
                
            # Volume control with rapid response
            elif hand_state == "THUMBS_UP":
                if current_time - self.last_volume_time > self.volume_cooldown:
                    self.show_status("Volume Up")
                    pyautogui.press('volumeup', presses=5, interval=0.1)
                    time.sleep(0.01)
                self.last_volume_time = current_time
                
            elif hand_state == "THUMBS_DOWN":
                if current_time - self.last_volume_time > self.volume_cooldown:
                    self.show_status("Volume Down")
                    pyautogui.press('volumedown', presses=5, interval=0.1)
                    time.sleep(0.01)
                self.last_volume_time = current_time
                
            elif hand_state == "THREE_FINGERS":
                if current_time - self.last_task_view_time > self.task_view_cooldown:
                    self.show_status("Task View")
                    pyautogui.hotkey('winleft', 'tab')
                    self.last_task_view_time = current_time
                
            elif hand_state == "BACK_HAND":
                self.show_status("Minimize Window")
                active_window = win32gui.GetForegroundWindow()
                win32gui.ShowWindow(active_window, win32con.SW_MINIMIZE)
                
            elif hand_state == "OPEN_HAND":
                if hand_landmarks:
                    palm_y = hand_landmarks.landmark[9].y  # Middle finger MCP as reference
                    
                    # Simple position-based scrolling
                    if palm_y < (self.scroll_threshold - self.scroll_deadzone):  # Upper zone
                        scroll_amount = self.scroll_speed
                        pyautogui.scroll(scroll_amount)
                        self.show_status("Scroll Up")
                    elif palm_y > (self.scroll_threshold + self.scroll_deadzone):  # Lower zone
                        scroll_amount = -self.scroll_speed
                        pyautogui.scroll(scroll_amount)
                        self.show_status("Scroll Down")
            
            elif hand_state == "FIST":
                self.show_status("Closing Gesture")
                if self.is_alt_tab_active:
                    pyautogui.keyUp('alt')
                    self.is_alt_tab_active = False
                    
        except Exception as e:
            self.show_status(f"Gesture Error: {str(e)}")

    def process_landmarks(self, hand_landmarks):
        """Process hand landmarks for cursor position"""
        try:
            # Get index finger tip position
            index_tip = hand_landmarks.landmark[8]
            
            # Convert to screen coordinates
            screen_x = int(index_tip.x * self.screen_width)
            screen_y = int(index_tip.y * self.screen_height)
            
            if self.prev_x is None:
                self.prev_x = screen_x
                self.prev_y = screen_y
            
            # Apply smoothing
            smooth_x = int(self.prev_x + (screen_x - self.prev_x) * self.movement_smoothing)
            smooth_y = int(self.prev_y + (screen_y - self.prev_y) * self.movement_smoothing)
            
            # Update previous positions
            self.prev_x = smooth_x
            self.prev_y = smooth_y
            
            return (smooth_x, smooth_y)
            
        except Exception as e:
            print(f"Error in process_landmarks: {e}")
            return None

    def process_frame(self, frame):
        """Process each frame from the camera"""
        try:
            # Skip frames to improve performance
            self.frame_count += 1
            if self.frame_count % self.PROCESS_EVERY_N_FRAMES != 0:
                return frame, None
            
            # Convert the BGR image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame and detect hands
            results = self.hands.process(frame_rgb)
            
            cursor_pos = None
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks with improved visibility
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
                    
                    # Get hand state
                    hand_state = self.get_hand_state(hand_landmarks)
                    
                    # Update current gesture text
                    if hand_state == "YOUTUBE":
                        self.current_gesture = "Opening YouTube"
                    elif hand_state == "TASK_VIEW":
                        self.current_gesture = "Task View"
                    elif hand_state == "CURSOR":
                        self.current_gesture = "Cursor Control"
                    elif hand_state == "CLICK":
                        self.current_gesture = "Left Click"
                    elif hand_state == "RIGHT_CLICK":
                        self.current_gesture = "Right Click"
                    elif hand_state == "SCROLL":
                        self.current_gesture = "Scroll Mode"
                    elif hand_state == "VOLUME":
                        self.current_gesture = "Volume Control"
                    else:
                        self.current_gesture = "Detecting..."
                    
                    # Process landmarks for cursor control and gestures
                    cursor_pos = self.process_landmarks(hand_landmarks)
                    
                    # Process the detected hand state
                    self.handle_gestures(hand_state, hand_landmarks)
            else:
                self.current_gesture = "No hand detected"
                # Reset states when no hand is detected
                self.prev_x = None
                self.prev_y = None
                self.prev_hand_state = None
                if self.is_alt_tab_active:
                    pyautogui.keyUp('alt')
                    self.is_alt_tab_active = False
            
            # Add gesture name to frame with better visibility
            text = f"Current Gesture: {self.current_gesture}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Draw background rectangle for better text visibility
            cv2.rectangle(frame, (10, 10), (text_size[0] + 20, 40), (0, 0, 0), -1)
            cv2.putText(frame, text, (15, 30), font, font_scale, (0, 255, 0), thickness)
            
            # Draw overlay with status
            frame = self.draw_overlay(frame)
            
            return frame, cursor_pos
            
        except Exception as e:
            print(f"Error in process_frame: {e}")
            return frame, None

    def run(self):
        """Main run loop"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # Set window to stay on top
        try:
            hwnd = win32gui.FindWindow(None, self.window_name)
            if hwnd:
                style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
                style = style & ~win32con.WS_MINIMIZEBOX
                win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, style)
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                    win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        except Exception as e:
            print(f"Window handling error: {e}")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Flip the frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Process the frame
            frame, cursor_pos = self.process_frame(frame)
            
            # Update cursor position if valid
            if cursor_pos:
                x, y = cursor_pos
                if x is not None and y is not None:
                    pyautogui.moveTo(x, y)
            
            # Display the frame
            cv2.imshow(self.window_name, frame)
            
            # Check for ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    try:
        recognizer = HandGestureRecognizer()
        recognizer.run()
    except Exception as e:
        print(f"Error in main: {e}")
        input("Press Enter to exit...")  # Keep window open on error

if __name__ == "__main__":
    main()