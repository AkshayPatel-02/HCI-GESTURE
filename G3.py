import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from ctypes import windll
import win32gui
import win32con
import os

class HandGestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            model_complexity=1
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
        self.cursor_speed_multiplier = 2
        self.edge_acceleration = 1.1
        self.movement_smoothing = 0.7
        self.prev_x = None
        self.prev_y = None
        
        # Scroll settings
        self.scroll_speed = 200
        self.scroll_smoothing = 0.8
        self.prev_y_position = None
        self.movement_threshold = 0.02
        self.last_scroll_time = time.time()
        self.scroll_cooldown = 0.02
        self.scroll_state = None
        self.scroll_momentum = 0
        self.scroll_momentum_decay = 0.95
        self.continuous_scroll_threshold = 0.15
        
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
        self.app_launch_cooldown = 1.0
        self.last_app_launch_time = 0
        self.peace_sign_threshold = 0.1

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
            
            # Check other fingers
            if index_tip.y < index_mcp.y:
                fingers_extended.append(index)
            if middle_tip.y < middle_mcp.y:
                fingers_extended.append(middle)
            if ring_tip.y < ring_mcp.y:
                fingers_extended.append(ring)
            if pinky_tip.y < pinky_mcp.y:
                fingers_extended.append(pinky)
            
            # Get wrist and middle finger base for palm orientation
            wrist = hand_landmarks.landmark[0]
            middle_finger_base = hand_landmarks.landmark[9]
            
            # Calculate palm orientation more accurately
            palm_direction = middle_finger_base.z - wrist.z
            is_palm_backward = palm_direction > 0.1  # Adjusted threshold
            
            # Determine hand state with improved accuracy
            num_extended = len(fingers_extended)
            
            # Check for three fingers (index, middle, ring)
            three_fingers_up = (index in fingers_extended and 
                              middle in fingers_extended and 
                              ring in fingers_extended and 
                              thumb not in fingers_extended and 
                              pinky not in fingers_extended)
            
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
            
            # Peace sign detection (index and middle fingers up, others down)
            is_peace_sign = (
                index_tip.y < index_mcp.y - 0.15 and  # Index up
                middle_tip.y < middle_mcp.y - 0.15 and  # Middle up
                thumb_tip.y > thumb_mcp.y and  # Thumb down
                ring_tip.y > ring_mcp.y and  # Ring down
                pinky_tip.y > pinky_mcp.y and  # Pinky down
                abs(index_tip.y - middle_tip.y) < 0.1  # Index and middle at similar height
            )
            
            # Return states including new gestures
            if is_thumb_up:
                return "THUMBS_UP"
            elif is_thumb_down:
                return "THUMBS_DOWN"
            elif is_peace_sign:
                return "PEACE_SIGN"
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

    def handle_gestures(self, hand_state):
        """Handle different gesture states and perform corresponding actions"""
        current_time = time.time()
        
        try:
            # Volume control with rapid response
            if hand_state == "THUMBS_UP" and current_time - self.last_volume_time > self.volume_cooldown:
                for _ in range(5):  # Multiple presses for faster volume change
                    pyautogui.press('volumeup')
                    time.sleep(0.01)
                self.last_volume_time = current_time
                
            elif hand_state == "THUMBS_DOWN" and current_time - self.last_volume_time > self.volume_cooldown:
                for _ in range(5):
                    pyautogui.press('volumedown')
                    time.sleep(0.01)
                self.last_volume_time = current_time
                
            # Launch YouTube on peace sign
            elif hand_state == "PEACE_SIGN" and current_time - self.last_app_launch_time > self.app_launch_cooldown:
                try:
                    # Try multiple methods to open YouTube
                    try:
                        # Method 1: Using default browser
                        pyautogui.hotkey('win', 'r')
                        time.sleep(0.1)
                        pyautogui.write('https://www.youtube.com')
                        time.sleep(0.1)
                        pyautogui.press('enter')
                    except:
                        # Method 2: Using command prompt
                        os.system('start https://www.youtube.com')
                    
                    self.last_app_launch_time = current_time
                except Exception as e:
                    print(f"Failed to open YouTube: {e}")
            
            # Handle other gestures if not in cooldown
            elif hand_state != self.prev_hand_state and current_time - self.last_gesture_time > self.gesture_cooldown:
                try:
                    if hand_state == "FIST":
                        # Minimize active window with retry mechanism
                        window = win32gui.GetForegroundWindow()
                        if window:
                            try:
                                win32gui.ShowWindow(window, win32con.SW_MINIMIZE)
                                self.last_gesture_time = current_time
                            except Exception as e:
                                print(f"Failed to minimize window: {e}")
                    
                    elif hand_state == "OPEN_HAND" and self.prev_hand_state == "FIST":
                        # Maximize active window with retry mechanism
                        window = win32gui.GetForegroundWindow()
                        if window:
                            try:
                                # Check if window is minimized
                                placement = win32gui.GetWindowPlacement(window)
                                if placement[1] == win32con.SW_SHOWMINIMIZED:
                                    win32gui.ShowWindow(window, win32con.SW_RESTORE)
                                else:
                                    win32gui.ShowWindow(window, win32con.SW_MAXIMIZE)
                                self.last_gesture_time = current_time
                            except Exception as e:
                                print(f"Failed to maximize window: {e}")
                    
                    elif hand_state == "BACK_HAND":
                        # Simulate browser back button
                        pyautogui.hotkey('alt', 'left')
                        self.last_gesture_time = current_time
                    
                    elif hand_state == "THREE_FINGERS":
                        # Show Task View using Windows + Tab
                        pyautogui.keyDown('win')
                        time.sleep(0.1)  # Small delay to ensure key registration
                        pyautogui.keyDown('tab')
                        time.sleep(0.1)
                        pyautogui.keyUp('tab')
                        pyautogui.keyUp('win')
                        self.last_gesture_time = current_time
                    
                except Exception as e:
                    print(f"Error in handle_gestures: {e}")
            
            self.prev_hand_state = hand_state
            
        except Exception as e:
            print(f"Error in handle_gestures: {e}")
        
        self.prev_hand_state = hand_state

    def process_landmarks(self, hand_landmarks, frame_width, frame_height):
        try:
            # Get current hand state
            hand_state = self.get_hand_state(hand_landmarks)
            self.handle_gestures(hand_state)
            
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            palm_center = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            
            current_time = time.time()
            
            extended_fingers = self.count_extended_fingers(hand_landmarks)
            self.is_scroll_mode = len(extended_fingers) >= 4
            
            if self.is_scroll_mode:
                if self.prev_y_position is not None:
                    y_movement = palm_center.y - self.prev_y_position
                    
                    if abs(y_movement) > self.movement_threshold:
                        self.scroll_state = "up" if y_movement < 0 else "down"
                        self.scroll_momentum = abs(y_movement / self.movement_threshold)
                    elif palm_center.y > (1 - self.continuous_scroll_threshold):
                        self.scroll_state = "down"
                        self.scroll_momentum = 1.0
                    elif palm_center.y < self.continuous_scroll_threshold:
                        self.scroll_state = "up"
                        self.scroll_momentum = 1.0
                    
                    if self.scroll_state and current_time - self.last_scroll_time > self.scroll_cooldown:
                        scroll_amount = int(self.scroll_speed * self.scroll_momentum)
                        if self.scroll_state == "up":
                            pyautogui.scroll(scroll_amount)
                        else:
                            pyautogui.scroll(-scroll_amount)
                        self.last_scroll_time = current_time
                        self.last_click_type = f"Scroll {self.scroll_state.title()}"
                        
                        if not (palm_center.y > (1 - self.continuous_scroll_threshold) or 
                               palm_center.y < self.continuous_scroll_threshold):
                            self.scroll_momentum *= self.scroll_momentum_decay
                
                self.prev_y_position = palm_center.y
                return None, None  # Don't move cursor in scroll mode
            
            normalized_x = (index_tip.x * 1.1) - 0.05
            normalized_y = (index_tip.y * 1.1) - 0.05
            
            normalized_x = max(0, min(1, normalized_x))
            normalized_y = max(0, min(1, normalized_y))
            
            if 0.1 < normalized_x < 0.9 and 0.1 < normalized_y < 0.9:
                edge_mult = 1.0
            else:
                edge_mult = self.edge_acceleration
            
            target_x = int(normalized_x * self.screen_width)
            target_y = int(normalized_y * self.screen_height)
            
            if self.prev_x is not None and self.prev_y is not None:
                x = int(self.prev_x + (target_x - self.prev_x) * self.movement_smoothing)
                y = int(self.prev_y + (target_y - self.prev_y) * self.movement_smoothing)
            else:
                x = target_x
                y = target_y
            
            self.prev_x = x
            self.prev_y = y
            
            dx = x - (self.prev_x or x)
            dy = y - (self.prev_y or y)
            x = int(x + dx * (self.cursor_speed_multiplier - 1) * edge_mult)
            y = int(y + dy * (self.cursor_speed_multiplier - 1) * edge_mult)
            
            x = max(0, min(x, self.screen_width - 1))
            y = max(0, min(y, self.screen_height - 1))
            
            thumb_index_distance = ((thumb_tip.x - index_tip.x) ** 2 + 
                                  (thumb_tip.y - index_tip.y) ** 2) ** 0.5
            
            if thumb_index_distance < self.click_threshold:
                if current_time - self.last_click_time < self.double_click_threshold:
                    if self.click_count == 1:
                        pyautogui.doubleClick()
                        self.click_count = 0
                        self.last_click_type = "double"
                    else:
                        self.click_count += 1
                else:
                    pyautogui.click()
                    self.click_count = 1
                    self.last_click_type = "single"
                self.last_click_time = current_time
            
            if current_time - self.last_click_time > self.double_click_threshold:
                self.click_count = 0
            
            return x, y
        except Exception as e:
            print(f"Error in process_landmarks: {e}")
            return None, None

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        frame_height, frame_width, _ = frame.shape
        
        cursor_pos = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            if self.frame_count % self.PROCESS_EVERY_N_FRAMES == 0:
                cursor_pos = self.process_landmarks(hand_landmarks, frame_width, frame_height)
        
        self.frame_count += 1
        
        status = "Hand Detected - "
        if results.multi_hand_landmarks:
            hand_state = self.get_hand_state(results.multi_hand_landmarks[0])
            status += f"Gesture: {hand_state}"
        else:
            status = "No Hand Detected"
        
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, cursor_pos

def main():
    recognizer = HandGestureRecognizer()
    
    while True:
        ret, frame = recognizer.cap.read()
        if not ret:
            break
            
        frame, cursor_pos = recognizer.process_frame(frame)
        if cursor_pos:
            x, y = cursor_pos
            if x is not None and y is not None:
                pyautogui.moveTo(x, y)
        
        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == 27:  
            break
    
    recognizer.cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
