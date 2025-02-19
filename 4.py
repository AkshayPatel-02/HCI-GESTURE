import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

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

    def count_extended_fingers(self, hand_landmarks):
        finger_tips = [8, 12, 16, 20]  
        finger_pips = [6, 10, 14, 18]
        
        extended_fingers = []
        for tip, pip in zip(finger_tips, finger_pips):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                extended_fingers.append(tip)
        
        return extended_fingers

    def process_landmarks(self, hand_landmarks, frame_width, frame_height):
        try:
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
            if self.is_scroll_mode:
                if self.last_click_type and "Scroll" in self.last_click_type:
                    status += self.last_click_type
                else:
                    status += "Scroll Mode"
            elif self.last_click_type == "double":
                status += "Double Click!"
            elif self.last_click_type == "single":
                status += "Single Click"
            else:
                status += "Ready"
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