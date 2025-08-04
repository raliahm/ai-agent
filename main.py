import numpy as np
import cv2 as cv
import mediapipe as mp
import math 
import clickmanager as clickmanager
import pyautogui
import rerun as rr
from typing import Tuple, Optional
print("🔁 Script started")

# Initialize Rerun for visualization and debugging
rr.init("air_mouse_tracker", spawn=True)
print("📊 Rerun viewer initialized")

# Initialize camera
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Set higher camera resolution for better mouse control
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv.CAP_PROP_FPS, 30)

print(f"📹 Camera resolution: {int(cap.get(cv.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))}")
print(f"🖥️ Screen resolution: {pyautogui.size()}")

# Initialize click manager with shorter cooldown
click_mgr = clickmanager.ClickManager(cooldown_seconds=0.1)

# Add click state tracking
is_currently_pinching: bool = False
click_performed: bool = False

# Add scroll state tracking - simplified for finger curl detection
is_scrolling: bool = False
scroll_accumulator: float = 0.0

# Add gesture stability tracking
gesture_stability_buffer: int = 0
gesture_buffer_threshold: int = 2  # Reduced from 3 to 2 for better responsiveness
previous_gesture: str = "Unknown"

def perform_mouse_click(click_type: str = "left") -> None:
    """
    Perform mouse click using PyAutoGUI.
    
    Args:
        click_type: Type of click ("left", "right", "double")
    """
    try:
        if click_type == "left":
            pyautogui.click()
        elif click_type == "right":
            pyautogui.rightClick()
        elif click_type == "double":
            pyautogui.doubleClick()
        
        print(f"✅ {click_type.capitalize()} click performed")
        
    except Exception as e:
        print(f"❌ Error performing click: {e}")

def move_mouse_cursor(hand_landmarks, img_shape: Tuple[int, int], smoothing_factor: int = 7) -> None:
    """
    Move mouse cursor based on index finger position.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        img_shape: (height, width) of the camera image
        smoothing_factor: Higher values = smoother movement
    """
    try:
        # Get screen dimensions
        screen_width, screen_height = pyautogui.size()
        
        # Get index finger tip position (landmark 8)
        index_tip = hand_landmarks.landmark[8]
        
        # Use normalized coordinates directly (0.0 to 1.0)
        # This gives better coverage of the entire screen
        finger_x_normalized = index_tip.x
        finger_y_normalized = index_tip.y
        
        # Add margin expansion - expand the usable area beyond camera bounds
        # This allows reaching screen edges more easily
        margin = 0.1  # 10% margin expansion
        finger_x_expanded = (finger_x_normalized - 0.5) * (1 + margin) + 0.5
        finger_y_expanded = (finger_y_normalized - 0.5) * (1 + margin) + 0.5
        
        # Clamp to valid range
        finger_x_expanded = max(0.0, min(1.0, finger_x_expanded))
        finger_y_expanded = max(0.0, min(1.0, finger_y_expanded))
        
        # Map to screen coordinates
        screen_x = int(finger_x_expanded * screen_width)
        screen_y = int(finger_y_expanded * screen_height)
        
        # Move cursor (PyAutoGUI handles smoothing internally)
        pyautogui.moveTo(screen_x, screen_y)
        
    except Exception as e:
        print(f"❌ Error moving cursor: {e}")

def finger_is_up(hand_landmarks, tip_id: int, pip_id: int) -> bool:
    """
    Check if finger is up based on landmark positions.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        tip_id: Finger tip landmark ID
        pip_id: PIP joint landmark ID
        
    Returns:
        True if finger is extended, False otherwise
    """
    return hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y

def is_thumb_up(hand_landmarks, handedness: str) -> bool:
    """
    Check if thumb is up based on landmarks.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        handedness: Hand orientation ("Right" or "Left")
        
    Returns:
        True if thumb is extended, False otherwise
    """
    thumb_tip = hand_landmarks.landmark[4]    # Thumb tip
    thumb_ip = hand_landmarks.landmark[3]     # Thumb IP joint
        
    if handedness == "Right":
        return thumb_tip.x > thumb_ip.x  # Thumb pointing right
    else:
        return thumb_tip.x < thumb_ip.x  # Thumb pointing left

def euclidean_distance(lm1, lm2, w: int, h: int) -> float:
    """
    Calculate Euclidean distance between two landmarks.
    
    Args:
        lm1: First landmark
        lm2: Second landmark
        w: Image width
        h: Image height
        
    Returns:
        Distance in pixels
    """
    x1, y1 = int(lm1.x * w), int(lm1.y * h)
    x2, y2 = int(lm2.x * w), int(lm2.y * h)
    return math.hypot(x2 - x1, y2 - y1)

def detect_click_gesture(hand_landmarks, w: int, h: int, click_threshold: float = 40.0) -> bool:
    """
    Detect click gesture by measuring distance between thumb and index finger.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        w: Image width
        h: Image height
        click_threshold: Distance threshold for click detection
        
    Returns:
        True if click gesture detected, False otherwise
    """
    try:
        # Get thumb tip (landmark 4) and index finger tip (landmark 8)
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        
        # Calculate distance in pixels
        distance = euclidean_distance(thumb_tip, index_tip, w, h)
        
        # Return True if fingers are close enough (pinching gesture)
        return distance < click_threshold
        
    except (IndexError, AttributeError) as e:
        print(f"Error in click detection: {e}")
        return False

def detect_scroll_gesture(hand_landmarks, w: int, h: int, handedness: str = "Right") -> str:
    """
    Detect scroll gesture based on finger curl intensity while keeping wrist stationary.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        w: Image width
        h: Image height
        handedness: Hand orientation
        
    Returns:
        "scroll_up", "scroll_down", or "none"
    """
    try:
        # Calculate how curled each finger is (distance from tip to base)
        wrist = hand_landmarks.landmark[0]
        
        # Get finger tip to wrist distances (normalized)
        fingers_extended = []
        finger_landmarks = [
            (4, 2),   # Thumb (tip, base)
            (8, 5),   # Index 
            (12, 9),  # Middle
            (16, 13), # Ring
            (20, 17)  # Pinky
        ]
        
        for tip_id, base_id in finger_landmarks:
            tip = hand_landmarks.landmark[tip_id]
            base = hand_landmarks.landmark[base_id]
            
            # Distance from tip to base (how extended the finger is)
            tip_to_base_dist = math.sqrt((tip.x - base.x)**2 + (tip.y - base.y)**2)
            fingers_extended.append(tip_to_base_dist)
        
        # Calculate average finger extension
        avg_extension = sum(fingers_extended) / len(fingers_extended)
        
        # Log finger extension data to Rerun
        rr.log("hand_analysis/finger_extensions", rr.Scalars(avg_extension))
        rr.log("hand_analysis/individual_fingers", rr.BarChart(fingers_extended))
        
        # Thresholds for scroll detection
        curl_threshold_high = 0.08   # Very curled = scroll down
        curl_threshold_low = 0.15    # Extended = scroll up
        
        # Log thresholds to Rerun
        rr.log("hand_analysis/threshold_high", rr.Scalars(curl_threshold_high))
        rr.log("hand_analysis/threshold_low", rr.Scalars(curl_threshold_low))
        
        scroll_result = "none"
        if avg_extension < curl_threshold_high:
            scroll_result = "scroll_down"  # Fingers very curled
        elif avg_extension > curl_threshold_low:
            scroll_result = "scroll_up"    # Fingers extended
        
        # Log scroll decision to Rerun
        rr.log("gesture_detection/scroll_direction", rr.TextLog(scroll_result))
        
        return scroll_result
        
    except Exception as e:
        print(f"❌ Error in scroll detection: {e}")
        rr.log("errors/scroll_detection", rr.TextLog(f"Error: {e}"))
        return "none"

def get_mouse_position(hand_landmarks, img_shape: Tuple[int, int]) -> Tuple[int, int]:
    """
    Get current mouse position based on hand landmarks.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        img_shape: (height, width) of the camera image
        
    Returns:
        Tuple of (screen_x, screen_y) coordinates
    """
    try:
        screen_width, screen_height = pyautogui.size()
        index_tip = hand_landmarks.landmark[8]
        
        # Use the same coordinate mapping as move_mouse_cursor
        finger_x_normalized = index_tip.x
        finger_y_normalized = index_tip.y
        
        margin = 0.1
        finger_x_expanded = (finger_x_normalized - 0.5) * (1 + margin) + 0.5
        finger_y_expanded = (finger_y_normalized - 0.5) * (1 + margin) + 0.5
        
        finger_x_expanded = max(0.0, min(1.0, finger_x_expanded))
        finger_y_expanded = max(0.0, min(1.0, finger_y_expanded))
        
        screen_x = int(finger_x_expanded * screen_width)
        screen_y = int(finger_y_expanded * screen_height)
        
        return (screen_x, screen_y)
        
    except Exception as e:
        print(f"❌ Error getting mouse position: {e}")
        return (0, 0)

def start_scroll() -> None:
    """Start scrolling operation."""
    global is_scrolling, scroll_accumulator
    
    try:
        if not is_scrolling:
            is_scrolling = True
            scroll_accumulator = 0.0
            print("📜 Scroll mode activated - curl/extend fingers to scroll")
    except Exception as e:
        print(f"❌ Error starting scroll: {e}")

def perform_scroll(scroll_direction: str) -> None:
    """Perform scroll action based on finger curl."""
    global scroll_accumulator
    
    try:
        if is_scrolling:
            # Accumulate scroll actions to prevent too rapid scrolling
            scroll_rate = 0.4  # Adjust this to control scroll speed
            
            # Log scroll accumulator to Rerun
            rr.log("scroll_system/accumulator", rr.Scalars(scroll_accumulator))
            rr.log("scroll_system/rate", rr.Scalars(scroll_rate))
            
            if scroll_direction == "scroll_down":
                scroll_accumulator += scroll_rate
                if scroll_accumulator >= 1.0:
                    scroll_amount = int(scroll_accumulator) * 20  # Increased for more distance
                    pyautogui.scroll(-scroll_amount)  # Negative for down
                    print(f"📜 Scroll down ({scroll_amount} units)")
                    
                    # Log actual scroll action to Rerun
                    rr.log("scroll_system/scroll_events", rr.TextLog(f"Scroll down: {scroll_amount} units"))
                    rr.log("scroll_system/scroll_amount", rr.Scalars(-scroll_amount))
                    
                    scroll_accumulator = 0.0
                    
            elif scroll_direction == "scroll_up":
                scroll_accumulator += scroll_rate
                if scroll_accumulator >= 1.0:
                    scroll_amount = int(scroll_accumulator) * 20  # Increased for more distance
                    pyautogui.scroll(scroll_amount)   # Positive for up
                    print(f"📜 Scroll up ({scroll_amount} units)")
                    
                    # Log actual scroll action to Rerun
                    rr.log("scroll_system/scroll_events", rr.TextLog(f"Scroll up: {scroll_amount} units"))
                    rr.log("scroll_system/scroll_amount", rr.Scalars(scroll_amount))
                    
                    scroll_accumulator = 0.0
            else:
                # Neutral position - decay accumulator
                scroll_accumulator *= 0.8
                
    except Exception as e:
        print(f"❌ Error performing scroll: {e}")
        rr.log("errors/scroll_perform", rr.TextLog(f"Error: {e}"))

def end_scroll() -> None:
    """End scrolling operation."""
    global is_scrolling, scroll_accumulator
    
    try:
        if is_scrolling:
            is_scrolling = False
            print(f"📜 Scroll mode ended")
            scroll_accumulator = 0.0
    except Exception as e:
        print(f"❌ Error ending scroll: {e}")

def classify_gesture(hand_landmarks, w: int, h: int, handedness: str = "Right") -> str:
    """
    Classify hand gesture with finger-curl scroll functionality.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        w: Image width
        h: Image height
        handedness: Hand orientation
        
    Returns:
        Detected gesture name
    """
    global click_mgr, is_currently_pinching, click_performed, is_scrolling
    global gesture_stability_buffer, gesture_buffer_threshold, previous_gesture
    
    try:
        # Check for scroll gesture first
        scroll_direction = detect_scroll_gesture(hand_landmarks, w, h, handedness)
        
        if scroll_direction != "none":
            if not is_scrolling:
                # Start scroll after just 2 consecutive detections
                if gesture_stability_buffer >= gesture_buffer_threshold:
                    start_scroll()
                    gesture_stability_buffer = 0
                    previous_gesture = "Scrolling"
                else:
                    gesture_stability_buffer += 1
                    return "Preparing Scroll"
            else:
                # Already scrolling, perform scroll action
                perform_scroll(scroll_direction)
                gesture_stability_buffer = 0
                previous_gesture = "Scrolling"
            
            if scroll_direction == "scroll_down":
                return "Scrolling Down"
            else:
                return "Scrolling Up"
        else:
            # Not in scroll mode
            if is_scrolling:
                # End scroll when returning to neutral
                end_scroll()
                gesture_stability_buffer = 0
                previous_gesture = "Unknown"
            else:
                gesture_stability_buffer = 0
        
        # Check for pinch click (only if not scrolling)
        if not is_scrolling:
            pinch_detected = detect_click_gesture(hand_landmarks, w, h)
            
            if pinch_detected:
                if not is_currently_pinching:
                    is_currently_pinching = True
                    click_performed = False
                    
                    if click_mgr.can_click():
                        perform_mouse_click("left")
                        click_mgr.register_click()
                        click_performed = True
                        
                return "Clicking"
            else:
                if is_currently_pinching:
                    is_currently_pinching = False
                    click_performed = False
                    click_mgr.reset_click_state()
        
        # Get finger states for other gestures (only if not scrolling)
        if not is_scrolling:
            fingers = []
            fingers.append(is_thumb_up(hand_landmarks, handedness))  # Thumb
            fingers.append(finger_is_up(hand_landmarks, 8, 6))       # Index
            fingers.append(finger_is_up(hand_landmarks, 12, 10))     # Middle
            fingers.append(finger_is_up(hand_landmarks, 16, 14))     # Ring
            fingers.append(finger_is_up(hand_landmarks, 20, 18))     # Pinky

            # Check for mouse control (index finger only)
            index_up = fingers[1]
            others_down = not fingers[2] and not fingers[3] and not fingers[4]

            # Calculate distance from index fingertip to wrist
            dist_index_wrist = euclidean_distance(
                hand_landmarks.landmark[8],  # index fingertip
                hand_landmarks.landmark[0],  # wrist
                w, h
            )

            # Distance threshold to confirm index is extended
            distance_threshold = 0.2 * w

            if index_up and others_down and dist_index_wrist > distance_threshold and not fingers[0]:
                # Move mouse cursor when pointing (excluding thumb up)
                move_mouse_cursor(hand_landmarks, (h, w))
                return "Mouse Control"
            
            # Other gesture classifications
            if all(fingers):
                return "Open Palm"
            elif not any(fingers):
                return "Fist (Ready to Scroll)"
            elif fingers[0] and not any(fingers[1:]):
                return "Thumbs Up"
            elif index_up and others_down:
                return "Pointing"

        return "Unknown"
        
    except Exception as e:
        print(f"❌ Error in gesture classification: {e}")
        return "Error"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Single hand for better performance
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

print("🖱️ Air Mouse with Click and Finger-Curl Scroll Detection Active!")
print("Gestures:")
print("- Index finger only: Mouse control")
print("- Pinch (thumb + index close): Click")
print("- Curl fingers (keep wrist still): Scroll down")
print("- Extend fingers (keep wrist still): Scroll up")
print("- Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    # Flip horizontally for mirror effect
    frame = cv.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert BGR to RGB for MediaPipe
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture_text = "No Hand Detected"

    # Log camera frame to Rerun
    rr.log("camera/rgb_frame", rr.Image(rgb))
    rr.log("camera/processed_frame", rr.Image(frame))

    # Process hand landmarks
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )
            
            # Log hand landmarks to Rerun as 3D points
            landmarks_3d = []
            landmarks_2d = []
            for landmark in hand_landmarks.landmark:
                # 3D coordinates (MediaPipe provides x, y, z)
                landmarks_3d.append([landmark.x, landmark.y, landmark.z])
                # 2D coordinates for overlay
                landmarks_2d.append([landmark.x * w, landmark.y * h])
            
            rr.log(f"hands/hand_{i}/landmarks_3d", rr.Points3D(landmarks_3d, radii=0.01))
            rr.log(f"hands/hand_{i}/landmarks_2d", rr.Points2D(landmarks_2d, radii=3))
            
            # Get handedness
            handedness = "Right"
            if results.multi_handedness and i < len(results.multi_handedness):
                handedness = results.multi_handedness[i].classification[0].label
            
            rr.log(f"hands/hand_{i}/handedness", rr.TextLog(handedness))
            
            # Classify gesture and perform actions
            gesture_text = classify_gesture(hand_landmarks, w, h, handedness)
            
            # Log gesture classification to Rerun
            rr.log("gesture_detection/current_gesture", rr.TextLog(gesture_text))
            rr.log("gesture_detection/stability_buffer", rr.Scalars(gesture_stability_buffer))
            
            # Log mouse cursor position
            current_mouse_pos = pyautogui.position()
            rr.log("mouse/cursor_position", rr.Points2D([[current_mouse_pos.x, current_mouse_pos.y]], radii=5))
            
            # Visual feedback for click detection
            if gesture_text == "Clicking":
                thumb_pos = (int(hand_landmarks.landmark[4].x * w), 
                           int(hand_landmarks.landmark[4].y * h))
                index_pos = (int(hand_landmarks.landmark[8].x * w), 
                           int(hand_landmarks.landmark[8].y * h))
                
                # Log click gesture details to Rerun
                click_distance = euclidean_distance(hand_landmarks.landmark[4], hand_landmarks.landmark[8], w, h)
                rr.log("gesture_detection/click_distance", rr.Scalars(click_distance))
                rr.log("gesture_detection/click_positions", rr.Points2D([thumb_pos, index_pos], radii=8))
                
                # Different colors based on click state
                color = (0, 255, 0) if click_performed else (0, 255, 255)  # Green if clicked, yellow if pinching
                
                cv.line(frame, thumb_pos, index_pos, color, 4)
                cv.circle(frame, thumb_pos, 8, color, cv.FILLED)
                cv.circle(frame, index_pos, 8, color, cv.FILLED)
            
            # Visual feedback for scroll detection
            elif "Scrolling" in gesture_text:
                # Draw scroll indicator around wrist
                wrist_pos = (int(hand_landmarks.landmark[0].x * w), 
                           int(hand_landmarks.landmark[0].y * h))
                
                # Log wrist position to Rerun
                rr.log("gesture_detection/wrist_position", rr.Points2D([wrist_pos], radii=10))
                
                # Draw scroll zone around wrist
                if "Down" in gesture_text:
                    cv.circle(frame, wrist_pos, 30, (0, 0, 255), 3)  # Red for down
                    cv.putText(frame, "↓", (wrist_pos[0] - 10, wrist_pos[1] + 10), 
                              cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    rr.log("gesture_detection/scroll_direction_visual", rr.TextLog("DOWN ↓"))
                else:
                    cv.circle(frame, wrist_pos, 30, (0, 255, 0), 3)  # Green for up
                    cv.putText(frame, "↑", (wrist_pos[0] - 10, wrist_pos[1] + 10), 
                              cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    rr.log("gesture_detection/scroll_direction_visual", rr.TextLog("UP ↑"))

    # Display gesture with color coding
    color = (0, 255, 0) if gesture_text in ["Mouse Control", "Clicking", "Scrolling Up", "Scrolling Down"] else (255, 255, 255)
    cv.putText(frame, f"Gesture: {gesture_text}", (10, 30), 
              cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Log system state to Rerun
    rr.log("system_state/is_pinching", rr.Scalars(1.0 if is_currently_pinching else 0.0))
    rr.log("system_state/is_scrolling", rr.Scalars(1.0 if is_scrolling else 0.0))
    rr.log("system_state/click_performed", rr.Scalars(1.0 if click_performed else 0.0))

    # Display click state
    if is_currently_pinching:
        status = "CLICKED!" if click_performed else "PINCHING..."
        cv.putText(frame, status, (10, 70), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Display scroll state
    if is_scrolling:
        cv.putText(frame, "SCROLL MODE ACTIVE", (10, 70), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    
    # Display preparation state for scroll
    if gesture_text == "Preparing Scroll":
        cv.putText(frame, "PREPARING SCROLL...", (10, 70), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        # Draw stability indicator
        cv.putText(frame, f"Stability: {gesture_stability_buffer}/{gesture_buffer_threshold}", (10, 110), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Display instructions
    cv.putText(frame, "Point: Mouse | Pinch: Click | Curl/Extend: Scroll | Keep wrist still!", (10, h - 30), 
              cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Log final processed frame to Rerun
    rr.log("camera/final_frame_with_ui", rr.Image(frame))

    # Show frame
    cv.imshow('Air Mouse Interface', frame)

    if cv.waitKey(1) == ord('q'):
        print("Exiting...")
        break

cap.release()
cv.destroyAllWindows()
