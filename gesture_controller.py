"""
Real-Time Gesture-Controlled Spotify Media Player
With Embedded Signal Signal Processing Using
Raspberry Pi

Author: Pacate, Reyjean    
        Galvez, John June Yusuke  
        Bihasa, Nicole
        Abrugena, Angelo

Note: Guys, We have used AI for commenting just so you know what these code snippets actually do. So when you present this your professor you might actually tell what each part does.
You can remove these comments. 
"""

import cv2
import pyautogui
from collections import deque
import time
import sys

# Try importing MediaPipe with error handling
try:
    import mediapipe as mp
    if not hasattr(mp, 'solutions'):
        print("✗ Error: MediaPipe is installed but 'solutions' module is missing")
        print("  Try reinstalling: pip uninstall mediapipe && pip install mediapipe")
        sys.exit(1)
except ImportError:
    print("✗ Error: MediaPipe is not installed")
    print("  Install it with: pip install mediapipe")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Camera settings
CAMERA_INDEX = 0  # USB webcam (usually 0, try 1 if it doesn't work)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Gesture detection settings
BUFFER_SIZE = 5  # Number of frames to smooth gesture detection
CONFIDENCE_THRESHOLD = 0.7  # Minimum hand detection confidence (0-1)
COMMAND_COOLDOWN = 1.5  # Seconds between sending same command (prevents spam)

# MediaPipe hand detection settings
MAX_HANDS = 1  # Detect only one hand for simplicity
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5


# ============================================================================
# GESTURE TO COMMAND MAPPING
# ============================================================================

GESTURE_COMMANDS = {
    0: ('playpause', 'Play/Pause'),  # Changed from 'pause' to 'playpause'
    1: ('playpause', 'Play/Pause'),  # Same as 0 - toggle play/pause
    2: ('nexttrack', 'Next Track'),
    3: ('prevtrack', 'Previous Track'),
    4: ('volumeup', 'Volume Up'),
    5: ('volumedown', 'Volume Down')
}


# ============================================================================
# HAND LANDMARK DETECTION AND FINGER COUNTING
# ============================================================================

class HandGestureDetector:
    """Detects hand gestures and counts raised fingers using MediaPipe"""

    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=MAX_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Finger tip and base landmark IDs
        # Each finger has a tip and a PIP (middle joint) we compare
        # Thumb, Index, Middle, Ring, Pinky
        self.finger_tips = [4, 8, 12, 16, 20]
        self.finger_pips = [2, 6, 10, 14, 18]  # Base joints for comparison

    def count_fingers(self, hand_landmarks, handedness):
        """
        Count how many fingers are raised

        Args:
            hand_landmarks: MediaPipe hand landmark object
            handedness: Whether hand is 'Left' or 'Right'

        Returns:
            int: Number of raised fingers (0-5)
        """
        if hand_landmarks is None:
            return None

        fingers_up = 0
        landmarks = hand_landmarks.landmark

        # Determine if hand is left or right
        is_right_hand = handedness == 'Right'

        # Thumb: Special case (checks horizontal position instead of vertical)
        # For right hand: thumb is up if tip is to the right of base
        # For left hand: thumb is up if tip is to the left of base
        thumb_tip = landmarks[self.finger_tips[0]]
        thumb_base = landmarks[self.finger_pips[0]]

        if is_right_hand:
            if thumb_tip.x > thumb_base.x:
                fingers_up += 1
        else:
            if thumb_tip.x < thumb_base.x:
                fingers_up += 1

        # Other four fingers: Check if tip is above PIP joint (raised)
        # Lower y value means higher on screen
        for i in range(1, 5):
            tip = landmarks[self.finger_tips[i]]
            pip = landmarks[self.finger_pips[i]]

            if tip.y < pip.y:  # Finger is raised
                fingers_up += 1

        return fingers_up

    def detect_hand(self, frame):
        """
        Detect hand in frame and count fingers

        Args:
            frame: OpenCV BGR image

        Returns:
            tuple: (finger_count, annotated_frame)
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        finger_count = None

        # Process detected hands
        if results.multi_hand_landmarks and results.multi_handedness:
            # Get first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label

            # Draw hand landmarks on frame
            self.mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )

            # Count fingers
            finger_count = self.count_fingers(hand_landmarks, handedness)

        return finger_count, frame


# ============================================================================
# GESTURE SMOOTHING AND COMMAND EXECUTION
# ============================================================================

class GestureController:
    """Smooths gesture detection and sends media control commands"""

    def __init__(self, buffer_size=BUFFER_SIZE):
        self.gesture_buffer = deque(maxlen=buffer_size)
        self.last_command_time = {}
        self.current_gesture = None

    def add_gesture(self, finger_count):
        """Add detected gesture to smoothing buffer"""
        if finger_count is not None:
            self.gesture_buffer.append(finger_count)

    def get_stable_gesture(self):
        """
        Get the most common gesture from buffer
        Only returns gesture if buffer is full and majority agree
        """
        if len(self.gesture_buffer) < self.gesture_buffer.maxlen:
            return None

        # Find most common gesture in buffer
        gesture_counts = {}
        for gesture in self.gesture_buffer:
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1

        most_common_gesture = max(gesture_counts, key=gesture_counts.get)

        # Require at least 60% agreement
        if gesture_counts[most_common_gesture] >= len(self.gesture_buffer) * 0.6:
            return most_common_gesture

        return None

    def execute_command(self, gesture):
        """
        Execute media control command for detected gesture
        Includes cooldown to prevent command spam
        """
        current_time = time.time()

        # Check if enough time has passed since last command
        if gesture in self.last_command_time:
            time_since_last = current_time - self.last_command_time[gesture]
            if time_since_last < COMMAND_COOLDOWN:
                return False

        # Execute command based on gesture
        if gesture in GESTURE_COMMANDS:
            command_key, command_name = GESTURE_COMMANDS[gesture]

            try:
                # Send media key press
                pyautogui.press(command_key)
                print(f"✓ Executed: {command_name} ({gesture} fingers)")
                self.last_command_time[gesture] = current_time
                self.current_gesture = gesture
                return True
            except Exception as e:
                print(f"✗ Error executing command: {e}")
                return False

        return False


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application loop"""

    print("=" * 60)
    print("Hand Gesture Spotify Controller")
    print("=" * 60)
    print("\nGesture Commands:")
    for fingers, (_, name) in GESTURE_COMMANDS.items():
        print(f"  {fingers} fingers → {name}")
    print("\nPress 'q' to quit")
    print("=" * 60)

    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("✗ Error: Could not open webcam")
        return

    print("✓ Webcam initialized")

    # Initialize detector and controller
    detector = HandGestureDetector()
    controller = GestureController()

    # FPS calculation
    prev_time = 0

    try:
        while True:
            # Capture frame
            success, frame = cap.read()
            if not success:
                print("✗ Error reading frame")
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Detect hand and count fingers
            finger_count, annotated_frame = detector.detect_hand(frame)

            # Add to smoothing buffer
            controller.add_gesture(finger_count)

            # Get stable gesture
            stable_gesture = controller.get_stable_gesture()

            # Execute command if stable gesture detected and different from current
            if stable_gesture is not None and stable_gesture != controller.current_gesture:
                controller.execute_command(stable_gesture)

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time

            # Draw UI elements on frame
            # Status bar background
            cv2.rectangle(annotated_frame, (0, 0), (640, 80), (0, 0, 0), -1)

            # Display FPS
            cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display detected fingers
            if finger_count is not None:
                cv2.putText(annotated_frame, f"Fingers: {finger_count}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(annotated_frame, "No hand detected", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display current command
            if controller.current_gesture is not None:
                _, command_name = GESTURE_COMMANDS[controller.current_gesture]
                cv2.putText(annotated_frame, f"Command: {command_name}", (320, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Show frame
            cv2.imshow('Hand Gesture Spotify Controller', annotated_frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nExiting...")
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        detector.hands.close()
        print("✓ Cleanup complete")


if __name__ == "__main__":
    main()


