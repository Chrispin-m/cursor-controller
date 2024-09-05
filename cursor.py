import cv2
import mediapipe as mp
import pyautogui
import time

# MediaPipe hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Variables for click control
index_finger_up = False
click_time = 0
last_click = 0

# Screen dimensions
screen_width, screen_height = pyautogui.size()

cap = cv2.VideoCapture(0)

def control_cursor(finger_tip_x, finger_tip_y):
    """Move cursor based on the finger position."""
    x_pos = int(finger_tip_x * screen_width)
    y_pos = int(finger_tip_y * screen_height)
    pyautogui.moveTo(x_pos, y_pos)

def detect_click(current_time):
    """Detect single and double clicks based on index finger movement."""
    global last_click
    if current_time - last_click < 0.3:  # Double click detection within 300ms
        pyautogui.doubleClick()
        print("Double Click!")
    else:
        pyautogui.click()
        print("Single Click!")
    last_click = current_time

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # landmark positions for index finger
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

            # Normalize 
            x_norm = index_finger_tip.x
            y_norm = index_finger_tip.y
            control_cursor(x_norm, y_norm)

            # if the index finger is up or down (tip above or below PIP joint)
            if index_finger_tip.y < index_finger_pip.y:  # Finger is up
                if not index_finger_up:
                    index_finger_up = True
                    click_time = time.time()

            else:  # Finger is down
                if index_finger_up:
                    index_finger_up = False
                    current_time = time.time()
                    detect_click(current_time)

    #  webcam feed
    cv2.imshow("Hand Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
