import cv2
import mediapipe as mp
import numpy as np
import random
import time
import threading

# Function to reset game variables
def reset_game():
    global score, high_score, lives, circle_position, circle_collected, circle_appear_time, game_started, game_start_time
    score = 0
    lives = 3
    circle_position = (random.randint(50, 590), random.randint(50, 430))
    circle_collected = False
    circle_appear_time = time.time()
    game_started = False

# Initialize MediaPipe Hands and OpenCV video capture
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)  # Adjust confidence values
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Game variables
circle_radius = 30
high_score = 0  # Initialize high score
reset_game()  # Start the game for the first time

# Global variable for the frame
frame = None

# Function to capture video in a separate thread
def capture_video():
    global frame
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            continue
        frame = cv2.flip(img, 1)  # Flip image horizontally

# Start video capture thread
threading.Thread(target=capture_video, daemon=True).start()

def is_hand_over_circle(hand_x, hand_y, circle_x, circle_y, radius):
    # Check if the hand is over the circle using distance
    distance = np.sqrt((hand_x - circle_x) ** 2 + (hand_y - circle_y) ** 2)
    return distance < radius

# Main loop
while True:
    if frame is None:
        continue

    img = frame.copy()  # Work with the latest frame
    img_height, img_width, _ = img.shape

    # Convert image to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Check for hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of the index finger tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_x = int(index_finger_tip.x * img_width)
            index_y = int(index_finger_tip.y * img_height)

            # Check if the index finger is raised
            finger_positions = [hand_landmarks.landmark[i].y for i in range(5)]
            if finger_positions[0] < finger_positions[1] and finger_positions[0] < finger_positions[2]:
                # If index finger is raised, start the game immediately
                if not game_started:  
                    game_started = True
                    circle_appear_time = time.time()
                    game_start_time = time.time()

            # Check if hand is over the circle only if the game has started
            if game_started and is_hand_over_circle(index_x, index_y, circle_position[0], circle_position[1], circle_radius):
                score += 1
                circle_collected = True
                circle_appear_time = time.time()

    # Check if the time for the circle has exceeded
    if game_started:
        elapsed_time = time.time() - circle_appear_time
        if elapsed_time > 5:  # Circle disappears after 5 seconds
            lives -= 1
            circle_position = (random.randint(50, 590), random.randint(50, 430))
            circle_appear_time = time.time()

    # Generate a new circle if the old one is collected
    if circle_collected:
        circle_position = (random.randint(50, 590), random.randint(50, 430))
        circle_collected = False

    # Draw the circle on the screen in red
    if game_started:
        cv2.circle(img, circle_position, circle_radius, (0, 0, 255), -1)

        # Calculate remaining time for the current circle
        remaining_time = max(0, 5 - int(elapsed_time))
        total_game_time = int(time.time() - game_start_time)

        # Display the score, lives, and high score
        cv2.putText(img, f"Score: {score}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Lives: {lives}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"High Score: {high_score}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Time: {remaining_time}", (img_width - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Total Time: {total_game_time}", (img_width - 200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if lives <= 0:
            if score > high_score:
                high_score = score
            cv2.putText(img, "Game Over!", (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            cv2.putText(img, f"High Score: {high_score}", (230, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, "Press 'R' to reset!", (180, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Hand Tracking Game", img)

            key = cv2.waitKey(0)
            if key & 0xFF == ord('r'):
                reset_game()
                continue
            else:
                break

    else:
        cv2.putText(img, "Raise your hand to start!", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Hand Tracking Game", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
