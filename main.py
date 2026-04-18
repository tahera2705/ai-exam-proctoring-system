import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from datetime import datetime
import time
import pyautogui
import numpy as np
import csv
import os

def eye_aspect_ratio(eye):
    # Compute Eye Aspect Ratio (EAR)
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize MediaPipe Face Landmarker
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, output_facial_transformation_matrixes=True)
detector = vision.FaceLandmarker.create_from_options(options)

# Start webcam
cap = cv2.VideoCapture(0)

# Variables
log = []
start_time = None
screenshot_timer = time.time()
blink_start = None
eye_closed_threshold = 0.25

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Create mp.Image
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    # Detect
    results = detector.detect(image)

    num_faces = len(results.face_landmarks) if results.face_landmarks else 0
    status = "No Face"
    angles_text = ""

    if num_faces > 1:
        status = "Multiple Faces Detected"
    elif num_faces == 1:
        face_landmarks = results.face_landmarks[0]  # Assume first face

        # Eye blink detection
        # Left eye landmarks: [159, 145, 158, 133, 153, 144]
        left_eye = np.array([(face_landmarks[159].x, face_landmarks[159].y),
                             (face_landmarks[145].x, face_landmarks[145].y),
                             (face_landmarks[158].x, face_landmarks[158].y),
                             (face_landmarks[133].x, face_landmarks[133].y),
                             (face_landmarks[153].x, face_landmarks[153].y),
                             (face_landmarks[144].x, face_landmarks[144].y)])
        # Right eye landmarks: [386, 374, 387, 263, 373, 380]
        right_eye = np.array([(face_landmarks[386].x, face_landmarks[386].y),
                              (face_landmarks[374].x, face_landmarks[374].y),
                              (face_landmarks[387].x, face_landmarks[387].y),
                              (face_landmarks[263].x, face_landmarks[263].y),
                              (face_landmarks[373].x, face_landmarks[373].y),
                              (face_landmarks[380].x, face_landmarks[380].y)])

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < eye_closed_threshold:
            if blink_start is None:
                blink_start = time.time()
            if time.time() - blink_start > 0.5:  # Eyes closed for >0.5 second
                status = "Eyes Closed (Suspicious)"
        else:
            blink_start = None

        # Nose landmark (approx center)
        nose = face_landmarks[1]

        # Head direction detection (basic)
        if nose.x > 0.6:
            status = "Looking Right"
        elif nose.x < 0.4:
            status = "Looking Left"
        else:
            status = "Looking Forward"

        # Compute head pose angles if available
        if results.facial_transformation_matrixes:
            matrix = results.facial_transformation_matrixes[0]
            rotation_matrix = matrix[:3, :3]

            # Compute Euler angles (pitch, yaw, roll)
            sy = np.sqrt(rotation_matrix[0,0]**2 + rotation_matrix[1,0]**2)
            singular = sy < 1e-6

            if not singular:
                pitch = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
                yaw = np.arctan2(-rotation_matrix[2,0], sy)
                roll = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
            else:
                pitch = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
                yaw = np.arctan2(-rotation_matrix[2,0], sy)
                roll = 0

            # Convert to degrees
            pitch_deg = np.degrees(pitch)
            yaw_deg = np.degrees(yaw)
            roll_deg = np.degrees(roll)

            angles_text = f"Tilt: {roll_deg:.1f}° Yaw: {yaw_deg:.1f}° Pitch: {pitch_deg:.1f}°"

            # Update status based on angles
            if abs(yaw_deg) > 30:
                status = f"Looking {'Right' if yaw_deg > 0 else 'Left'} ({abs(yaw_deg):.1f}°)"
            elif abs(pitch_deg) > 20:
                status = f"Tilted ({pitch_deg:.1f}°)"
            else:
                status = "Looking Forward"

    # 🔥 Periodic screenshots
    if time.time() - screenshot_timer > 10:
        try:
            screenshot = pyautogui.screenshot()
            screenshot.save(f"screenshot_{int(time.time())}.png")
            screenshot_timer = time.time()
        except Exception as e:
            print(f"Screenshot error: {e}")

    # 🔥 Time-based cheating detection
    if status != "Looking Forward" and status != "No Face" and "Eyes Closed" not in status:
        if start_time is None:
            start_time = time.time()
        elif time.time() - start_time > 2:
            status = "Cheating Detected"
    else:
        start_time = None

    # 🔥 Screen switch detection
    try:
        current_window = pyautogui.getActiveWindowTitle()
        if current_window and "Visual Studio Code" not in current_window:
            status = "Screen Switched (Suspicious)"
    except:
        pass

    # Logging
    log.append((status, datetime.now()))

    # Show status on screen
    cv2.putText(frame, status, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if angles_text:
        cv2.putText(frame, angles_text, (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if num_faces == 1:
        cv2.putText(frame, f"EAR: {ear:.2f}", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("AI Proctoring System", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
detector.close()

# Export log to CSV
try:
    with open('proctor_log.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Status', 'Timestamp'])
        for status_entry, ts in log:
            writer.writerow([status_entry, ts.strftime('%Y-%m-%d %H:%M:%S')])
    print("Log exported to proctor_log.csv")
except Exception as e:
    print(f"Log export error: {e}")