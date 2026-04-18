 AI-Based Exam Proctoring System

Overview
This project implements a **real-time AI-based exam proctoring system** using computer vision techniques to monitor user behavior during online exams.
The system analyzes webcam input to detect **suspicious activities** such as head movement, eye closure, multiple faces, and screen switching.

---

Key Features

* Facial Landmark Detection using MediaPipe FaceLandmarker API
* Head Pose Estimation (Yaw, Pitch, Roll)
* Time-Based Cheating Detection (detects prolonged distraction)
* Multiple Face Detection (flags unauthorized presence)
* Eye Closure Detection (EAR-based)
* Screen Switching Detection (detects tab/app switching)
* Periodic Screenshot Capture (every 10 seconds)
* Event Logging (exports logs to CSV with timestamps)

---

 Tech Stack

* Python
* OpenCV
* MediaPipe (FaceLandmarker API)
* NumPy
* PyAutoGUI

---

 How It Works

1. The system captures real-time video using the webcam.
2. MediaPipe extracts **facial landmarks** from each frame.
3. The system computes:

   * Head orientation (yaw, pitch, roll)
   * Eye Aspect Ratio (EAR) for blink detection
4. A **rule-based engine** evaluates behavior and flags suspicious activity:

   * Looking away for extended duration
   * Eyes closed for abnormal duration
   * Multiple faces in frame
   * Switching active window
5. Events are:

   * Displayed on screen
   * Logged with timestamps
   * Saved periodically as screenshots

---

Run the Project

1. Install dependencies

```bash
pip install -r requirements.txt
```

 2. Run the system

```bash
python main.py
```

Press **ESC** to exit the application.

---

Output

* 📹 Real-time video feed with detection overlays
* 📝 `proctor_log.csv` file with event logs
* 📸 Screenshots saved locally at intervals

---

Limitations

* Basic rule-based system (not trained ML model)
* Screen monitoring depends on OS permissions
* Lighting and camera quality may affect detection accuracy

---

Future Improvements

* Audio-based cheating detection
* Real-time alerts (email/notifications)
* Full session recording
* Web-based dashboard (Flask/React)
* ML-based behavior classification

---




