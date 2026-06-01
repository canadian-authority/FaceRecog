import cv2
import dlib
import pickle
import numpy as np
import time
import csv
import os
from datetime import datetime
from collections import deque

# --- IMPORT PICAMERA2 ---
from picamera2 import Picamera2

# --- CONFIGURATION ---
ENCODINGS_FILE = "encodings.pickle"
RECOGNITION_THRESHOLD = 0.5 
PROCESS_EVERY_N_FRAMES = 4  
FRAME_WIDTH = 500           
SCREENSHOT_DIR = "attendance_screenshots"

# --- CONSISTENCY CHECKS ---
HISTORY_SIZE = 30
REQUIRED_HITS = 15          

if not os.path.exists(SCREENSHOT_DIR):
    os.makedirs(SCREENSHOT_DIR)

# Model Paths
SHAPE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
FACE_RECOGNITION_MODEL = "dlib_face_recognition_resnet_model_v1.dat"

present_students = set()

def mark_attendance(name):
    if name in present_students:
        return False

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    filename = f"attendance_{date_str}.csv"
    time_str = now.strftime("%H:%M:%S")

    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Time", "Date"])
        writer.writerow([name, time_str, date_str])

    present_students.add(name)
    print(f"[ATTENDANCE] Verified and Marked {name} at {time_str}")
    return True

def load_resources():
    print("[INFO] Loading encodings and models...")
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)

    print("[INFO] Using HOG Detector (CPU optimized)...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR)
    facerec = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL)

    return data, detector, predictor, facerec

def run_inference():
    data, detector, predictor, facerec = load_resources()
    known_encodings = np.array(data["encodings"])
    known_names = data["names"]

    print("[INFO] Initializing Picamera2...")

    # --- Picamera2 Setup ---
    picam2 = Picamera2()
    
    # Configure camera for video:
    # 1. Main stream size: 640x480
    # 2. Format: BGR888 (Native OpenCV format)
    config = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "BGR888"}
    )
    picam2.configure(config)
    picam2.start()
    print("[INFO] Video stream started via Picamera2.")

    frame_count = 0
    face_locations = []
    face_names = []
    recognition_history = deque(maxlen=HISTORY_SIZE)
    current_verification_stats = {}

    fps_start_time = time.time()
    fps_frame_counter = 0
    fps_display = 0

    try:
        while True:
            # Capture Frame
            frame = picam2.capture_array()
            
            if frame is None:
                print("[ERROR] Frame capture failed.")
                break

            # Resize for faster processing
            h, w = frame.shape[:2]
            ratio = FRAME_WIDTH / float(w)
            new_h = int(h * ratio)
            frame_resized = cv2.resize(frame, (FRAME_WIDTH, new_h))
            
            # Convert to RGB for dlib (dlib expects RGB, OpenCV gives BGR)
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            newly_marked_in_this_frame = []

            # Skip frames logic
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                face_locations = []
                face_names = []
                names_in_current_frame = set()

                rectangles = detector(rgb_frame, 0)

                for rect in rectangles:
                    left, top, right, bottom = rect.left(), rect.top(), rect.right(), rect.bottom()
                    face_locations.append((left, top, right, bottom))

                    shape = predictor(rgb_frame, rect)
                    face_encoding = np.array(facerec.compute_face_descriptor(rgb_frame, shape))

                    distances = np.linalg.norm(known_encodings - face_encoding, axis=1)
                    min_distance_idx = np.argmin(distances)
                    min_distance = distances[min_distance_idx]

                    if min_distance < RECOGNITION_THRESHOLD:
                        name = known_names[min_distance_idx]
                        names_in_current_frame.add(name)
                    else:
                        name = "Unknown"

                    face_names.append(name)

                recognition_history.append(names_in_current_frame)
                current_verification_stats = {}

                for name in names_in_current_frame:
                    if name == "Unknown": continue
                    occurrence_count = sum(1 for frame_set in recognition_history if name in frame_set)
                    current_verification_stats[name] = occurrence_count

                    if occurrence_count >= REQUIRED_HITS:
                        if mark_attendance(name):
                            newly_marked_in_this_frame.append(name)

            frame_count += 1

            # FPS Calculation
            fps_frame_counter += 1
            if (time.time() - fps_start_time) > 1:
                fps_display = fps_frame_counter / (time.time() - fps_start_time)
                fps_frame_counter = 0
                fps_start_time = time.time()

            # Drawing
            for (left, top, right, bottom), name in zip(face_locations, face_names):
                scale = 1 / ratio
                left, top, right, bottom = int(left * scale), int(top * scale), int(right * scale), int(bottom * scale)

                if name == "Unknown":
                    color = (0, 0, 255) # Red in BGR
                    label = name
                elif name in present_students:
                    color = (0, 255, 0) # Green in BGR
                    label = f"{name} (Present)"
                else:
                    color = (0, 255, 255) # Yellow in BGR
                    hits = current_verification_stats.get(name, 0)
                    label = f"{name} {hits}/{REQUIRED_HITS}"

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, label, (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1)

            # Snapshot (Saves BGR frame - usually preferred for disk)
            if newly_marked_in_this_frame:
                for name in newly_marked_in_this_frame:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{name}_{timestamp}.jpg"
                    filepath = os.path.join(SCREENSHOT_DIR, filename)
                    cv2.imwrite(filepath, frame)
                    print(f"[SNAPSHOT] Saved: {filepath}")

            cv2.putText(frame, f"FPS: {fps_display:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # --- CHANGE START: Convert to RGB strictly for display ---
            # We create a new variable so we don't mess up 'frame' for the next loop/saving
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('Face Recognition (Picamera2)', display_frame)
            # --- CHANGE END ---

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
    finally:
        # Stop and close the camera gracefully
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if os.path.exists(ENCODINGS_FILE):
        run_inference()
    else:
        print(f"[ERROR] {ENCODINGS_FILE} not found.")