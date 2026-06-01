import cv2
import face_recognition
import numpy as np
import os
from threading import Thread
from picamera2 import Picamera2
from queue import Queue
import faiss
from datetime import datetime
import csv

class VideoStream:
    """Threaded video stream using Picamera2 on Raspberry Pi 5"""
    def __init__(self):
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"format": "BGR888", "size": (640, 480)}
        )
        self.picam2.configure(config)
        self.picam2.start()

        self.stopped = False
        self.queue = Queue(maxsize=2)

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.queue.full():
                frame = self.picam2.capture_array()
                print("Captured frame", frame.shape)
                self.queue.put(frame)

    def read(self):
        if self.queue.empty():
            return None
        return self.queue.get()

    def stop(self):
        self.stopped = True
        self.picam2.stop()
        self.picam2.close()


class AttendanceSystem:
    def __init__(self, attendance_file="attendance.csv"):
        self.attendance_file = attendance_file
        self.attendance_records = {}
        self.marked_today = set()

        # Create attendance file if it doesn't exist
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Date', 'Time'])

        # Load today's attendance
        self.load_todays_attendance()

    def load_todays_attendance(self):
        """Load attendance records from today"""
        today = datetime.now().strftime("%Y-%m-%d")

        if os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['Date'] == today:
                        self.marked_today.add(row['Name'])

    def mark_attendance(self, name):
        """Mark attendance for a person"""
        if name == "Unknown":
            return False

        if name in self.marked_today:
            return False

        # Mark attendance
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        with open(self.attendance_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, date_str, time_str])

        self.marked_today.add(name)
        print(f"✓ Attendance marked for {name} at {time_str}")
        return True

    def is_marked(self, name):
        """Check if person's attendance is already marked today"""
        return name in self.marked_today

    def get_attendance_count(self):
        """Get count of people who marked attendance today"""
        return len(self.marked_today)

def run_face_recognition_attendance():
    """
    Runs webcam face recognition with attendance tracking.
    """

    known_face_encodings = []
    known_face_names = []

    # ---------------------------------------------------------
    # 1. Load images from the 'faces' folder
    # ---------------------------------------------------------
    images_dir = "facesm2"

    if not os.path.exists(images_dir):
        print(f"Error: Directory '{images_dir}' not found.")
        print(f"Please create a folder named '{images_dir}' and add photos of people named 'Name.jpg'.")
        os.makedirs(images_dir)
        print(f"Created empty folder '{images_dir}' for you.")
        return

    print("Loading known faces...")

    for filename in os.listdir(images_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filepath = os.path.join(images_dir, filename)

            # Load the image
            image = face_recognition.load_image_file(filepath)

            # Get the encoding using HOG model
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])

                # Use the filename (without extension) as the person's name
                name = os.path.splitext(filename)[0]
                known_face_names.append(name)
                print(f"Loaded: {name}")
            else:
                print(f"Warning: No face found in {filename}")

    if not known_face_encodings:
        print("No valid faces found in 'faces' folder. Exiting.")
        return

    # Convert to numpy array for FAISS indexing
    known_face_encodings_array = np.array(known_face_encodings).astype('float32')

    # ---------------------------------------------------------
    # 2. Build FAISS Index for Fast Similarity Search
    # ---------------------------------------------------------
    print("\nBuilding FAISS index...")
    dimension = 128  # face_recognition encoding dimension

    # Create FAISS index (using L2 distance)
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(known_face_encodings_array)

    print(f"FAISS index built with {faiss_index.ntotal} faces")

    # Distance threshold (relaxed for better recognition)
    distance_threshold = 0.40

    # ---------------------------------------------------------
    # 3. Initialize Attendance System
    # ---------------------------------------------------------
    attendance = AttendanceSystem()
    print(f"\nAttendance system initialized")
    print(f"Already marked today: {attendance.get_attendance_count()} people")

    # ---------------------------------------------------------
    # 4. Initialize Threaded Webcam
    # ---------------------------------------------------------
    print("\nInitializing webcam...")
    video_stream = VideoStream().start()

    print("Webcam started. Press 'q' to quit, 's' to save snapshot, 'r' to reset today's attendance")

    # Initialize variables
    face_locations = []
    face_encodings = []
    face_names = []

    # Optimization parameters
    frame_count = 0
    process_frequency = 3  # Process every 3rd frame
    scale_factor = 0.25  # Scale to 1/4 size for better recognition

    while True:
        # Grab frame from threaded stream
        frame = video_stream.read()
        if frame is None:
            continue
        frame_count += 1

        # Only process every Nth frame
        if frame_count % process_frequency == 0:
            # ---------------------------------------------------------
            # 5. Optimization: Downscaling
            # ---------------------------------------------------------
            small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

            # Convert BGR to RGB
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # ---------------------------------------------------------
            # 6. Detect Faces (using HOG model - faster on CPU)
            # ---------------------------------------------------------
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []

            # ---------------------------------------------------------
            # 7. FAISS Nearest Neighbor Search
            # ---------------------------------------------------------
            for face_encoding in face_encodings:
                # Convert to float32 for FAISS
                query_encoding = np.array([face_encoding]).astype('float32')

                # Search for 1 nearest neighbor
                distances, indices = faiss_index.search(query_encoding, k=1)

                # Check if distance is below threshold
                if distances[0][0] < distance_threshold:
                    name = known_face_names[indices[0][0]]

                    # Mark attendance automatically when face is recognized
                    attendance.mark_attendance(name)
                else:
                    name = "Unknown"

                face_names.append(name)

        # ---------------------------------------------------------
        # 8. Draw Results with Attendance Status
        # ---------------------------------------------------------
        scale_multiplier = int(1 / scale_factor)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations
            top *= scale_multiplier
            right *= scale_multiplier
            bottom *= scale_multiplier
            left *= scale_multiplier

            # Set color based on attendance status
            if name == "Unknown":
                color = (0, 0, 255)  # Red for unknown
                status = ""
            elif attendance.is_marked(name):
                color = (0, 255, 0)  # Green for marked
                status = " ✓"
            else:
                color = (0, 165, 255)  # Orange for not marked yet
                status = ""

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw label with name and status
            label = f"{name}{status}"
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

        # Display info panel
        info_y = 30
        cv2.putText(frame, f"Registered: {len(known_face_names)} | Attendance Today: {attendance.get_attendance_count()}",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, "Press 'q' to quit | 's' for snapshot | 'r' to reset",
                   (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Face Recognition Attendance', frame)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save snapshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Snapshot saved: {filename}")
        elif key == ord('r'):
            # Reset today's attendance (for testing)
            response = input("\nAre you sure you want to reset today's attendance? (yes/no): ")
            if response.lower() == 'yes':
                attendance.marked_today.clear()
                print("Attendance reset for today")

    # Clean up
    video_stream.stop()
    cv2.destroyAllWindows()

    print(f"\nFinal attendance count: {attendance.get_attendance_count()}")
    print(f"Attendance saved to: {attendance.attendance_file}")

if __name__ == "__main__":
    run_face_recognition_attendance()