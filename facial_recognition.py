import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import pickle

print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

picam2 = Picamera2()
# RGB888: Picamera2 outputs true R,G,B order — we convert to BGR for OpenCV
picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (1920, 1080)}))
picam2.start()

cv_scaler = 4
DISTANCE_THRESHOLD = 0.45  # Lower = stricter match required
ABSENCE_THRESHOLD  = 1.0   # Seconds gap with no detection = face left the frame
COOLDOWN_SECONDS   = 10    # If absent >= 10s and returns = decrease counter

face_locations     = []
face_encodings_list = []
face_names         = []
frame_count        = 0
fps_start_time     = time.time()
fps                = 0

# Counter
presence_counter = 0

# last_seen_time[name] — timestamp of the last frame this person was detected
# Updated every frame they are visible; gap between frames reveals when they left
last_seen_time = {}



def process_frame(frame):
    """Detect faces, match names, update counter with correct logic."""
    global face_locations, face_encodings_list, face_names, presence_counter, last_seen_time

    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    face_locations = face_recognition.face_locations(resized_frame)
    face_encodings_list = face_recognition.face_encodings(
        resized_frame, face_locations, model='large'
    )

    face_names = []
    now = time.time()
    current_frame_seen_names = set()
    unknown_count_in_frame = 0

    # --- STEP 1: Detect and handle boardings ---
    for face_encoding in face_encodings_list:
        name = "Unknown"

        if len(known_face_encodings) > 0:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < DISTANCE_THRESHOLD:
                name = known_face_names[best_match_index]
                current_frame_seen_names.add(name)

                # If they are not currently marked as on board, they just boarded!
                if name not in last_seen_time:
                    print(f"[BOARDING] Registered passenger {name} entered.")
                
                # Update/Initialize their timestamp keeping them alive on board
                last_seen_time[name] = now

        if name == "Unknown":
            unknown_count_in_frame += 1

        face_names.append(name)

    # --- STEP 2: Independent check for exits (Known Passengers Only) ---
    passengers_who_left = []
    for tracked_name, last_time in last_seen_time.items():
        if tracked_name not in current_frame_seen_names:
            time_absent = now - last_time
            if time_absent >= COOLDOWN_SECONDS:
                passengers_who_left.append(tracked_name)

    for tracked_name in passengers_who_left:
        del last_seen_time[tracked_name]
        print(f"[EXIT] Registered passenger {tracked_name} left the vehicle.")

    # --- STEP 3: Dynamic Counter Computation ---
    # Total = tracked registered passengers + temporary unregistered faces in this frame
    presence_counter = len(last_seen_time) + unknown_count_in_frame

    return frame


def draw_results(frame):
    """Draw rectangles: GREEN for recognized, RED for unknown."""
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top    *= cv_scaler
        right  *= cv_scaler
        bottom *= cv_scaler
        left   *= cv_scaler

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
        cv2.rectangle(frame, (left - 3, top - 35), (right + 3, top), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)

    return frame


def draw_counter(frame):
    """Draw the presence counter in the top-left corner."""
    label = f"Passengers: {presence_counter}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(label, font, 1, 2)
    cv2.rectangle(frame, (5, 5), (text_w + 25, text_h + 25), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, label, (10, text_h + 15), font, 1, (0, 255, 0), 2)
    return frame


def calculate_fps():
    global frame_count, fps_start_time, fps
    frame_count += 1
    elapsed = time.time() - fps_start_time
    if elapsed > 1:
        fps = frame_count / elapsed
        frame_count = 0
        fps_start_time = time.time()
    return fps


while True:
    frame = picam2.capture_array()  # RGB888 — true RGB

    processed_frame = process_frame(frame)

    # Convert RGB → BGR for correct OpenCV display colors
    bgr_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

    display_frame = draw_results(bgr_frame)
    display_frame = draw_counter(display_frame)

    current_fps = calculate_fps()
    cv2.putText(display_frame, f"FPS: {current_fps:.1f}",
                (display_frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Video', display_frame)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
picam2.stop()