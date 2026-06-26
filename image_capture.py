import cv2
import os
from datetime import datetime
from picamera2 import Picamera2
import time

# Change this to the name of the person you're photographing
PERSON_NAME = "elias"

def create_folder(name):
    dataset_folder = "dataset"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    person_folder = os.path.join(dataset_folder, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    return person_folder

def capture_photos(name):
    folder = create_folder(name)

    picam2 = Picamera2()
    # RGB888: true R,G,B output from Picamera2
    picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}))
    picam2.start()
    time.sleep(2)

    photo_count = 0
    print(f"Taking photos for {name}. Press SPACE to capture, 'q' to quit.")

    while True:
        frame = picam2.capture_array()  # RGB

        # Convert RGB → BGR for correct OpenCV display and saving
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('Capture', frame_bgr)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            photo_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.jpg"
            filepath = os.path.join(folder, filename)
            cv2.imwrite(filepath, frame_bgr)
            print(f"Photo {photo_count} saved: {filepath}")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    picam2.stop()
    print(f"Photo capture completed. {photo_count} photos saved for {name}.")

if __name__ == "__main__":
    capture_photos(PERSON_NAME)