import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


devices = AudioUtilities.GetSpeakers()
interface = devices._dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get the volume range (e.g., -65.25 to 0.0)
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for start, end in HAND_CONNECTIONS:
        cv.line(frame, points[start], points[end], (0, 255, 0), 2)
    for point in points:
        cv.circle(frame, point, 5, (0, 0, 255), -1)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)


with vision.HandLandmarker.create_from_options(options) as landmarker:
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Unable to open camera")
    else:
        print("Camera opened successfully")

        try:
            while True:
                ret, frame = cap.read()

                if ret:
                    timestamp_ms = int(cap.get(cv.CAP_PROP_POS_MSEC))

                    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                    results = landmarker.detect_for_video(mp_image, timestamp_ms)

                    if results.hand_landmarks:
                        h, w, _ = frame.shape
                        for hand_landmarks in results.hand_landmarks:
                            draw_landmarks(frame, hand_landmarks)
                            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                            thumb_x, thumb_y = points[4]
                            finger_x, finger_y = points[8]
                            distance = math.hypot(thumb_x - finger_x, thumb_y - finger_y)
                            print(f"Distance between thumb and index finger: {distance:.2f} inches")

                            target_volume = np.interp(distance, [10, 300], [min_vol, max_vol])
                            volume.SetMasterVolumeLevel(target_volume, None)
                    cv.imshow("Frame", frame)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv.destroyAllWindows()


