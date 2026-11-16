from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = Path(__file__).resolve().parent / "assets" / "hand_landmarker.task"


def create_hand_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_hands(landmarker, frame, timestamp_ms):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect_for_video(image, timestamp_ms)


def draw_hand_landmarks(frame, landmarks):
    height, width = frame.shape[:2]

    def point(index):
        landmark = landmarks[index]
        return int(landmark.x * width), int(landmark.y * height)

    for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(
            frame,
            point(connection.start),
            point(connection.end),
            (0, 255, 0),
            2,
        )

    for landmark in landmarks:
        cv2.circle(
            frame,
            (int(landmark.x * width), int(landmark.y * height)),
            3,
            (0, 0, 255),
            -1,
        )
