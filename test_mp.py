import mediapipe as mp
from gesture.hand_tracking import create_hand_landmarker

landmarker = create_hand_landmarker()
landmarker.close()

print(f"MediaPipe {mp.__version__}: hand landmarker ready")
