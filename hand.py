import cv2
import time

from gesture.hand_tracking import (
    create_hand_landmarker,
    detect_hands,
    draw_hand_landmarks,
)

hands = create_hand_landmarker()

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = detect_hands(
        hands,
        frame,
        int(time.monotonic() * 1000)
    )

    # detect them hands
    if results.hand_landmarks:

        for hand_landmarks in results.hand_landmarks:

            # wrist landmark
            wrist = hand_landmarks[0]

            y = wrist.y

            # determine hand position
            if y < 0.5:

                text = "HAND UP"
                color = (0, 255, 0)

            else:

                text = "HAND DOWN"
                color = (0, 0, 255)

            # display text
            cv2.putText(
                frame,
                text,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

            # draw hand skeleton
            draw_hand_landmarks(frame, hand_landmarks)

    # display frame
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()

cv2.destroyAllWindows()
