import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector

# Load king expressions
img_thumbs = cv2.imread("thumbs_up.png")
img_angry = cv2.imread("angry.png")
img_cry = cv2.imread("cry.png")
img_laugh = cv2.imread("laugh.png")

expressions = {
    "happy": img_thumbs,
    "angry": img_angry,
    "sad": img_cry,
    "laugh": img_laugh
}

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

current_expression = "happy"

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    frame, faces = detector.findFaceMesh(frame, draw=False)

    if faces:
        face = faces[0]

        # points
        left_eye = face[145][1]
        right_eye = face[374][1]
        mouth_up = face[13][1]
        mouth_down = face[14][1]

        mouth_open = mouth_down - mouth_up
        eye_open = abs(left_eye - right_eye)

        # SIMPLE RULES
        if mouth_open > 25:
            current_expression = "laugh"
        elif mouth_open > 15:
            current_expression = "happy"
        elif eye_open < 8:
            current_expression = "angry"
        else:
            current_expression = "sad"

    # Load corresponding king expression
    exp_img = expressions[current_expression]
    exp_img = cv2.resize(exp_img, (400, frame.shape[0]))

    # Combine webcam + king side by side
    frame_resized = cv2.resize(frame, (400, frame.shape[0]))
    final = np.hstack((frame_resized, exp_img))

    cv2.imshow("Expression Detector", final)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
