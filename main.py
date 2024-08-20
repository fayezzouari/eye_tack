import cv2
import mediapipe as mp
import time
import numpy as np

# Settings
maximum_time = 15  # Seconds

# Load Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection()
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Take frame from camera
cap = cv2.VideoCapture(0)

# Track TIME
starting_time = time.time()

def get_face_orientation(landmarks, width, height):
    left_eye = np.array([landmarks[33].x, landmarks[33].y])  # Left eye
    right_eye = np.array([landmarks[263].x, landmarks[263].y])  # Right eye
    nose_tip = np.array([landmarks[1].x, landmarks[1].y])  # Nose tip

    # Calculate the eye center and the nose direction vector
    eye_center = (left_eye + right_eye) / 2
    nose_vector = nose_tip - eye_center

    # Normalize by the width and height of the frame
    nose_vector[0] *= width
    nose_vector[1] *= height

    x_threshold = width * 0.02  # Adjust this threshold to tune sensitivity for left-right detection
    y_threshold = height * 0.02  # Adjust this threshold to tune sensitivity for up-down detection

    if nose_vector[0] < -x_threshold:
        return "Looking Right"
    elif nose_vector[0] > x_threshold:
        return "Looking Left"
    elif nose_vector[1] < -y_threshold:
        return "Looking Up"
    elif nose_vector[1] > y_threshold:
        return "Looking Down"
    else:
        return "Looking up"

while True:
    # Take frame from camera
    ret, frame = cap.read()
    height, width, channels = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw rectangle
    cv2.rectangle(frame, (0, 0), (width, 70), (10, 10, 10), -1)

    # Face Detection
    results_detection = face_detection.process(rgb_frame)
    results_mesh = face_mesh.process(rgb_frame)

    # Is the face DETECTED?
    if results_detection.detections:
        elapsed_time = int(time.time() - starting_time)

        if elapsed_time > maximum_time:
            # Reached maximum time, show alert
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 225), 10)

        # Draw elapsed time on screen
        cv2.putText(frame, "{} seconds".format(elapsed_time), (10, 50), cv2.FONT_HERSHEY_PLAIN,
                    3, (15, 225, 215), 2)

        # Face mesh landmarks detection
        if results_mesh.multi_face_landmarks:
            for face_landmarks in results_mesh.multi_face_landmarks:
                orientation = get_face_orientation(face_landmarks.landmark, width, height)
                cv2.putText(frame, orientation, (10, 100), cv2.FONT_HERSHEY_PLAIN,
                            3, (15, 225, 215), 2)

                # Optionally draw face mesh landmarks on the frame
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

        print("Face looking at the screen")
    else:
        print("NO FACE")
        # Reset the counter
        starting_time = time.time()

    # Display frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()