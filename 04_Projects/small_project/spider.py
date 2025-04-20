import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Define face and eye landmark indices
FACE_OUTLINE = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
    148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
NOSE_TIP = 1  # Nose tip for web lines origin

# Function to draw the Spider-Man mask
def draw_spiderman_mask(frame, landmarks):
    h, w, _ = frame.shape
    mask = np.zeros_like(frame, dtype=np.uint8)

    # Get face points
    face_points = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in FACE_OUTLINE], np.int32)

    # Draw the red base mask
    cv2.fillPoly(mask, [face_points], (0, 0, 255))  # Red

    # Draw white eye regions
    left_eye_points = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE], np.int32)
    right_eye_points = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE], np.int32)
    cv2.fillPoly(mask, [left_eye_points], (255, 255, 255))
    cv2.fillPoly(mask, [right_eye_points], (255, 255, 255))

    # Add black web lines
    nose = (int(landmarks[NOSE_TIP].x * w), int(landmarks[NOSE_TIP].y * h))

    # Radial web lines
    for i in range(0, 360, 30):  # Every 30 degrees
        angle = np.radians(i)
        end_x = int(nose[0] + 400 * np.cos(angle))
        end_y = int(nose[1] + 400 * np.sin(angle))
        cv2.line(mask, nose, (end_x, end_y), (0, 0, 0), 2)

    # Concentric web arcs
    for radius in range(50, 300, 50):
        cv2.ellipse(mask, nose, (radius, radius), 0, 0, 360, (0, 0, 0), 2)

    # Blend the Spider-Man mask with the frame
    spiderman_frame = cv2.addWeighted(frame, 0.5, mask, 0.5, 0)
    return spiderman_frame

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Flip frame for a mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw Spider-Man mask
            frame = draw_spiderman_mask(frame, face_landmarks.landmark)

    # Display the frame
    cv2.imshow("Spider-Man Mask Effect", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
