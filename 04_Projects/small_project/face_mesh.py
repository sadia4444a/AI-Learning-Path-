import cv2
import mediapipe as mp
import time

# Initialize variables
cTime = 0
pTime = 0

# Set up video capture and MediaPipe FaceMesh
cap = cv2.VideoCapture(0)
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
mpDraw = mp.solutions.drawing_utils

# Ensure FACE_CONNECTIONS is available
if hasattr(mpFaceMesh, 'FACEMESH_TESSELATION'):
    face_connections = mpFaceMesh.FACEMESH_TESSELATION
else:
    face_connections = mpFaceMesh.FACE_CONNECTIONS  # Try fallback to FACE_CONNECTIONS if available

while True:
    success, img = cap.read()
    if not success:
        break

    # Convert the image to RGB for MediaPipe processing
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    
    # Draw landmarks if faces are detected
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, face_connections)  # Use the correct connections
    
    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    
    # Show the video
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
