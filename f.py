import cv2
import mediapipe as mp
import time
from mediapipe.python.solutions.drawing_utils import DrawingSpec

from mediapipe.python.solutions.face_mesh import FaceMesh

cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
# parameters for drawing marks , higher recomanded for HD videos
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS,
                                  drawSpec, drawSpec)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow("Image", img)
    # cv2.waitKey(1)

    key = cv2.waitKey(1)

    # segment for press q or Q to quit the app
    if key == 81 or key == 113:
        break
