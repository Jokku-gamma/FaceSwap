import cv2
import numpy as np
import dlib

TARGET = "trump.jpeg"
LANDMARKS = "shape_predictor_68_face_landmarks.dat"

det = dlib.get_frontal_face_detector()
pred = dlib.shape_predictor(LANDMARKS)

trget_img = cv2.imread(TARGET)
if trget_img is None:
    print(f"Error: could not load target image")
    exit()

trget_gray = cv2.cvtColor(trget_img, cv2.COLOR_BGR2GRAY)
trget_faces = det(trget_gray)

if len(trget_faces) == 0:
    print("Error: No faces detected")
    exit()

trget_landmrks = pred(trget_gray, trget_faces[0])
trget_points = np.array([[p.x, p.y] for p in trget_landmrks.parts()])
trget_mask = np.zeros(trget_img.shape[:2], dtype=np.uint8)

trget_hull = cv2.convexHull(trget_points)
cv2.fillConvexPoly(trget_mask, trget_hull, 255)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error! could not open webcam")
    exit()

print("Webcam initialized")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    frm_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    live_faces = det(frm_gray)
    out_frm = frame.copy()

    for face in live_faces:
        live_landmarks = pred(frm_gray, face)
        live_points = np.array([[p.x, p.y] for p in live_landmarks.parts()])

        try:
            M, _ = cv2.estimateAffinePartial2D(trget_points.astype(np.float32), live_points.astype(np.float32))
            warped_trget_face = cv2.warpAffine(trget_img, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            warped_trget_mask = cv2.warpAffine(trget_mask, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

            live_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            live_hull = cv2.convexHull(live_points)
            cv2.fillConvexPoly(live_mask, live_hull, 255)

            comb_mask = live_mask & warped_trget_mask
            nose_tip = live_points[30]
            center_face = (nose_tip[0], nose_tip[1])

            out_frm = cv2.seamlessClone(warped_trget_face, out_frm, comb_mask, center_face, cv2.NORMAL_CLONE)

        except Exception as e:
            pass

    cv2.imshow('Live Face swap', out_frm)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
