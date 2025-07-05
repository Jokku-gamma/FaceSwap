import cv2
import numpy as np
import dlib

TARGET="trump1.jpeg"
LANDMARKS="shape_predictor_68_face_landmarks.dat"
det=dlib.get_frontal_face_detector()
pred=dlib.shape_predictor(LANDMARKS)
trget_img=cv2.imread(TARGET)
if trget_img is None:
    print(f"Error:could not load target image")
    exit()

trget_gray=cv2.cvtColor(trget_img,cv2.COLOR_BGR2GRAY)
trget_faces=det(trget_gray)
if len(trget_faces)==0:
    print("Error.No faces detecetd")
    exit()

trget_face_rect=trget_faces[0]
trget_landmrks=pred(trget_gray,trget_face_rect)
trget_points=np.array([[p.x,p.y] for p in trget_landmrks.parts()])

x,y,w,h=trget_face_rect.left(),trget_face_rect.top(),trget_face_rect.width(),trget_face_rect.height()
expand_x=int(w*0.2)
expand_y_top=int(h*0.5)
expand_y_bottom=int(h*0.1)

x1=max(0,x-expand_x)
y1=max(0,y-expand_y_top)
x2=min(trget_img.shape[1],x+w+expand_x)
y2=min(trget_img.shape[0],y+h+expand_y_bottom)


trget_mask=np.zeros(trget_img.shape[:2],dtype=np.uint8)
cv2.rectangle(trget_mask,(x1,y1),(x2,y2),255,-1)

cap=cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error! could not open webcam")
    exit()
print("Webcam initialized")
while True:
    ret,frame=cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    frame=cv2.flip(frame,1)
    frm_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    live_faces=det(frm_gray)
    out_frm=frame.copy()
    for face in live_faces:
        live_landmarks=pred(frm_gray,face)
        live_points=np.array([[p.x,p.y] for p in live_landmarks.parts()])
        lx,ly,lw,lh=face.left(),face.top(),face.width(),face.height()
        live_expand_x=int(lw*0.2)
        live_expand_y_top=int(lh*0.5)
        live_expand_y_bottom=int(lh*0.1)
        lx1=max(0,lx-live_expand_x)
        ly1=max(0,ly-live_expand_y_top)
        lx2=min(frame.shape[1],lx+lw+live_expand_x)
        ly2=min(frame.shape[0],ly+lh+live_expand_y_bottom)

        live_mask=np.zeros(frame.shape[:2],dtype=np.uint8)
        cv2.rectangle(live_mask,(lx1,ly1),(lx2,ly2),255,-1)
        
        try:
            M,_=cv2.estimateAffinePartial2D(trget_points.astype(np.float32),live_points.astype(np.float32))
            warped_trget_face=cv2.warpAffine(trget_img,M,(frame.shape[1],frame.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
            warped_trget_mask=cv2.warpAffine(trget_mask,M,(frame.shape[1],frame.shape[0]),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
            
            comb_mask=live_mask & warped_trget_mask
            nose_tip=live_points[30]
            center_face=(nose_tip[0],nose_tip[1])
            out_frm=cv2.seamlessClone(warped_trget_face,out_frm,comb_mask,center_face,cv2.NORMAL_CLONE)

        except Exception as e:
            pass
        
    cv2.imshow('Live Face swap',out_frm)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()