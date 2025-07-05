import gradio as gr
import cv2
import numpy as np
import dlib
import os
from PIL import Image

LANDMARKS="shape_predictor_68_face_landmarks.dat"
if not os.path.exists(LANDMARKS):
    print("DLIB model file not found")
det=dlib.get_frontal_face_detector()
pred=dlib.shape_predictor(LANDMARKS)

def face_swap(inp_img,trgt_img):
    if inp_img is None or trgt_img is None:
        return None
    inp_img=np.array(inp_img.convert('RGB'))
    inp_img=cv2.cvtColor(inp_img,cv2.COLOR_RGB2BGR)
    trgt_img=np.array(trgt_img.convert('RGB'))
    trgt_img=cv2.cvtColor(trgt_img,cv2.COLOR_RGB2BGR)
    inp_gray=cv2.cvtColor(inp_img,cv2.COLOR_BGR2GRAY)
    trgt_gray=cv2.cvtColor(trgt_img,cv2.COLOR_BGR2GRAY)

    inp_faces=det(inp_gray)
    trgt_faces=det(trgt_gray)

    if len(inp_faces)==0:
        return cv2.putText(inp_img.copy(),"No faces detected",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
    if len(trgt_faces)==0:
        return cv2.putText(trgt_img.copy(),"No face detected")
    live_face_rect=inp_faces[0]
    trgt_face_rect=trgt_faces[0]
    live_landmrks=pred(inp_gray,live_face_rect)
    live_points=np.array([[p.x,p.y] for p in live_landmrks.parts()])
    trgt_landmrks=pred(trgt_gray,trgt_face_rect)
    trgt_points=np.array([[p.x,p.y] for p in trgt_landmrks.parts()])

    trgt_mask=np.zeros(trgt_img.shape[:2],dtype=np.uint8)
    trgt_hull=cv2.convexHull(trgt_points)
    cv2.fillConvexPoly(trgt_mask,trgt_hull,255)
    out_img_bgr=inp_img.copy()
    try:
        M,_=cv2.estimateAffinePartial2D(trgt_points.astype(np.float32),live_points.astype(np.float32))
        warp_trgt_face=cv2.warpAffine(trgt_img,M,(inp_img.shape[1],inp_img.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_TRANSPARENT)
        warp_trgt_mask=cv2.warpAffine(trgt_mask,M,(inp_img.shape[1],inp_img.shape[0]),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)

        live_mask=np.zeros(inp_img.shape[:2],dtype=np.uint8)
        live_hull=cv2.convexHull(live_points)
        cv2.fillConvexPoly(live_mask,live_hull,255)
        comb_mask=live_mask & warp_trgt_mask
        nose_tip=live_points[30]
        center_face=(nose_tip[0],nose_tip[1])
        out_img_bgr=cv2.seamlessClone(warp_trgt_face,out_img_bgr,comb_mask,center_face,cv2.NORMAL_CLONE)
    
    except Exception as e:
        print(f"ERROR during face swap :{e}")
        return cv2.putText(inp_img.copy(),"FACE SWAP ERROR")
    out_img_bgr=cv2.cvtColor(out_img_bgr,cv2.COLOR_BGR2RGB)
    return Image.fromarray(out_img_bgr)


global_out_img=gr.Image(type="pil",label="swapped face result")

with gr.Blocks(title="Face swap app") as demo:
    gr.Markdown("Face swap App")
    gr.Markdown("Swap faces in real time webcam or uploaded images")
    with gr.Tab("Live webcam swap"):
        gr.Markdown("Live Face swap")
        gr.Markdown("Your web cam will be detected")
        live_webcam_inp=gr.Image(type="pil",label="Live webcam feed",sources=["webcam"],streaming=True)
        
        global_inp_trgt_upload=gr.Image(type="pil",label="Upload target face image")
        start_swap_btn = gr.Button("Start Face Swap")
        live_swap_output = gr.Image(type="pil", label="Live Swapped Face")
        start_swap_btn.click(
            fn=face_swap,
            inputs=[
                live_webcam_inp,
                global_inp_trgt_upload
            ],
            outputs=live_swap_output
        )
    with gr.Tab("Image upload swap"):
        gr.Markdown("Image to image swap")
        gr.Markdown("Upload your image")
        inp_img_upload=gr.Image(type="pil",label="Upload main image")
        global_inp_trgt_upload=gr.Image(type="pil",label="Upload target face image")

        upload_interface=gr.Interface(
            fn=face_swap,
            inputs=[
                inp_img_upload,
                global_inp_trgt_upload
            ],
            outputs=global_out_img,
            live=False,
            flagging_mode="never"
        )
if __name__=="__main__":
    demo.launch(share=False)
