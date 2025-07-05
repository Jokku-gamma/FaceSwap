import gradio as gr
import cv2
import numpy as np
import dlib
import os
from PIL import Image

# --- Configuration ---
# Path to dlib's facial landmark predictor model
LANDMARKS_MODEL_PATH = "shape_predictor_68_face_landmarks.dat"

# Check if the dlib model file exists
if not os.path.exists(LANDMARKS_MODEL_PATH):
    # For deployment, ensure this file is available in the same directory or accessible path.
    # On Render, you'd typically place it in your repository.
    print(f"Error: Dlib model file not found at {LANDMARKS_MODEL_PATH}. "
          "Please ensure it's in the same directory as app.py for deployment.")
    # In a real deployed app, you might raise an exception or log prominently.

# Initialize dlib's models
# Use a global variable or cache if not in a web framework
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(LANDMARKS_MODEL_PATH)

# --- Face Swap Function ---
# This function will be called by Gradio.
# input_image: The main image (from webcam or upload)
# target_face_image: The image of the face to swap in (from upload)
def perform_face_swap(input_image_pil, target_face_image_pil):
    if input_image_pil is None or target_face_image_pil is None:
        return None # Gradio will display nothing if None is returned

    # Convert PIL Image to OpenCV format (NumPy array)
    # Gradio passes PIL Images for 'image' type inputs
    input_cv_img = np.array(input_image_pil.convert('RGB'))
    input_cv_img = cv2.cvtColor(input_cv_img, cv2.COLOR_RGB2BGR) # Convert RGB to BGR for OpenCV

    target_cv_img = np.array(target_face_image_pil.convert('RGB'))
    target_cv_img = cv2.cvtColor(target_cv_img, cv2.COLOR_RGB2BGR) # Convert RGB to BGR for OpenCV

    input_gray = cv2.cvtColor(input_cv_img, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_cv_img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    input_faces = detector(input_gray)
    target_faces = detector(target_gray)

    if len(input_faces) == 0:
        # Gradio can display messages with exceptions, or you can return an image with text.
        # For this demo, let's return a blank image or the original with a message.
        return cv2.putText(input_cv_img.copy(), "No face detected in main input!", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    if len(target_faces) == 0:
        return cv2.putText(input_cv_img.copy(), "No face detected in target image!", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Assuming first detected face for simplicity
    live_face_rect = input_faces[0]
    target_face_rect = target_faces[0]

    # Get landmarks for both faces
    live_landmarks = predictor(input_gray, live_face_rect)
    live_points = np.array([[p.x, p.y] for p in live_landmarks.parts()])

    target_landmarks = predictor(target_gray, target_face_rect)
    target_points = np.array([[p.x, p.y] for p in target_landmarks.parts()])

    # Create mask for the target face
    target_mask = np.zeros(target_cv_img.shape[:2], dtype=np.uint8)
    target_hull = cv2.convexHull(target_points)
    cv2.fillConvexPoly(target_mask, target_hull, 255)

    # Make a copy of the input_cv_img to draw on
    output_img_bgr = input_cv_img.copy()

    # Perform affine transformation
    try:
        M, _ = cv2.estimateAffinePartial2D(target_points.astype(np.float32), live_points.astype(np.float32))
        warped_target_face = cv2.warpAffine(target_cv_img, M, (input_cv_img.shape[1], input_cv_img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        warped_target_mask = cv2.warpAffine(target_mask, M, (input_cv_img.shape[1], input_cv_img.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

        # Create mask for the live face in the input image
        live_mask = np.zeros(input_cv_img.shape[:2], dtype=np.uint8)
        live_hull = cv2.convexHull(live_points)
        cv2.fillConvexPoly(live_mask, live_hull, 255)

        combined_mask = live_mask & warped_target_mask

        # Find the center of the live face for seamlessClone (using nose tip)
        nose_tip = live_points[30]
        center_face = (nose_tip[0], nose_tip[1])

        # Seamless Blending
        output_img_bgr = cv2.seamlessClone(warped_target_face, output_img_bgr, combined_mask, center_face, cv2.NORMAL_CLONE)

    except Exception as e:
        # If swap fails, return the original image with an error message
        print(f"Error during face swap: {e}")
        return cv2.putText(input_cv_img.copy(), "Face swap error!", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Convert BGR back to RGB for Gradio display
    output_img_rgb = cv2.cvtColor(output_img_bgr, cv2.COLOR_BGR2RGB)
    return output_img_rgb

# --- Gradio Interface ---
# Define inputs
input_main_image_upload = gr.Image(type="pil", label="Upload Main Image (Your Photo)", sources=["upload"])
input_main_image_webcam = gr.Image(type="pil", label="Or Use Webcam (Your Live Feed)", sources=["webcam"])
input_target_face_upload = gr.Image(type="pil", label="Upload Target Face Image (Face to Swap In)", sources=["upload"])

# Combine inputs into a list for the interface
# We'll handle the choice between upload/webcam for the main image with a function
# For simplicity, let's just make two separate interfaces or one with clear instructions
# The best way to do this with Gradio is to allow both upload and webcam for the "main" input,
# and let the user choose. Gradio's `sources` parameter handles this automatically.

# Define output
output_image = gr.Image(type="pil", label="Swapped Face Result")

# Create the Gradio Interface
# Note: For real-time webcam processing, set live=True.
# However, this means it will run on every frame. If your processing is slow,
# it might lag. For a basic swap, it might be acceptable.
# If `live=True`, the target_face_image will be fixed from its initial upload.

# Option 1: Separate Interface for "Live" (Webcam only for main input)
live_interface = gr.Interface(
    fn=perform_face_swap,
    inputs=[
        gr.Image(type="pil", label="Your Live Webcam Feed", sources=["webcam"], streaming=True), # Use streaming=True for continuous frames
        input_target_face_upload # Target face is still an upload
    ],
    outputs=output_image,
    title="Joel's Live Face Swap (Gradio)",
    description="Swap your live webcam face with an uploaded target face! (Upload target first)",
    live=True # Enable real-time processing
)

# Option 2: Image Upload Interface
upload_interface = gr.Interface(
    fn=perform_face_swap,
    inputs=[
        input_main_image_upload,
        input_target_face_upload
    ],
    outputs=output_image,
    title="Joel's Image Face Swap (Gradio)",
    description="Upload a main image and a target face image to perform a swap.",
    live=False # Not live for static images
)

# Combine into a Tabbed Interface for better UX
demo = gr.TabbedInterface(
    [live_interface, upload_interface],
    ["Live Webcam Swap", "Image Upload Swap"]
)

# Launch the app
if __name__ == "__main__":
    # Ensure the dlib model path is correctly set.
    # This will run the Gradio app locally.
    demo.launch(share=False) 