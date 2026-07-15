import cv2
import pyvirtualcam
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import argparse
from transcriber import Transcriber

def main():
    parser = argparse.ArgumentParser(description="Virtual Camera with Overlay")
    parser.add_argument('flip', nargs='?', type=int, default=0, choices=[0, 1], help="Pass 1 to flip the camera horizontally, 0 to keep it normal.")
    args = parser.parse_args()

    # 1. Initialize physical webcam
    print("Opening physical webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open physical webcam.")
        return

    # Try to set high resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    if fps == 0:
        fps = 30 # Fallback FPS

    print(f"Webcam opened: {width}x{height} @ {fps}fps")

    # Start the background transcriber
    transcriber = Transcriber()
    transcriber.start()

    # 2. Open the virtual camera
    try:
        # We explicitly request 'UnityCapture' backend as discussed. 
        with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR, backend="unitycapture", device="Unity Video Capture") as cam:
            print(f'Virtual camera started: {cam.device} ({width}x{height} @ {fps}fps)')
            print("Press Ctrl+C to stop.")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error reading frame from webcam.")
                    break

                # 3. Add the Dummy overlay using PIL
                # Convert OpenCV BGR image to PIL RGB image
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                
                # Load a font (we try a standard Windows emoji font if possible, else default)
                try:
                    font = ImageFont.truetype("seguiemj.ttf", 64)
                except IOError:
                    font = ImageFont.load_default()

                # Draw a placeholder text and emoji
                text = "Dummy Virtual Camera 🌙"
                
                # Simple text positioning (bottom left)
                text_position = (20, height - 100)
                
                # Draw black background for text visibility
                bbox = draw.textbbox(text_position, text, font=font)
                draw.rectangle([bbox[0]-10, bbox[1]-10, bbox[2]+10, bbox[3]+10], fill=(0, 0, 0, 128))
                
                # Draw text
                draw.text(text_position, text, font=font, fill=(255, 255, 255), embedded_color=True)
                
                # Convert back to OpenCV BGR frame
                frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                # Flip the frame horizontally if requested (AFTER text overlay)
                if args.flip == 1:
                    frame = cv2.flip(frame, 1)

                # 4. Send to virtual camera
                cam.send(frame)
                cam.sleep_until_next_frame()

    except Exception as e:
        print(f"Failed to start virtual camera: {e}")
        print("Make sure you have run setup_driver.bat as Administrator to install the UnityCapture driver.")
    finally:
        transcriber.stop()
        cap.release()

if __name__ == "__main__":
    main()
