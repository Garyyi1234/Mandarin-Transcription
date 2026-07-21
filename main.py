import cv2
import pyvirtualcam
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import argparse
from transcriber import Transcriber

def main():
    parser = argparse.ArgumentParser(description="Virtual Camera with Overlay")
    parser.add_argument('flip', nargs='?', type=int, default=0, choices=[0, 1], help="Pass 1 to flip the text horizontally, 0 to keep it normal.")
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

                # 3. Add the text overlay using PIL
                # Convert OpenCV BGR image to PIL RGB image
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Create a transparent overlay for the text
                overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                
                # Load a font (try Microsoft YaHei or SimHei for Mandarin support, fallback to default)
                try:
                    font = ImageFont.truetype("msyh.ttc", 64)
                except IOError:
                    try:
                        font = ImageFont.truetype("simhei.ttf", 64)
                    except IOError:
                        font = ImageFont.load_default()

                # Get transcriber text
                previous_text, current_text = transcriber.get_texts()
                
                margin_x = 20
                current_y = height - 120
                previous_y = current_y - 80
                
                # Draw previous text
                if previous_text:
                    bbox_prev = draw.textbbox((margin_x, previous_y), previous_text, font=font)
                    draw.rectangle([bbox_prev[0]-10, bbox_prev[1]-10, bbox_prev[2]+10, bbox_prev[3]+10], fill=(0, 0, 0, 128))
                    draw.text((margin_x, previous_y), previous_text, font=font, fill=(200, 200, 200))

                # Draw current text
                if current_text:
                    bbox_curr = draw.textbbox((margin_x, current_y), current_text, font=font)
                    draw.rectangle([bbox_curr[0]-10, bbox_curr[1]-10, bbox_curr[2]+10, bbox_curr[3]+10], fill=(0, 0, 0, 128))
                    draw.text((margin_x, current_y), current_text, font=font, fill=(255, 255, 255))
                
                # Mirror the text left-right if requested
                if args.flip == 1:
                    overlay = overlay.transpose(Image.FLIP_LEFT_RIGHT)
                
                # Paste the text overlay onto the main image
                img_pil.paste(overlay, (0, 0), overlay)
                
                # Convert back to OpenCV BGR frame
                frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                # 4. Send to virtual camera
                cam.send(frame)
                cam.sleep_until_next_frame()

    except KeyboardInterrupt:
        print("\nending...")
    except Exception as e:
        print(f"Failed to start virtual camera: {e}")
        print("Make sure you have run setup_driver.bat as Administrator to install the UnityCapture driver.")
    finally:
        transcriber.stop()
        cap.release()

if __name__ == "__main__":
    main()
