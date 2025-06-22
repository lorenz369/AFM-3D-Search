import cv2
import os

def extract_frames(video_path, out_dir, step=10):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    saved_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % step == 0:
            cv2.imwrite(os.path.join(out_dir, f"frame_{saved_id:04d}.png"), frame)
            saved_id += 1
        frame_id += 1
    cap.release()

video_path = "data/videos/AFM_Video_Marco_1.mp4"
out_dir = "data/frames/AFM_Video_Marco_1/"
extract_frames(video_path, out_dir, step=1)

