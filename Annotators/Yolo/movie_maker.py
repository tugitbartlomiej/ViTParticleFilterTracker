import os

import cv2

frames_dir = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Annotators\Yolo\output_frames_80_100\annotated"
output_video = "merged_video.mp4"
image_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".png"))])
if not image_files:
    raise ValueError("No image files found in the specified directory.")

first_image = cv2.imread(os.path.join(frames_dir, image_files[0]))
height, width, _ = first_image.shape
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 5
writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

for file_name in image_files:
    path = os.path.join(frames_dir, file_name)
    frame = cv2.imread(path)
    if frame is None:
        continue
    cv2.putText(frame, file_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    writer.write(frame)

writer.release()
cv2.destroyAllWindows()
print("Video saved:", output_video)
