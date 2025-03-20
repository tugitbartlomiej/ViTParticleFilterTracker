import os

import cv2


def merge_frames_to_video(frames_dir: str, output_path: str, fps: int = 5) -> None:
    image_files = sorted(
        f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".png"))
    )
    if not image_files:
        raise ValueError("No image files found in the specified directory.")

    first_image = cv2.imread(os.path.join(frames_dir, image_files[0]))
    height, width, _ = first_image.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for file_name in image_files:
        path = os.path.join(frames_dir, file_name)
        frame = cv2.imread(path)
        if frame is not None:
            cv2.putText(
                frame,
                file_name,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            writer.write(frame)

    writer.release()
    cv2.destroyAllWindows()
    print("Video saved:", output_path)

if __name__ == "__main__":
    frames_directory = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Annotators\Hybrid\output_hybrid_yolo_tracker_pf_hardreset\annotated"
    output_video_path = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Annotators\Hybrid\output_hybrid_yolo_tracker_pf_hardreset\merged_video_hybrid_annotated.mp4"
    merge_frames_to_video(frames_directory, output_video_path, fps=30)
