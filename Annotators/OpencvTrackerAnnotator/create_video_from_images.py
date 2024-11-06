import cv2
import os
from tqdm import tqdm  # For the progress bar

def create_video_from_images(image_folder, output_video, fps=5):  # Set fps to 5
    # Get the list of all image files in the folder, sorted by filename
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".jpg")]

    if not images:
        print("No images found in the folder.")
        return

    # Get the size of the first image to determine the video size
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4 file
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Loop through all images and write them to the video
    for image in tqdm(images, desc="Creating video", unit="frame"):
        img_path = os.path.join(image_folder, image)
        img = cv2.imread(img_path)
        video.write(img)  # Add the image frame to the video

    # Release the video writer object
    video.release()
    print(f"Video saved as {output_video}")

# Main function to call the video creation
if __name__ == "__main__":
    image_folder = "../DetrAnnotator/output/sorted_images/75_100/Annotated_Images/"  # Path to your folder containing images
    output_video = "./output/output_annotated_images_video.mp4"  # Name of the output video file
    fps = 5  # Set the playback speed to 5 frames per second

    create_video_from_images(image_folder, output_video, fps)
