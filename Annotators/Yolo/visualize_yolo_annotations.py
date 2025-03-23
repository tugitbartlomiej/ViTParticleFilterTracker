import glob
import os

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def visualize_yolo_annotations(images_dir, labels_dir, output_dir, color="green", draw_labels=True):
    """
    Visualize YOLO format annotations on images and save them to output directory.

    Args:
        images_dir (str): Directory containing images
        labels_dir (str): Directory containing YOLO labels (txt files)
        output_dir (str): Directory where annotated images will be saved
        color (str): Color of the bounding boxes (e.g., "red", "green", "blue")
        draw_labels (bool): Whether to draw class labels on the bounding boxes
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, f'**/*{ext}'), recursive=True))
        image_files.extend(glob.glob(os.path.join(images_dir, f'**/*{ext.upper()}'), recursive=True))

    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))

    print(f"Found {len(image_files)} images in {images_dir}")

    # Define class names (modify this according to your classes)
    class_names = {0: "surgical_tool"}  # Default for surgical tool detection

    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        print("Warning: Arial font not found, using default font")
        font = ImageFont.load_default()

    # Process each image
    processed_count = 0
    error_count = 0

    for image_path in tqdm(image_files, desc="Visualizing annotations"):
        try:
            # Open the image
            image = Image.open(image_path).convert("RGB")
            img_width, img_height = image.size

            # Get corresponding label file
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(labels_dir, f"{base_name}.txt")

            # Check if label file exists
            if not os.path.exists(label_path):
                print(f"Warning: No label file found for {image_path}")
                # Save the original image to output directory
                output_path = os.path.join(output_dir, os.path.basename(image_path))
                image.save(output_path)
                continue

            # Read annotations from file
            draw = ImageDraw.Draw(image)

            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"Warning: Invalid annotation format in {label_path}: {line}")
                        continue

                    try:
                        class_id = int(parts[0])
                        # YOLO format is: class_id, x_center, y_center, width, height (normalized)
                        x_center = float(parts[1]) * img_width
                        y_center = float(parts[2]) * img_height
                        width = float(parts[3]) * img_width
                        height = float(parts[4]) * img_height

                        # Calculate bounding box coordinates
                        x1 = max(0, x_center - width / 2)
                        y1 = max(0, y_center - height / 2)
                        x2 = min(img_width, x_center + width / 2)
                        y2 = min(img_height, y_center + height / 2)

                        # Draw rectangle
                        draw.rectangle(
                            [(x1, y1), (x2, y2)],
                            outline=color,
                            width=3
                        )

                        # Draw label if requested
                        if draw_labels:
                            class_name = class_names.get(class_id, f"Class {class_id}")
                            label = class_name

                            # Draw label background
                            text_width, text_height = draw.textbbox((0, 0), label, font=font)[2:]
                            draw.rectangle(
                                [(x1, y1 - text_height - 4), (x1 + text_width + 4, y1)],
                                fill=color
                            )

                            # Draw label text
                            draw.text(
                                (x1 + 2, y1 - text_height - 2),
                                label,
                                fill="white",
                                font=font
                            )
                    except ValueError as e:
                        print(f"Error parsing annotation in {label_path}: {line}, {e}")
                        continue

            # Create subdirectories in output if needed
            rel_path = os.path.relpath(image_path, images_dir)
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save the image with annotations
            image.save(output_path)
            processed_count += 1

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            error_count += 1

    print(f"Processing complete. Processed {processed_count} images. Errors: {error_count}")
    print(f"Annotated images saved to: {output_dir}")


def main():
    # Hardcoded paths
    images_dir = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DeepSortYolo/ProcessedVideos/yolo_dataset_20250218/images/train"
    labels_dir = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DeepSortYolo/ProcessedVideos/yolo_dataset_20250218/labels/train"
    output_dir = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DeepSortYolo/ProcessedVideos/visualized_yolo"

    # Configuration
    color = "green"  # Change to "red", "blue", etc. if needed
    draw_labels = True

    visualize_yolo_annotations(
        images_dir,
        labels_dir,
        output_dir,
        color=color,
        draw_labels=draw_labels
    )


if __name__ == "__main__":
    main()