import json
import os

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def visualize_coco_annotations(images_dir, annotations_file, output_dir, draw_labels=True, color="green"):
    """
    Visualize COCO annotations on images and save them to output directory.

    Args:
        images_dir (str): Directory containing images
        annotations_file (str): Path to COCO annotations JSON file
        output_dir (str): Directory where annotated images will be saved
        draw_labels (bool): Whether to draw class labels on the bounding boxes
        color (str): Color of the bounding boxes (e.g., "red", "green", "blue")
    """
    print(f"Loading annotations from: {annotations_file}")
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create mappings for easier access
    image_id_to_filename = {}
    for image in coco_data['images']:
        image_id_to_filename[image['id']] = image['file_name']

    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    # Create a mapping from category_id to category name
    categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
    if not categories:
        # If there are no categories, use a default category
        categories = {0: "surgical_tool"}

    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        print("Warning: Arial font not found, using default font")
        font = ImageFont.load_default()

    # Process each image
    processed_count = 0
    error_count = 0
    print(f"Processing images from: {images_dir}")

    # Get list of all image ids with annotations
    image_ids_with_annotations = list(annotations_by_image.keys())

    # Use tqdm to show a progress bar
    for image_id in tqdm(image_ids_with_annotations, desc="Visualizing annotations"):
        filename = image_id_to_filename.get(image_id)
        if not filename:
            print(f"Warning: No filename found for image_id {image_id}")
            continue

        image_path = os.path.join(images_dir, filename)
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            error_count += 1
            continue

        try:
            # Open the image
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)

            # Draw the bounding boxes
            for ann in annotations_by_image[image_id]:
                # COCO bbox format is [x, y, width, height]
                bbox = ann['bbox']
                x, y, width, height = bbox

                # Draw rectangle
                draw.rectangle(
                    [(x, y), (x + width, y + height)],
                    outline=color,
                    width=3
                )

                # Draw label if requested
                if draw_labels:
                    category_id = ann.get('category_id', 0)
                    category_name = categories.get(category_id, f"Category {category_id}")
                    label = f"{category_name}: {ann.get('score', 1.0):.2f}" if 'score' in ann else category_name

                    # Draw label background
                    text_width, text_height = draw.textbbox((0, 0), label, font=font)[2:]
                    draw.rectangle(
                        [(x, y - text_height - 4), (x + text_width + 4, y)],
                        fill=color
                    )

                    # Draw label text
                    draw.text(
                        (x + 2, y - text_height - 2),
                        label,
                        fill="white",
                        font=font
                    )

            # Save the image with annotations
            output_path = os.path.join(output_dir, filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image.save(output_path)
            processed_count += 1

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            error_count += 1

    print(f"Processing complete. Processed {processed_count} images. Errors: {error_count}")
    print(f"Annotated images saved to: {output_dir}")


def main():
    # Hardcoded paths
    images_dir = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/augmented_dataset/images"
    annotations_file = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/augmented_dataset/augmented_coco_200-300_20250325_022605.json"
    output_dir = "./visualized_annotations"


    # Configuration
    color = "green"  # Change to "red", "blue", etc. if needed
    draw_labels = True

    visualize_coco_annotations(
        images_dir,
        annotations_file,
        output_dir,
        draw_labels=draw_labels,
        color=color
    )


if __name__ == "__main__":
    main()