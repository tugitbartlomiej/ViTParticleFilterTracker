import argparse
import json
import os
import random

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def visualize_coco_annotations(images_dir, annotations_file, output_dir, num_samples=None, draw_labels=True,
                               color="green"):
    """
    Visualize COCO annotations on images and save them to output directory.

    Args:
        images_dir (str): Directory containing images
        annotations_file (str): Path to COCO annotations JSON file
        output_dir (str): Directory where annotated images will be saved
        num_samples (int, optional): Number of random images to sample. If None, process all images.
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

    # Get list of all image ids with annotations
    image_ids_with_annotations = list(annotations_by_image.keys())

    # Sample random images if num_samples is specified
    if num_samples is not None and num_samples < len(image_ids_with_annotations):
        print(f"Randomly sampling {num_samples} images from {len(image_ids_with_annotations)} available")
        image_ids_with_annotations = random.sample(image_ids_with_annotations, num_samples)
    else:
        print(f"Processing all {len(image_ids_with_annotations)} images with annotations")

    # Process each image
    processed_count = 0
    error_count = 0
    print(f"Processing images from: {images_dir}")

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

            # Get annotations for this image and filter out duplicates
            image_annotations = annotations_by_image[image_id]

            # Filter annotations to remove near-duplicates
            filtered_annotations = []
            for ann in image_annotations:
                bbox1 = ann['bbox']
                is_duplicate = False

                # Check if this annotation is a duplicate of one we've already filtered
                for filtered_ann in filtered_annotations:
                    bbox2 = filtered_ann['bbox']

                    # Convert COCO format [x, y, width, height] to corners [x1, y1, x2, y2]
                    x1_1, y1_1 = bbox1[0], bbox1[1]
                    x2_1, y2_1 = bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]

                    x1_2, y1_2 = bbox2[0], bbox2[1]
                    x2_2, y2_2 = bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]

                    # Calculate area of each box
                    area1 = bbox1[2] * bbox1[3]
                    area2 = bbox2[2] * bbox2[3]

                    # Calculate intersection
                    x1_i = max(x1_1, x1_2)
                    y1_i = max(y1_1, y1_2)
                    x2_i = min(x2_1, x2_2)
                    y2_i = min(y2_1, y2_2)

                    # Check if there is intersection
                    if x2_i <= x1_i or y2_i <= y1_i:
                        continue

                    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
                    union_area = area1 + area2 - intersection_area

                    # Calculate IoU
                    iou = intersection_area / union_area if union_area > 0 else 0

                    # Consider as duplicate if IoU is high
                    if iou > 0.7:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    filtered_annotations.append(ann)

            # Draw the bounding boxes
            for ann in filtered_annotations:
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

            # Debug annotation info
            draw.text(
                (10, 10),
                f"Image ID: {image_id}, Annotations: {len(filtered_annotations)} (filtered from {len(image_annotations)})",
                fill="blue",
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


def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize COCO annotations on images')
    parser.add_argument('--images_dir',
                        default="F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/augmented_dataset/images",
                        help='Directory containing images')
    parser.add_argument('--annotations_file',
                        default="F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/augmented_dataset/augmented_coco_14655-14660_20250417_011402.json",
                        help='Path to COCO annotations JSON file')
    parser.add_argument('--output_dir',
                        default="./visualized_annotations",
                        help='Directory where annotated images will be saved')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of random images to sample. If not specified, process all images.')
    parser.add_argument('--draw_labels', action='store_true', default=True,
                        help='Whether to draw class labels on the bounding boxes')
    parser.add_argument('--color', default='green',
                        help='Color of the bounding boxes (e.g., "red", "green", "blue")')
    return parser.parse_args()



def main():
    args = parse_arguments()

    visualize_coco_annotations(
        args.images_dir,
        args.annotations_file,
        args.output_dir,
        num_samples=args.num_samples,
        draw_labels=args.draw_labels,
        color=args.color
    )


if __name__ == "__main__":
    main()