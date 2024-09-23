import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image, ImageDraw, ImageFont
import json
import os
from tqdm.auto import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class SurgicalToolTestDataset(Dataset):
    def __init__(self, images_dir, annotations_file):
        self.images_dir = images_dir

        # Load COCO annotations
        with open(annotations_file, 'r') as f:
            self.coco = json.load(f)

        # Create a mapping from filename to image_id
        self.filename_to_id = {img['file_name']: img['id'] for img in self.coco['images']}

        # List of image filenames
        self.image_filenames = [img['file_name'] for img in self.coco['images']]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        image_id = self.filename_to_id[filename]
        image_path = os.path.join(self.images_dir, filename)
        image = Image.open(image_path).convert('RGB')

        return image, image_id  # Return image and image_id


def draw_boxes(image, boxes, labels, scores, category_mapping, score_threshold=0.5):
    """
    Draw bounding boxes on the image.

    Args:
        image (PIL.Image): The image to draw on.
        boxes (list): List of bounding boxes [x_min, y_min, width, height].
        labels (list): List of label IDs.
        scores (list): List of confidence scores.
        category_mapping (dict): Mapping from label IDs to category names.
        score_threshold (float): Minimum score to display the box.

    Returns:
        PIL.Image: Image with bounding boxes drawn.
    """
    draw = ImageDraw.Draw(image)
    try:
        # Attempt to use a truetype font
        font = ImageFont.truetype("arial.ttf", size=15)
    except IOError:
        # Fallback to the default font if arial.ttf is not available
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue
        x, y, w, h = box
        x_min, y_min, x_max, y_max = x, y, x + w, y + h
        # Draw rectangle
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
        # Get label name
        label_name = category_mapping.get(label, f"ID:{label}")
        text = f"{label_name}: {score:.2f}"

        # Calculate text size using textbbox
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Draw background rectangle for text
        text_background = [x_min, y_min - text_height, x_min + text_width, y_min]
        draw.rectangle(text_background, fill="red")

        # Draw text
        draw.text((x_min, y_min - text_height), text, fill="white", font=font)

    return image


def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    images_dir = 'E:/wspolne_labelowanie/test/'
    annotations_file = 'E:/wspolne_labelowanie/test/_annotations.coco.json'
    output_dir = 'E:/wspolne_labelowanie/test_predictions/'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load model and processor
    model = DetrForObjectDetection.from_pretrained('./surgical_tool_detector')
    processor = DetrImageProcessor.from_pretrained('./surgical_tool_detector')

    # Prepare dataset and dataloader
    test_dataset = SurgicalToolTestDataset(images_dir, annotations_file)

    def collate_fn(batch):
        images, image_ids = list(zip(*batch))
        encoding = processor(images=list(images), return_tensors="pt")
        pixel_values = encoding['pixel_values']
        # Get image sizes (width, height)
        image_sizes = [image.size for image in images]
        return {
            'pixel_values': pixel_values,
            'image_ids': image_ids,
            'image_sizes': image_sizes,
            'original_images': images  # Pass original images for visualization
        }

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Run inference and collect predictions
    model.to(device)
    all_predictions = []
    model.eval()

    # Load ground truth annotations
    coco_gt = COCO(annotations_file)

    # Create category mapping from COCO annotations
    category_mapping = {cat['id']: cat['name'] for cat in coco_gt.loadCats(coco_gt.getCatIds())}

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Running Inference'):
            pixel_values = batch['pixel_values'].to(device)
            image_ids = batch['image_ids']
            image_sizes = batch['image_sizes']
            original_images = batch['original_images']

            outputs = model(pixel_values=pixel_values)

            # Convert outputs to COCO format predictions
            # Note: PIL image size is (width, height), but target_sizes expects (height, width)
            target_sizes = [(h, w) for (w, h) in image_sizes]
            processed_outputs = processor.post_process_object_detection(
                outputs, threshold=0.5, target_sizes=target_sizes
            )

            for idx, (output, image_id, original_image) in enumerate(
                    zip(processed_outputs, image_ids, original_images)):
                boxes = output['boxes'].tolist()
                scores = output['scores'].tolist()
                labels = output['labels'].tolist()

                # Save predictions in COCO format
                for box, score, label in zip(boxes, scores, labels):
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min

                    prediction = {
                        'image_id': image_id,
                        'category_id': label,
                        'bbox': [x_min, y_min, width, height],
                        'score': score
                    }
                    all_predictions.append(prediction)

                # Draw bounding boxes on the image and save
                image_with_boxes = draw_boxes(
                    original_image.copy(),
                    boxes,
                    labels,
                    scores,
                    category_mapping,
                    score_threshold=0.5
                )

                # Retrieve the original image filename using image_id
                image_info = coco_gt.loadImgs(image_id)[0]
                image_filename = image_info['file_name']
                save_path = os.path.join(output_dir, f"predicted_{image_filename}")
                image_with_boxes.save(save_path)

    # Save predictions to a JSON file
    predictions_file = os.path.join(output_dir, 'predictions.json')
    with open(predictions_file, 'w') as f:
        json.dump(all_predictions, f)

    # Load predictions
    coco_dt = coco_gt.loadRes(predictions_file)

    # Initialize COCOeval
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Print mAP
    mAP = coco_eval.stats[0]  # Average Precision (AP) @[ IoU=0.50:0.95 | area=all | maxDets=100 ]
    print(f'mAP: {mAP:.4f}')


if __name__ == '__main__':
    main()
