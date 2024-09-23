# utils/tool_tip_data_processor.py

import cv2
import os
from pycocotools.coco import COCO
from PIL import Image, ImageDraw

class ToolTipDataProcessor:
    def __init__(self, video_path, json_path, output_dir='extracted_frames', debug_dir='debug'):
        self.video_path = video_path
        self.json_path = json_path
        self.output_dir = output_dir
        self.debug_dir = debug_dir

        # Sprawdź, czy katalog na klatki i katalog debug istnieją, jeśli nie, utwórz je
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)

    def extract_frames(self):
        """Ekstrakcja klatek z pliku wideo"""
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_filename = os.path.join(self.output_dir, f"frame_{frame_count:06d}.jpg")  # Użyj formatu .jpg
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

        cap.release()
        print(f"Ekstrakcja zakończona: {frame_count} klatek zapisanych do {self.output_dir}.")

    def load_annotations(self):
        """Załaduj adnotacje COCO"""
        self.coco = COCO(self.json_path)
        print(f"Załadowano adnotacje z {self.json_path}.")

    def load_image(self, img_id):
        """Załaduj obraz na podstawie ID"""
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.output_dir, img_info['file_name'])
        image = Image.open(img_path)
        return image, img_info['file_name']

    def draw_bounding_boxes(self, img_id, image):
        """Rysowanie bounding boxów na obrazie"""
        # Pobierz wszystkie adnotacje dla danego obrazu
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Rysuj bounding boxy
        draw = ImageDraw.Draw(image)
        for ann in anns:
            bbox = ann['bbox']  # COCO format: [x, y, width, height]
            x, y, w, h = bbox
            draw.rectangle([x, y, x + w, y + h], outline="red", width=3)

        return image

    def show_sample_images_with_boxes(self, num_images=5):
        """Wyświetl i zapisz przykładowe obrazy z narysowanymi ramkami"""
        img_ids = self.coco.getImgIds()
        for i, img_id in enumerate(img_ids):
            if i >= num_images:
                break
            image, filename = self.load_image(img_id)
            image_with_boxes = self.draw_bounding_boxes(img_id, image)
            image_with_boxes.show()  # Wyświetl obraz z ramkami
            image_with_boxes.save(f"{self.debug_dir}/boxed_{filename}")  # Zapisz obraz z ramkami do folderu debug
