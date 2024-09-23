import cv2
import numpy as np
import json
import os

class ToolTipAnnotator:
    def __init__(self, video_path, output_file):
        self.video = cv2.VideoCapture(video_path)
        self.output_file = output_file
        self.annotations = []
        self.current_frame = 0

    def annotate(self):
        while True:
            ret, frame = self.video.read()
            if not ret:
                break

            cv2.imshow('Frame', frame)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('a'):  # Annotate
                cv2.setMouseCallback('Frame', self.click_event)
            elif key == ord('n'):  # Next frame
                self.current_frame += 1
            elif key == ord('q'):  # Quit
                break

        self.save_annotations()

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.annotations.append({
                'frame': self.current_frame,
                'x': x,
                'y': y
            })
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Frame', frame)

    def save_annotations(self):
        with open(self.output_file, 'w') as f:
            json.dump(self.annotations, f)

# Użycie:
annotator = ToolTipAnnotator('path/to/cataracts_video.mp4', 'annotations.json')
annotator.annotate()
