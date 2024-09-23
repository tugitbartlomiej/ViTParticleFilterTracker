import cv2
import numpy as np
import torch

def evaluate_model(model, test_loader, video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    video = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (int(video.get(3)), int(video.get(4))))
    
    with torch.no_grad():
        for frames, true_positions in test_loader:
            frames, true_positions = frames.to(device), true_positions.to(device)
            predicted_positions = model(frames)
            
            for i in range(frames.size(1)):
                video.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = video.read()
                if not ret:
                    break
                
                # Draw true position
                cv2.circle(frame, (int(true_positions[0, i, 0]), int(true_positions[0, i, 1])), 5, (0, 255, 0), -1)
                
                # Draw predicted position
                cv2.circle(frame, (int(predicted_positions[0, i, 0]), int(predicted_positions[0, i, 1])), 5, (0, 0, 255), -1)
                
                out.write(frame)
    
    video.release()
    out.release()

# Użycie:
model = ToolTipTracker()
model.load_state_dict(torch.load('trained_model.pth'))
evaluate_model(model, test_loader, 'path/to/test_video.mp4')
