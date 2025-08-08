import torch
import json

checkpoints = [
    r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\checkpoint_epoch_15.pth",
    r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\DETR\checkpoint_epoch_60.pth",
    r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\Eden\DETR\Background\final_checkpoint.pth"
]

for i, checkpoint_path in enumerate(checkpoints):
    try:
        print(f"\n{'='*60}")
        print(f"CHECKPOINT {i+1}: {checkpoint_path}")
        print(f"{'='*60}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("Keys:", list(checkpoint.keys()))
        
        if 'epoch' in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")
        if 'best_val_loss' in checkpoint:
            print(f"Best validation loss: {checkpoint['best_val_loss']}")
        if 'val_loss' in checkpoint:
            print(f"Validation loss: {checkpoint['val_loss']}")
        if 'train_loss' in checkpoint:
            print(f"Train loss: {checkpoint['train_loss']}")
        if 'loss' in checkpoint:
            print(f"Loss: {checkpoint['loss']}")
            
        # Check model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            if isinstance(state_dict, dict):
                if 'model_state_dict' in state_dict:
                    actual_state = state_dict['model_state_dict']
                else:
                    actual_state = state_dict
                    
                if 'class_labels_classifier.weight' in actual_state:
                    print(f"Classes: {actual_state['class_labels_classifier.weight'].shape[0]}")
                    
        elif 'class_labels_classifier.weight' in checkpoint:
            print(f"Classes: {checkpoint['class_labels_classifier.weight'].shape[0]}")
            
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")

print(f"\n{'='*60}")
print("CHECKPOINT ANALYSIS COMPLETE")
print(f"{'='*60}")