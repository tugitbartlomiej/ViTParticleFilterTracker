import torch

# Load checkpoint to check structure
checkpoint_path = r"F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\models\DETR\detr_inference_model.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("Checkpoint keys:", list(checkpoint.keys()))
print()

# Print additional info from checkpoint
if 'args' in checkpoint:
    print("Training args:", checkpoint['args'])
if 'epoch' in checkpoint:
    print("Epoch:", checkpoint['epoch'])
if 'num_classes' in checkpoint:
    print("Num classes:", checkpoint['num_classes'])
print()

# This seems to be a nested checkpoint
if 'model_state_dict' in checkpoint:
    model_state = checkpoint['model_state_dict']
    print("Model state dict keys count:", len(list(model_state.keys())))
    
    # Check if this has another nested level
    if 'model_state_dict' in model_state:
        actual_state = model_state['model_state_dict']
        print("Actual state dict keys count:", len(list(actual_state.keys())))
        
        if 'class_labels_classifier.weight' in actual_state:
            print("class_labels_classifier.weight shape:", actual_state['class_labels_classifier.weight'].shape)
            print("class_labels_classifier.bias shape:", actual_state['class_labels_classifier.bias'].shape)
    else:
        if 'class_labels_classifier.weight' in model_state:
            print("class_labels_classifier.weight shape:", model_state['class_labels_classifier.weight'].shape) 
            print("class_labels_classifier.bias shape:", model_state['class_labels_classifier.bias'].shape)
else:
    # Direct state dict
    if 'class_labels_classifier.weight' in checkpoint:
        print("class_labels_classifier.weight shape:", checkpoint['class_labels_classifier.weight'].shape)
        print("class_labels_classifier.bias shape:", checkpoint['class_labels_classifier.bias'].shape)