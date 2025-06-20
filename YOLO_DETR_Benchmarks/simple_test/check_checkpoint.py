import torch

checkpoint_path = r'F:\Studia\PhD_projekt\VIT\ViTParticleFilterTracker\YOLO_DETR_Benchmarks\models\DETR\detr_inference_model.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("=== CHECKPOINT ANALYSIS ===")
print(f"Keys in checkpoint: {list(checkpoint.keys())}")
print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"Num classes: {checkpoint.get('num_classes', 'N/A')}")

# Check if state_dict is nested
if 'model_state_dict' in checkpoint:
    model_data = checkpoint['model_state_dict']
    print(f"\nType of model_state_dict: {type(model_data)}")
    
    # If it's a dict and has model_state_dict inside, it's nested
    if isinstance(model_data, dict) and 'model_state_dict' in model_data:
        print("WARNING: State dict is nested!")
        actual_state_dict = model_data['model_state_dict']
    else:
        actual_state_dict = model_data
    
    print(f"Number of parameters in model: {len(actual_state_dict)}")
    
    # Find classifier keys
    classifier_keys = [k for k in actual_state_dict.keys() if 'class' in k and 'classifier' in k]
    print(f"\nClassifier keys found: {len(classifier_keys)}")
    for key in classifier_keys[:5]:  # Show first 5
        print(f"  {key}: shape = {actual_state_dict[key].shape}")
    
    # Find query embeddings
    query_keys = [k for k in actual_state_dict.keys() if 'query' in k]
    print(f"\nQuery-related keys found: {len(query_keys)}")
    if query_keys:
        print(f"  Example: {query_keys[0]}: shape = {actual_state_dict[query_keys[0]].shape}")
    
    # Check for DETR-specific keys
    detr_keys = ['model.query_embed.weight', 'class_embed.weight', 'class_embed.bias']
    print(f"\nChecking DETR-specific keys:")
    for key in detr_keys:
        if key in actual_state_dict:
            print(f"  ✓ {key}: shape = {actual_state_dict[key].shape}")
        else:
            # Try with different prefixes
            found = False
            for prefix in ['', 'model.', 'detr.']:
                full_key = prefix + key
                if full_key in actual_state_dict:
                    print(f"  ✓ {full_key}: shape = {actual_state_dict[full_key].shape}")
                    found = True
                    break
            if not found:
                print(f"  ✗ {key}: NOT FOUND")

print("\n=== DONE ===")