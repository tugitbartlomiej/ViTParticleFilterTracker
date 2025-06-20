import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import pickle

class DINOFeatureExtractor:
    """
    DINO (Self-Supervised Vision Transformer) Feature Extractor
    
    Uses pre-trained DINO model to extract rich visual features from images.
    DINO is particularly good at capturing semantic information and spatial relationships,
    making it ideal for selecting informative frames for DETR training.
    """
    
    def __init__(self, model_name='dino_vits16', patch_size=16, device='auto'):
        """
        Initialize DINO feature extractor
        
        Args:
            model_name: DINO model variant ('dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8')
            patch_size: Patch size for vision transformer
            device: Device to run model on ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.patch_size = patch_size
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load pre-trained DINO model
        self.model = self._load_dino_model()
        self.model.eval()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"DINO model {model_name} loaded successfully")
    
    def _load_dino_model(self):
        """Load pre-trained DINO model"""
        try:
            # Try to load from torch hub
            model = torch.hub.load('facebookresearch/dino:main', self.model_name)
            model.to(self.device)
            return model
        except Exception as e:
            print(f"Error loading DINO model from torch hub: {e}")
            print("Trying alternative loading method...")
            
            # Alternative: Load using timm if available
            try:
                import timm
                model = timm.create_model('vit_small_patch16_224', pretrained=True)
                model.to(self.device)
                return model
            except ImportError:
                print("timm not available. Please install: pip install timm")
                raise
    
    def extract_features(self, image_path):
        """
        Extract DINO features from a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary containing various feature representations
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Get features from DINO model
                features = self.model(input_tensor)
                
                # Extract different types of features
                feature_dict = {
                    'global_features': features.cpu().numpy().flatten(),
                    'feature_dim': features.shape[1],
                    'image_path': str(image_path),
                    'image_name': Path(image_path).name
                }
                
                # Additional feature statistics
                feature_dict.update({
                    'feature_mean': float(features.mean().cpu()),
                    'feature_std': float(features.std().cpu()),
                    'feature_max': float(features.max().cpu()),
                    'feature_min': float(features.min().cpu()),
                    'feature_norm': float(torch.norm(features).cpu())
                })
                
                return feature_dict
                
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def extract_patch_features(self, image_path, return_attention=False):
        """
        Extract patch-level features and optionally attention maps
        
        Args:
            image_path: Path to image file
            return_attention: Whether to return attention maps
            
        Returns:
            Dictionary with patch features and optional attention maps
        """
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                if return_attention:
                    # Get attention maps (if model supports it)
                    try:
                        attentions = self.model.get_last_selfattention(input_tensor)
                        features = self.model(input_tensor)
                        
                        return {
                            'patch_features': features.cpu().numpy(),
                            'attention_maps': attentions.cpu().numpy(),
                            'image_path': str(image_path),
                            'image_name': Path(image_path).name
                        }
                    except:
                        # Fallback to regular features
                        features = self.model(input_tensor)
                        return {
                            'patch_features': features.cpu().numpy(),
                            'image_path': str(image_path),
                            'image_name': Path(image_path).name
                        }
                else:
                    features = self.model(input_tensor)
                    return {
                        'patch_features': features.cpu().numpy(),
                        'image_path': str(image_path),
                        'image_name': Path(image_path).name
                    }
                    
        except Exception as e:
            print(f"Error extracting patch features from {image_path}: {e}")
            return None
    
    def process_image_directory(self, image_dir, output_file, max_images=None):
        """
        Process all images in directory and save features
        
        Args:
            image_dir: Directory containing images
            output_file: Path to save features
            max_images: Maximum number of images to process (None for all)
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(image_dir).glob(f'*{ext}'))
            image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"Processing {len(image_files)} images from {image_dir}")
        
        features_list = []
        failed_images = []
        
        for image_path in tqdm(image_files, desc="Extracting DINO features"):
            features = self.extract_features(image_path)
            if features is not None:
                features_list.append(features)
            else:
                failed_images.append(str(image_path))
        
        # Save features
        output_data = {
            'features': features_list,
            'failed_images': failed_images,
            'model_info': {
                'model_name': self.model_name,
                'patch_size': self.patch_size,
                'device': str(self.device),
                'feature_dim': features_list[0]['feature_dim'] if features_list else 0
            },
            'processing_stats': {
                'total_images': len(image_files),
                'successful_extractions': len(features_list),
                'failed_extractions': len(failed_images)
            }
        }
        
        # Save as pickle for numpy arrays
        with open(output_file, 'wb') as f:
            pickle.dump(output_data, f)
        
        # Also save metadata as JSON
        json_file = output_file.replace('.pkl', '_metadata.json')
        json_data = {
            'model_info': output_data['model_info'],
            'processing_stats': output_data['processing_stats'],
            'failed_images': output_data['failed_images']
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Features saved to: {output_file}")
        print(f"Metadata saved to: {json_file}")
        print(f"Successfully processed: {len(features_list)}/{len(image_files)} images")
        
        return features_list
    
    def compare_features(self, features1, features2):
        """
        Compare two feature vectors using cosine similarity
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            Cosine similarity score
        """
        f1 = np.array(features1)
        f2 = np.array(features2)
        
        # Normalize
        f1 = f1 / np.linalg.norm(f1)
        f2 = f2 / np.linalg.norm(f2)
        
        # Cosine similarity
        similarity = np.dot(f1, f2)
        return float(similarity)

def main():
    parser = argparse.ArgumentParser(description='Extract DINO features from images')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--output_file', type=str, default='dino_features.pkl',
                        help='Output file for features')
    parser.add_argument('--model_name', type=str, default='dino_vits16',
                        choices=['dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8'],
                        help='DINO model variant')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to process')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to run model on')
    
    args = parser.parse_args()
    
    # Create feature extractor
    extractor = DINOFeatureExtractor(
        model_name=args.model_name,
        device=args.device
    )
    
    # Process images
    features = extractor.process_image_directory(
        args.input_dir,
        args.output_file,
        args.max_images
    )
    
    print(f"Feature extraction completed. {len(features)} features extracted.")

if __name__ == "__main__":
    main()