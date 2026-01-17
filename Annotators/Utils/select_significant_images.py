#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to select significant images for DETR training based on YOLO annotations.
The script analyzes both visual features and annotation diversity to select a subset
of images that are most representative and diverse.
"""

import os
import glob
import shutil
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from PIL import Image
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import random

def parse_yolo_annotation(annotation_path):
    """Parse YOLO annotation file and return list of objects."""
    objects = []
    if os.path.exists(annotation_path):
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:  # class_id, x_center, y_center, width, height
                    obj = {
                        'class_id': int(parts[0]),
                        'x_center': float(parts[1]),
                        'y_center': float(parts[2]),
                        'width': float(parts[3]),
                        'height': float(parts[4])
                    }
                    objects.append(obj)
    return objects

def extract_annotation_features(objects):
    """Extract features from annotation data."""
    if not objects:
        return np.zeros(8)  # Return zeros if no objects
    
    # Count objects per class
    class_counts = {}
    for obj in objects:
        class_id = obj['class_id']
        if class_id not in class_counts:
            class_counts[class_id] = 0
        class_counts[class_id] += 1
    
    # Calculate spatial distribution
    x_positions = [obj['x_center'] for obj in objects]
    y_positions = [obj['y_center'] for obj in objects]
    widths = [obj['width'] for obj in objects]
    heights = [obj['height'] for obj in objects]
    
    # Basic statistics
    num_objects = len(objects)
    num_classes = len(class_counts)
    avg_size = np.mean([w * h for w, h in zip(widths, heights)]) if objects else 0
    size_var = np.var([w * h for w, h in zip(widths, heights)]) if len(objects) > 1 else 0
    
    # Spatial distribution
    x_spread = np.max(x_positions) - np.min(x_positions) if len(objects) > 1 else 0
    y_spread = np.max(y_positions) - np.min(y_positions) if len(objects) > 1 else 0
    
    # Return feature vector
    features = np.array([
        num_objects,
        num_classes,
        avg_size,
        size_var,
        x_spread,
        y_spread,
        np.mean(x_positions) if objects else 0.5,
        np.mean(y_positions) if objects else 0.5
    ])
    
    return features

def extract_image_features(image_path, method='histogram'):
    """Extract visual features from image."""
    try:
        if method == 'histogram':
            # Use color histograms as features
            image = cv2.imread(image_path)
            if image is None:
                return np.zeros(32*3)  # Return zeros if image can't be read
                
            # Convert to HSV for better color representation
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Calculate histograms for each channel
            hist_h = cv2.calcHist([image], [0], None, [32], [0, 180])
            hist_s = cv2.calcHist([image], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([image], [2], None, [32], [0, 256])
            
            # Normalize and flatten
            cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist_v, hist_v, 0, 1, cv2.NORM_MINMAX)
            
            # Combine features
            features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
            return features
            
        elif method == 'hog':
            # Use HOG features
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                return np.zeros(36*4*4)  # Return zeros if image can't be read
                
            # Resize image for consistency
            image = cv2.resize(image, (128, 128))
            
            # Calculate HOG features
            winSize = (128, 128)
            blockSize = (32, 32)
            blockStride = (16, 16)
            cellSize = (16, 16)
            nbins = 9
            
            hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
            features = hog.compute(image)
            return features.flatten()
        
        else:
            raise ValueError(f"Unknown feature extraction method: {method}")
            
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        # Return zeros with expected feature length
        if method == 'histogram':
            return np.zeros(32*3)
        elif method == 'hog':
            return np.zeros(36*4*4)
        else:
            return np.zeros(32*3)

def combine_features(image_features, annotation_features, image_weight=0.5):
    """Combine image and annotation features with specified weights."""
    # Normalize both feature sets
    if len(image_features) > 0 and np.max(image_features) != 0:
        image_features = image_features / np.linalg.norm(image_features)
    
    if len(annotation_features) > 0 and np.max(annotation_features) != 0:
        annotation_features = annotation_features / np.linalg.norm(annotation_features)
    
    # Apply weighting
    return np.concatenate([
        image_features * image_weight,
        annotation_features * (1 - image_weight)
    ])

def cluster_images(features, n_clusters):
    """Cluster the feature vectors using K-means."""
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    return cluster_labels

def select_representative_images(image_paths, features, cluster_labels, n_per_cluster=5):
    """Select representative images from each cluster."""
    selected_images = []
    cluster_centers = {}
    
    # Calculate mean feature vector for each cluster
    for label in np.unique(cluster_labels):
        indices = np.where(cluster_labels == label)[0]
        center = np.mean(features[indices], axis=0)
        cluster_centers[label] = center
    
    # Select images closest to cluster center
    for label in np.unique(cluster_labels):
        cluster_indices = np.where(cluster_labels == label)[0]
        center = cluster_centers[label]
        
        # Calculate distances to center
        distances = []
        for idx in cluster_indices:
            dist = np.linalg.norm(features[idx] - center)
            distances.append((idx, dist))
        
        # Sort by distance and select top n
        distances.sort(key=lambda x: x[1])
        selected_indices = [idx for idx, _ in distances[:n_per_cluster]]
        
        # Add selected images
        for idx in selected_indices:
            selected_images.append(image_paths[idx])
    
    return selected_images

def plot_clusters(features, cluster_labels, output_path):
    """Plot 2D visualization of clusters using PCA."""
    from sklearn.decomposition import PCA
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    
    # Plot clusters
    plt.figure(figsize=(10, 8))
    
    for label in np.unique(cluster_labels):
        indices = np.where(cluster_labels == label)[0]
        plt.scatter(
            reduced_features[indices, 0], 
            reduced_features[indices, 1], 
            label=f'Cluster {label}',
            alpha=0.7
        )
    
    plt.title('Image Clusters Visualization')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()

def copy_selected_images(selected_images, output_dir, copy_annotations=True):
    """Copy selected images and their annotations to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    for image_path in selected_images:
        # Copy image
        filename = os.path.basename(image_path)
        shutil.copy2(image_path, os.path.join(output_dir, filename))
        
        # Copy annotation if needed
        if copy_annotations:
            annotation_path = os.path.splitext(image_path)[0] + '.txt'
            annotation_filename = os.path.basename(annotation_path)
            
            if os.path.exists(annotation_path):
                shutil.copy2(annotation_path, os.path.join(output_dir, annotation_filename))

def main():
    parser = argparse.ArgumentParser(description='Select significant images for DETR training based on YOLO annotations')
    parser.add_argument('--images_dir', required=True, help='Directory with source images')
    parser.add_argument('--labels_dir', help='Directory with YOLO label files (if different from images_dir)')
    parser.add_argument('--output_dir', required=True, help='Output directory for selected images')
    parser.add_argument('--num_clusters', type=int, default=20, help='Number of clusters to create')
    parser.add_argument('--images_per_cluster', type=int, default=5, help='Number of images to select per cluster')
    parser.add_argument('--feature_method', choices=['histogram', 'hog'], default='histogram', 
                        help='Method for extracting image features')
    parser.add_argument('--image_weight', type=float, default=0.5, 
                        help='Weight for image features vs annotation features (0-1)')
    parser.add_argument('--visualization', action='store_true', help='Generate visualization of clusters')
    args = parser.parse_args()
    
    # If labels directory not provided, use same as images
    if not args.labels_dir:
        args.labels_dir = args.images_dir
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all images
    image_extensions = ['jpg', 'jpeg', 'png']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.images_dir, f'*.{ext}')))
    
    print(f"Found {len(image_paths)} images in {args.images_dir}")
    
    if not image_paths:
        print("No images found. Exiting.")
        return
    
    # Extract features from images and annotations
    all_features = []
    valid_image_paths = []
    
    for i, image_path in enumerate(image_paths):
        if i % 100 == 0:
            print(f"Processing image {i}/{len(image_paths)}")
        
        # Get annotation path
        image_basename = os.path.splitext(os.path.basename(image_path))[0]
        annotation_path = os.path.join(args.labels_dir, f"{image_basename}.txt")
        
        # Extract features
        try:
            image_features = extract_image_features(image_path, method=args.feature_method)
            annotation_objects = parse_yolo_annotation(annotation_path)
            annotation_features = extract_annotation_features(annotation_objects)
            
            # Combine features
            combined_features = combine_features(image_features, annotation_features, args.image_weight)
            
            all_features.append(combined_features)
            valid_image_paths.append(image_path)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Convert to numpy array
    all_features = np.array(all_features)
    
    print(f"Successfully processed {len(valid_image_paths)} images")
    
    # Apply clustering
    n_clusters = min(args.num_clusters, len(valid_image_paths))
    print(f"Clustering images into {n_clusters} clusters...")
    cluster_labels = cluster_images(all_features, n_clusters)
    
    # Select representative images
    print("Selecting representative images...")
    selected_images = select_representative_images(
        valid_image_paths, 
        all_features, 
        cluster_labels, 
        args.images_per_cluster
    )
    
    print(f"Selected {len(selected_images)} images")
    
    # Generate visualization if requested
    if args.visualization:
        print("Generating cluster visualization...")
        visualization_path = os.path.join(args.output_dir, 'cluster_visualization.png')
        plot_clusters(all_features, cluster_labels, visualization_path)
    
    # Copy selected images and annotations
    print(f"Copying selected images to {args.output_dir}...")
    copy_selected_images(selected_images, args.output_dir)
    
    print("Done!")

if __name__ == "__main__":
    main()
