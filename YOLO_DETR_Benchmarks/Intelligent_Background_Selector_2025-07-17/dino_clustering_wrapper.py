#!/usr/bin/env python3
"""
DINO Clustering Wrapper for Intelligent Background Frame Selector
================================================================

This module provides a simplified interface to the existing DINO clustering
implementation, specifically adapted for the background frame selection task.
"""

import numpy as np
import os
import sys
from pathlib import Path
import pickle

# Add the DINO_Frame_Selection scripts to path
project_root = Path(__file__).parent.parent
dino_scripts_path = project_root / "DINO_Frame_Selection" / "scripts"
sys.path.insert(0, str(dino_scripts_path))

from dino_clustering import DINOClustering

class DINOClusterer:
    """
    Simplified wrapper for DINO clustering functionality
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the DINO clusterer
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.clustering = None
    
    def perform_clustering(self, features_array, n_clusters=20, method='kmeans'):
        """
        Perform clustering on features array
        
        Args:
            features_array: NumPy array of features
            n_clusters: Number of clusters
            method: Clustering method ('kmeans' or 'dbscan')
            
        Returns:
            Cluster labels array
        """
        # Create a temporary features structure compatible with DINOClustering
        temp_features_data = {
            'features': [
                {
                    'global_features': features_array[i],
                    'image_path': f'temp_image_{i}.jpg',
                    'image_name': f'temp_image_{i}.jpg',
                    'feature_norm': float(np.linalg.norm(features_array[i]))
                }
                for i in range(len(features_array))
            ],
            'model_info': {
                'feature_dim': features_array.shape[1],
                'model_name': 'dino_vits16'
            },
            'processing_stats': {
                'successful_extractions': len(features_array),
                'total_images': len(features_array)
            }
        }
        
        # Save temporary features file
        temp_features_file = 'temp_dino_features.pkl'
        with open(temp_features_file, 'wb') as f:
            pickle.dump(temp_features_data, f)
        
        try:
            # Initialize clustering
            self.clustering = DINOClustering(temp_features_file, random_state=self.random_state)
            
            # Perform clustering
            labels, cluster_info = self.clustering.perform_clustering(
                n_clusters=n_clusters, 
                algorithm=method
            )
            
            return labels
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_features_file):
                os.remove(temp_features_file)
    
    def select_representative_frames(self, image_paths, features_array, cluster_labels, 
                                   frames_per_cluster=5, selection_method='centroid'):
        """
        Select representative frames from clusters
        
        Args:
            image_paths: List of image paths
            features_array: NumPy array of features
            cluster_labels: Cluster labels array
            frames_per_cluster: Number of frames per cluster
            selection_method: Selection method ('centroid', 'diverse', 'quality')
            
        Returns:
            List of selected image paths
        """
        if self.clustering is None:
            raise ValueError("Must perform clustering first")
        
        # Update the clustering object with actual image paths
        self.clustering.image_paths = image_paths
        self.clustering.image_names = [Path(p).name for p in image_paths]
        self.clustering.labels = cluster_labels
        
        # Update features list with actual paths
        for i, path in enumerate(image_paths):
            if i < len(self.clustering.features_list):
                self.clustering.features_list[i]['image_path'] = path
                self.clustering.features_list[i]['image_name'] = Path(path).name
        
        # Select representative frames
        selected_frames = self.clustering.select_representative_frames(
            frames_per_cluster=frames_per_cluster,
            selection_method=selection_method
        )
        
        # Return just the image paths
        return [frame['image_path'] for frame in selected_frames]
    
    def get_cluster_statistics(self, cluster_labels):
        """
        Get statistics for cluster labels
        
        Args:
            cluster_labels: Array of cluster labels
            
        Returns:
            Dictionary with cluster statistics
        """
        unique_labels = np.unique(cluster_labels)
        if -1 in unique_labels:  # Remove noise label for DBSCAN
            unique_labels = unique_labels[unique_labels != -1]
        
        cluster_stats = {}
        for label in unique_labels:
            mask = cluster_labels == label
            cluster_stats[int(label)] = {
                'size': int(np.sum(mask)),
                'indices': np.where(mask)[0].tolist()
            }
        
        return cluster_stats
    
    def save_clustering_results(self, output_dir, image_paths, cluster_labels, selected_frames):
        """
        Save clustering results
        
        Args:
            output_dir: Output directory
            image_paths: List of image paths
            cluster_labels: Cluster labels
            selected_frames: List of selected frame paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create results dictionary
        results = {
            'cluster_info': {
                'n_clusters': len(np.unique(cluster_labels)),
                'algorithm': 'kmeans',
                'total_images': len(image_paths)
            },
            'cluster_stats': self.get_cluster_statistics(cluster_labels),
            'selected_frames': selected_frames,
            'all_images': image_paths
        }
        
        # Save as JSON
        import json
        results_file = os.path.join(output_dir, 'clustering_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Clustering results saved to {results_file}")
        
        return results

# For backward compatibility
class DINOClusteringWrapper(DINOClusterer):
    """Alias for backward compatibility"""
    pass