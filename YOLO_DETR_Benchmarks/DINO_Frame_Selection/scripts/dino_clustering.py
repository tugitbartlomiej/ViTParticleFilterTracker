import numpy as np
import pickle
import json
import os
import argparse
from pathlib import Path
import shutil
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

class DINOClustering:
    """
    DINO-based clustering for intelligent frame selection
    
    Uses DINO features to cluster images and select the most representative
    frames for DETR training. This approach leverages DINO's semantic understanding
    to identify diverse and informative training samples.
    """
    
    def __init__(self, features_file, random_state=42):
        """
        Initialize clustering with DINO features
        
        Args:
            features_file: Path to pickle file containing DINO features
            random_state: Random seed for reproducibility
        """
        self.features_file = features_file
        self.random_state = random_state
        
        # Load features
        print(f"Loading DINO features from {features_file}")
        with open(features_file, 'rb') as f:
            self.data = pickle.load(f)
        
        self.features_list = self.data['features']
        self.model_info = self.data['model_info']
        self.processing_stats = self.data['processing_stats']
        
        print(f"Loaded {len(self.features_list)} feature vectors")
        print(f"Feature dimension: {self.model_info['feature_dim']}")
        
        # Prepare feature matrix
        self.feature_matrix = np.array([f['global_features'] for f in self.features_list])
        self.image_paths = [f['image_path'] for f in self.features_list]
        self.image_names = [f['image_name'] for f in self.features_list]
        
        # Standardize features
        self.scaler = StandardScaler()
        self.feature_matrix_scaled = self.scaler.fit_transform(self.feature_matrix)
        
        print(f"Feature matrix shape: {self.feature_matrix_scaled.shape}")
    
    def find_optimal_clusters(self, k_range=(2, 20), method='elbow'):
        """
        Find optimal number of clusters using elbow method or silhouette analysis
        
        Args:
            k_range: Range of k values to test
            method: Method to use ('elbow', 'silhouette')
            
        Returns:
            Optimal number of clusters
        """
        k_values = range(k_range[0], k_range[1] + 1)
        
        if method == 'elbow':
            inertias = []
            for k in tqdm(k_values, desc="Testing k values"):
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                kmeans.fit(self.feature_matrix_scaled)
                inertias.append(kmeans.inertia_)
            
            # Find elbow point (simplified method)
            differences = np.diff(inertias)
            second_differences = np.diff(differences)
            optimal_k = k_values[np.argmax(second_differences) + 2]  # +2 due to double diff
            
            return optimal_k, {'k_values': list(k_values), 'inertias': inertias}
        
        elif method == 'silhouette':
            silhouette_scores = []
            for k in tqdm(k_values, desc="Testing k values"):
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = kmeans.fit_predict(self.feature_matrix_scaled)
                score = silhouette_score(self.feature_matrix_scaled, labels)
                silhouette_scores.append(score)
            
            optimal_k = k_values[np.argmax(silhouette_scores)]
            
            return optimal_k, {'k_values': list(k_values), 'silhouette_scores': silhouette_scores}
    
    def perform_clustering(self, n_clusters=20, algorithm='kmeans'):
        """
        Perform clustering on DINO features
        
        Args:
            n_clusters: Number of clusters
            algorithm: Clustering algorithm ('kmeans', 'dbscan')
            
        Returns:
            Cluster labels and additional information
        """
        print(f"Performing {algorithm} clustering with {n_clusters} clusters...")
        
        if algorithm == 'kmeans':
            self.clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10,
                max_iter=300
            )
            labels = self.clusterer.fit_predict(self.feature_matrix_scaled)
            
            # Calculate cluster centers and distances
            cluster_centers = self.clusterer.cluster_centers_
            distances = self.clusterer.transform(self.feature_matrix_scaled)
            
            cluster_info = {
                'algorithm': 'kmeans',
                'n_clusters': n_clusters,
                'cluster_centers': cluster_centers,
                'inertia': self.clusterer.inertia_,
                'n_iter': self.clusterer.n_iter_
            }
            
        elif algorithm == 'dbscan':
            self.clusterer = DBSCAN(
                eps=0.5,
                min_samples=5,
                metric='euclidean'
            )
            labels = self.clusterer.fit_predict(self.feature_matrix_scaled)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            distances = None
            
            cluster_info = {
                'algorithm': 'dbscan',
                'n_clusters': n_clusters,
                'n_noise': list(labels).count(-1),
                'eps': 0.5,
                'min_samples': 5
            }
        
        self.labels = labels
        self.cluster_info = cluster_info
        self.distances = distances
        
        # Calculate cluster statistics
        self.cluster_stats = self._calculate_cluster_statistics()
        
        print(f"Clustering completed. Found {len(set(labels))} clusters")
        return labels, cluster_info
    
    def _calculate_cluster_statistics(self):
        """Calculate statistics for each cluster"""
        unique_labels = set(self.labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Remove noise label for DBSCAN
        
        cluster_stats = {}
        for label in unique_labels:
            mask = self.labels == label
            cluster_features = self.feature_matrix_scaled[mask]
            
            stats = {
                'size': int(np.sum(mask)),
                'center': np.mean(cluster_features, axis=0),
                'std': np.std(cluster_features, axis=0),
                'variance': np.var(cluster_features, axis=0),
                'images': [self.image_names[i] for i in range(len(mask)) if mask[i]]
            }
            
            cluster_stats[int(label)] = stats
        
        return cluster_stats
    
    def select_representative_frames(self, frames_per_cluster=5, selection_method='centroid'):
        """
        Select representative frames from each cluster
        
        Args:
            frames_per_cluster: Number of frames to select per cluster
            selection_method: Method for selection ('centroid', 'diverse', 'quality')
            
        Returns:
            List of selected frames with metadata
        """
        print(f"Selecting {frames_per_cluster} representative frames per cluster...")
        
        selected_frames = []
        
        for cluster_id, stats in self.cluster_stats.items():
            cluster_mask = self.labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            if selection_method == 'centroid':
                # Select frames closest to cluster center
                if self.distances is not None:
                    cluster_distances = self.distances[cluster_indices, cluster_id]
                    selected_indices = cluster_indices[np.argsort(cluster_distances)[:frames_per_cluster]]
                else:
                    # Calculate distances to cluster center manually
                    cluster_center = stats['center']
                    cluster_features = self.feature_matrix_scaled[cluster_indices]
                    distances_to_center = np.linalg.norm(cluster_features - cluster_center, axis=1)
                    selected_indices = cluster_indices[np.argsort(distances_to_center)[:frames_per_cluster]]
            
            elif selection_method == 'diverse':
                # Select diverse frames using maximum distance
                if len(cluster_indices) <= frames_per_cluster:
                    selected_indices = cluster_indices
                else:
                    # Start with centroid
                    cluster_center = stats['center']
                    cluster_features = self.feature_matrix_scaled[cluster_indices]
                    distances_to_center = np.linalg.norm(cluster_features - cluster_center, axis=1)
                    
                    selected_indices = [cluster_indices[np.argmin(distances_to_center)]]
                    
                    # Iteratively select most diverse frames
                    for _ in range(frames_per_cluster - 1):
                        remaining_indices = [i for i in cluster_indices if i not in selected_indices]
                        if not remaining_indices:
                            break
                        
                        max_min_distance = -1
                        best_idx = remaining_indices[0]
                        
                        for idx in remaining_indices:
                            min_distance = min([
                                np.linalg.norm(self.feature_matrix_scaled[idx] - self.feature_matrix_scaled[sel_idx])
                                for sel_idx in selected_indices
                            ])
                            if min_distance > max_min_distance:
                                max_min_distance = min_distance
                                best_idx = idx
                        
                        selected_indices.append(best_idx)
            
            elif selection_method == 'quality':
                # Select based on feature quality metrics
                cluster_features = [self.features_list[i] for i in cluster_indices]
                quality_scores = [f['feature_norm'] for f in cluster_features]  # Use feature norm as quality
                selected_indices = cluster_indices[np.argsort(quality_scores)[::-1][:frames_per_cluster]]
            
            # Add selected frames to result
            for idx in selected_indices:
                selected_frames.append({
                    'image_path': self.image_paths[idx],
                    'image_name': self.image_names[idx],
                    'cluster_id': int(cluster_id),
                    'cluster_size': stats['size'],
                    'features': self.features_list[idx],
                    'distance_to_center': float(np.linalg.norm(
                        self.feature_matrix_scaled[idx] - stats['center']
                    )) if selection_method == 'centroid' else None
                })
        
        self.selected_frames = selected_frames
        print(f"Selected {len(selected_frames)} frames from {len(self.cluster_stats)} clusters")
        return selected_frames
    
    def visualize_clusters(self, output_dir, n_components=2):
        """
        Create visualizations of the clustering results
        
        Args:
            output_dir: Directory to save visualizations
            n_components: Number of PCA components for visualization
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # PCA for visualization
        pca = PCA(n_components=n_components, random_state=self.random_state)
        features_pca = pca.fit_transform(self.feature_matrix_scaled)
        
        # Create cluster visualization
        plt.figure(figsize=(12, 8))
        
        if n_components == 2:
            scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], 
                                c=self.labels, cmap='tab20', alpha=0.7)
            plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.colorbar(scatter, label='Cluster ID')
        
        plt.title('DINO Feature Clustering Results')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Cluster size distribution
        plt.figure(figsize=(10, 6))
        cluster_sizes = [stats['size'] for stats in self.cluster_stats.values()]
        cluster_ids = list(self.cluster_stats.keys())
        
        plt.bar(cluster_ids, cluster_sizes)
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Images')
        plt.title('Cluster Size Distribution')
        plt.xticks(cluster_ids)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_sizes.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Selected frames visualization
        if hasattr(self, 'selected_frames'):
            selected_pca = features_pca[[i for i, path in enumerate(self.image_paths) 
                                       if path in [f['image_path'] for f in self.selected_frames]]]
            selected_labels = [self.labels[i] for i, path in enumerate(self.image_paths) 
                             if path in [f['image_path'] for f in self.selected_frames]]
            
            plt.figure(figsize=(12, 8))
            plt.scatter(features_pca[:, 0], features_pca[:, 1], 
                       c=self.labels, cmap='tab20', alpha=0.3, label='All frames')
            plt.scatter(selected_pca[:, 0], selected_pca[:, 1], 
                       c=selected_labels, cmap='tab20', s=100, 
                       edgecolors='black', linewidths=2, label='Selected frames')
            plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.title('Selected Representative Frames')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'selected_frames.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {output_dir}")
    
    def save_results(self, output_dir):
        """
        Save clustering results and selected frames
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save clustering results
        results = {
            'cluster_info': self.cluster_info,
            'cluster_stats': self.cluster_stats,
            'model_info': self.model_info,
            'processing_stats': self.processing_stats,
            'selected_frames': self.selected_frames if hasattr(self, 'selected_frames') else []
        }
        
        with open(os.path.join(output_dir, 'clustering_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save full data as pickle
        full_data = {
            'feature_matrix': self.feature_matrix,
            'feature_matrix_scaled': self.feature_matrix_scaled,
            'labels': self.labels,
            'image_paths': self.image_paths,
            'image_names': self.image_names,
            'cluster_info': self.cluster_info,
            'cluster_stats': self.cluster_stats
        }
        
        with open(os.path.join(output_dir, 'full_clustering_data.pkl'), 'wb') as f:
            pickle.dump(full_data, f)
        
        print(f"Results saved to {output_dir}")
    
    def copy_selected_frames(self, output_dir):
        """
        Copy selected frames to output directory
        
        Args:
            output_dir: Directory to copy selected frames
        """
        if not hasattr(self, 'selected_frames'):
            print("No frames selected yet. Run select_representative_frames first.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        copied_count = 0
        for frame in tqdm(self.selected_frames, desc="Copying selected frames"):
            src_path = frame['image_path']
            dst_path = os.path.join(output_dir, frame['image_name'])
            
            try:
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            except Exception as e:
                print(f"Error copying {src_path}: {e}")
        
        print(f"Copied {copied_count} selected frames to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Cluster DINO features and select representative frames')
    parser.add_argument('--features_file', type=str, required=True,
                        help='Path to DINO features pickle file')
    parser.add_argument('--output_dir', type=str, default='clustering_results',
                        help='Output directory for results')
    parser.add_argument('--n_clusters', type=int, default=20,
                        help='Number of clusters')
    parser.add_argument('--frames_per_cluster', type=int, default=5,
                        help='Number of frames to select per cluster')
    parser.add_argument('--selection_method', type=str, default='centroid',
                        choices=['centroid', 'diverse', 'quality'],
                        help='Frame selection method')
    parser.add_argument('--find_optimal_k', action='store_true',
                        help='Find optimal number of clusters')
    parser.add_argument('--copy_frames', action='store_true',
                        help='Copy selected frames to output directory')
    
    args = parser.parse_args()
    
    # Initialize clustering
    clustering = DINOClustering(args.features_file)
    
    # Find optimal clusters if requested
    if args.find_optimal_k:
        optimal_k, _ = clustering.find_optimal_clusters()
        print(f"Optimal number of clusters: {optimal_k}")
        args.n_clusters = optimal_k
    
    # Perform clustering
    clustering.perform_clustering(n_clusters=args.n_clusters)
    
    # Select representative frames
    clustering.select_representative_frames(
        frames_per_cluster=args.frames_per_cluster,
        selection_method=args.selection_method
    )
    
    # Create visualizations
    clustering.visualize_clusters(args.output_dir)
    
    # Save results
    clustering.save_results(args.output_dir)
    
    # Copy selected frames if requested
    if args.copy_frames:
        frames_dir = os.path.join(args.output_dir, 'selected_frames')
        clustering.copy_selected_frames(frames_dir)
    
    print("Clustering and frame selection completed successfully!")

if __name__ == "__main__":
    main()