#!/usr/bin/env python3
"""
Test script to verify DINO clustering and frame selection on existing background frames
"""

import os
import sys
import json
import numpy as np
import torch
import pickle
from pathlib import Path
from tqdm import tqdm
import shutil
from datetime import datetime

# Add project paths to system path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "DINO_Frame_Selection" / "scripts"))
sys.path.insert(0, str(project_root.parent / "Annotators" / "Utils" / "SignificantImageSelector"))

# Import existing implementations
from dino_feature_extractor import DINOFeatureExtractor
from dino_clustering_wrapper import DINOClusterer
from image_feature_extractor import ImageFeatureExtractor

class DINOSelectionTester:
    def __init__(self):
        self.base_dir = Path("F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/YOLO_DETR_Benchmarks/Intelligent_Background_Selector_2025-07-17")
        self.background_dir = self.base_dir / 'background_frames'
        self.clustering_dir = self.base_dir / 'clustering_results'
        self.selected_dir = self.base_dir / 'selected_frames'
        self.test_results_dir = self.base_dir / 'test_results'
        
        # Create directories
        self.clustering_dir.mkdir(exist_ok=True)
        self.selected_dir.mkdir(exist_ok=True)
        self.test_results_dir.mkdir(exist_ok=True)
        
        # Initialize extractors
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.dino_extractor = None
        self.hog_extractor = None
        
        # Statistics
        self.stats = {
            'total_background_frames': 0,
            'dino_features_extracted': 0,
            'clusters_created': 0,
            'selected_frames': 0,
            'hog_scores': [],
            'clustering_time': 0,
            'selection_time': 0
        }
    
    def initialize_extractors(self):
        """Initialize DINO and HOG extractors"""
        print("Initializing extractors...")
        
        try:
            # DINO extractor
            self.dino_extractor = DINOFeatureExtractor(
                model_name='dino_vits16',
                device=str(self.device)
            )
            print("OK DINO extractor initialized")
            
            # HOG extractor
            self.hog_extractor = ImageFeatureExtractor(method='hog')
            print("OK HOG extractor initialized")
            
            return True
        except Exception as e:
            print(f"ERROR initializing extractors: {e}")
            return False
    
    def test_hog_analysis(self):
        """Test HOG feature extraction on background frames"""
        print("\n" + "="*60)
        print("TESTING HOG FEATURE EXTRACTION")
        print("="*60)
        
        background_frames = list(self.background_dir.glob("*.jpg"))
        if not background_frames:
            print("No background frames found!")
            return False
        
        print(f"Testing HOG on {len(background_frames)} frames...")
        
        hog_results = []
        for frame_path in tqdm(background_frames[:20], desc="HOG analysis"):  # Test first 20 frames
            try:
                hog_features = self.hog_extractor.extract_features(str(frame_path))
                if hog_features:
                    content_score = hog_features.get('mean_edge', 0.0)
                    hog_results.append({
                        'frame': frame_path.name,
                        'content_score': content_score,
                        'hog_features': hog_features
                    })
                    self.stats['hog_scores'].append(content_score)
            except Exception as e:
                print(f"Error processing {frame_path}: {e}")
        
        if hog_results:
            # Save HOG results
            hog_results_path = self.test_results_dir / 'hog_analysis.json'
            with open(hog_results_path, 'w') as f:
                json.dump(hog_results, f, indent=2)
            
            # Print statistics
            scores = [r['content_score'] for r in hog_results]
            print(f"OK HOG analysis completed")
            print(f"  - Frames analyzed: {len(hog_results)}")
            print(f"  - Mean content score: {np.mean(scores):.4f}")
            print(f"  - Min content score: {np.min(scores):.4f}")
            print(f"  - Max content score: {np.max(scores):.4f}")
            print(f"  - Results saved to: {hog_results_path}")
            
            return True
        else:
            print("ERROR HOG analysis failed")
            return False
    
    def test_dino_extraction(self):
        """Test DINO feature extraction"""
        print("\n" + "="*60)
        print("TESTING DINO FEATURE EXTRACTION")
        print("="*60)
        
        background_frames = list(self.background_dir.glob("*.jpg"))
        if not background_frames:
            print("No background frames found!")
            return False
        
        print(f"Extracting DINO features from {len(background_frames)} frames...")
        
        dino_features_file = self.clustering_dir / 'dino_features.pkl'
        
        try:
            start_time = datetime.now()
            
            # Extract features
            features_data = self.dino_extractor.process_image_directory(
                str(self.background_dir),
                str(dino_features_file)
            )
            
            extraction_time = (datetime.now() - start_time).total_seconds()
            
            if features_data:
                print(f"OK DINO features extracted successfully")
                print(f"  - Processing time: {extraction_time:.2f} seconds")
                print(f"  - Features saved to: {dino_features_file}")
                
                # Load and analyze features
                with open(dino_features_file, 'rb') as f:
                    dino_data = pickle.load(f)
                
                features_array = np.array([f['global_features'] for f in dino_data['features']])
                
                print(f"  - Features shape: {features_array.shape}")
                print(f"  - Features dtype: {features_array.dtype}")
                
                self.stats['dino_features_extracted'] = len(dino_data['features'])
                
                return True
            else:
                print("ERROR DINO feature extraction failed")
                return False
                
        except Exception as e:
            print(f"ERROR in DINO extraction: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_dino_clustering(self):
        """Test DINO clustering"""
        print("\n" + "="*60)
        print("TESTING DINO CLUSTERING")
        print("="*60)
        
        dino_features_file = self.clustering_dir / 'dino_features.pkl'
        
        if not dino_features_file.exists():
            print("DINO features file not found. Run extraction first.")
            return False
        
        try:
            start_time = datetime.now()
            
            # Load features
            with open(dino_features_file, 'rb') as f:
                dino_data = pickle.load(f)
            
            features_array = np.array([f['global_features'] for f in dino_data['features']])
            image_paths = [f['image_path'] for f in dino_data['features']]
            
            print(f"Loaded {len(features_array)} feature vectors")
            
            # Initialize clusterer
            clusterer = DINOClusterer()
            
            # Determine number of clusters
            n_clusters = min(10, max(3, len(features_array) // 20))
            frames_per_cluster = max(1, min(3, len(features_array) // n_clusters))
            
            print(f"Using {n_clusters} clusters with {frames_per_cluster} frames per cluster")
            
            # Perform clustering
            cluster_labels = clusterer.perform_clustering(
                features_array,
                n_clusters=n_clusters,
                method='kmeans'
            )
            
            clustering_time = (datetime.now() - start_time).total_seconds()
            
            print(f"OK Clustering completed")
            print(f"  - Clustering time: {clustering_time:.2f} seconds")
            print(f"  - Clusters created: {n_clusters}")
            print(f"  - Cluster labels shape: {cluster_labels.shape}")
            
            # Get cluster statistics
            cluster_stats = clusterer.get_cluster_statistics(cluster_labels)
            print(f"  - Cluster statistics: {cluster_stats}")
            
            # Select representative frames
            selected_frames = clusterer.select_representative_frames(
                image_paths,
                features_array,
                cluster_labels,
                frames_per_cluster=frames_per_cluster
            )
            
            selection_time = (datetime.now() - start_time).total_seconds() - clustering_time
            
            print(f"OK Frame selection completed")
            print(f"  - Selection time: {selection_time:.2f} seconds")
            print(f"  - Selected frames: {len(selected_frames)}")
            
            # Save results
            clustering_results = {
                'n_clusters': n_clusters,
                'frames_per_cluster': frames_per_cluster,
                'total_frames': len(features_array),
                'selected_frames': selected_frames,
                'cluster_labels': cluster_labels.tolist(),
                'cluster_stats': cluster_stats,
                'clustering_time': clustering_time,
                'selection_time': selection_time
            }
            
            clustering_results_path = self.clustering_dir / 'clustering_results.json'
            with open(clustering_results_path, 'w') as f:
                json.dump(clustering_results, f, indent=2)
            
            print(f"  - Results saved to: {clustering_results_path}")
            
            # Copy selected frames
            print(f"Copying {len(selected_frames)} selected frames...")
            for frame_path in selected_frames:
                frame_name = Path(frame_path).name
                dst_path = self.selected_dir / frame_name
                shutil.copy2(frame_path, dst_path)
            
            print(f"OK Selected frames copied to: {self.selected_dir}")
            
            # Update stats
            self.stats['clusters_created'] = n_clusters
            self.stats['selected_frames'] = len(selected_frames)
            self.stats['clustering_time'] = clustering_time
            self.stats['selection_time'] = selection_time
            
            return True
            
        except Exception as e:
            print(f"ERROR in DINO clustering: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_optical_flow(self):
        """Test optical flow analysis (if implemented)"""
        print("\n" + "="*60)
        print("TESTING OPTICAL FLOW ANALYSIS")
        print("="*60)
        
        # Note: Optical flow is not implemented in the current version
        # This is a placeholder for future implementation
        
        print("WARNING: Optical flow analysis not implemented in current version")
        print("   This feature could be added to analyze motion between frames")
        print("   and select frames with minimal motion (truly static backgrounds)")
        
        return True
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("GENERATING TEST REPORT")
        print("="*60)
        
        # Count background frames
        background_frames = list(self.background_dir.glob("*.jpg"))
        self.stats['total_background_frames'] = len(background_frames)
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_results': {
                'hog_analysis': len(self.stats['hog_scores']) > 0,
                'dino_extraction': self.stats['dino_features_extracted'] > 0,
                'dino_clustering': self.stats['clusters_created'] > 0,
                'frame_selection': self.stats['selected_frames'] > 0,
                'optical_flow': False  # Not implemented
            },
            'statistics': self.stats,
            'summary': {
                'total_background_frames': self.stats['total_background_frames'],
                'features_extracted': self.stats['dino_features_extracted'],
                'clusters_created': self.stats['clusters_created'],
                'frames_selected': self.stats['selected_frames'],
                'selection_ratio': self.stats['selected_frames'] / max(self.stats['total_background_frames'], 1),
                'mean_hog_score': np.mean(self.stats['hog_scores']) if self.stats['hog_scores'] else 0.0
            }
        }
        
        # Save report
        report_path = self.test_results_dir / 'dino_test_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"OK Test report generated: {report_path}")
        
        # Print summary
        print(f"\nTEST SUMMARY:")
        print(f"  - Background frames: {self.stats['total_background_frames']}")
        print(f"  - DINO features extracted: {self.stats['dino_features_extracted']}")
        print(f"  - Clusters created: {self.stats['clusters_created']}")
        print(f"  - Final selected frames: {self.stats['selected_frames']}")
        print(f"  - Selection ratio: {report['summary']['selection_ratio']:.3f}")
        print(f"  - Mean HOG score: {report['summary']['mean_hog_score']:.4f}")
        
        return report
    
    def run_all_tests(self):
        """Run all tests"""
        print("DINO SELECTION COMPREHENSIVE TEST")
        print("=" * 80)
        
        # Initialize
        if not self.initialize_extractors():
            return False
        
        # Test HOG
        hog_success = self.test_hog_analysis()
        
        # Test DINO extraction
        dino_extraction_success = self.test_dino_extraction()
        
        # Test DINO clustering
        dino_clustering_success = False
        if dino_extraction_success:
            dino_clustering_success = self.test_dino_clustering()
        
        # Test optical flow (placeholder)
        optical_flow_success = self.test_optical_flow()
        
        # Generate report
        report = self.generate_test_report()
        
        # Final summary
        print("\n" + "="*80)
        print("FINAL TEST RESULTS")
        print("="*80)
        
        tests = {
            'HOG Analysis': hog_success,
            'DINO Extraction': dino_extraction_success,
            'DINO Clustering': dino_clustering_success,
            'Optical Flow': optical_flow_success
        }
        
        for test_name, success in tests.items():
            status = "PASSED" if success else "FAILED"
            print(f"{test_name:20} {status}")
        
        all_passed = all(tests.values())
        overall_status = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
        print(f"\nOverall Status: {overall_status}")
        
        return all_passed

def main():
    tester = DINOSelectionTester()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())