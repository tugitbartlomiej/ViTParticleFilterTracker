"""
Test script for Advanced Dataset Selection Pipeline.

Tests the pipeline on a small subset of images to verify everything works.
"""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from glob import glob

def test_fourier_analyzer():
    """Test Fourier Analyzer on real images."""
    print("\n" + "="*60)
    print("TEST 1: Fourier Analyzer")
    print("="*60)

    from AdvancedDatasetSelection.feature_extractors.fourier_analyzer import FourierAnalyzer
    import cv2

    # Get some test images
    test_dir = "E:/cataract_surgery_Instruments_detection.v1i.coco/train"
    image_paths = glob(os.path.join(test_dir, "*.jpg"))[:20]

    if not image_paths:
        print("ERROR: No images found!")
        return False

    print(f"Testing on {len(image_paths)} images...")

    analyzer = FourierAnalyzer(similarity_threshold=0.85)

    # Extract features
    features, valid_paths = analyzer.compute_features_batch(image_paths, show_progress=True)
    print(f"Extracted features shape: {features.shape}")

    # Test filtering
    selected = analyzer.filter_redundant(features, threshold=0.90)
    print(f"After redundancy filter: {len(selected)}/{len(features)} kept")

    # Analyze diversity
    diversity = analyzer.analyze_diversity(features)
    print(f"Mean similarity: {diversity['mean_similarity']:.3f}")
    print(f"Highly similar pairs: {diversity['highly_similar_pairs']}")

    print("Fourier Analyzer: PASSED")
    return True


def test_sam_proxy():
    """Test SAM proxy (without actual SAM model)."""
    print("\n" + "="*60)
    print("TEST 2: SAM Proxy (complexity without model)")
    print("="*60)

    from AdvancedDatasetSelection.feature_extractors.sam_extractor import SAMExtractor
    import cv2

    test_dir = "E:/cataract_surgery_Instruments_detection.v1i.coco/train"
    image_paths = glob(os.path.join(test_dir, "*.jpg"))[:10]

    if not image_paths:
        print("ERROR: No images found!")
        return False

    extractor = SAMExtractor(checkpoint_path=None)  # No SAM model

    print(f"Testing complexity proxy on {len(image_paths)} images...")

    scores = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            result = extractor.compute_complexity_without_sam(img)
            scores.append(result['complexity_score'])
            print(f"  {Path(path).name[:30]}: complexity={result['complexity_score']:.3f}")

    print(f"\nComplexity range: [{min(scores):.3f}, {max(scores):.3f}]")
    print(f"Mean complexity: {np.mean(scores):.3f}")

    print("SAM Proxy: PASSED")
    return True


def test_combined_selector():
    """Test combined selector on small dataset."""
    print("\n" + "="*60)
    print("TEST 3: Combined Selector (full pipeline)")
    print("="*60)

    from AdvancedDatasetSelection.selection_methods.combined_selector import CombinedSelector

    test_dir = "E:/cataract_surgery_Instruments_detection.v1i.coco/train"
    image_paths = glob(os.path.join(test_dir, "*.jpg"))[:50]

    if not image_paths:
        print("ERROR: No images found!")
        return False

    print(f"Testing combined selection on {len(image_paths)} images...")
    print(f"Target: select 10 best images")

    selector = CombinedSelector(cache_dir=None)

    # Run selection
    selected_indices, selected_paths, stats = selector.select_optimal_subset(
        image_paths=image_paths,
        target_size=10,
        fourier_threshold=0.90,
        oversampling=2.0,
        keep_hard=True,
        use_cache=False
    )

    print(f"\nSelection Statistics:")
    print(f"  Original: {stats['original_count']}")
    print(f"  Valid: {stats['valid_count']}")
    print(f"  After Fourier: {stats['after_fourier']}")
    print(f"  After k-Center: {stats['after_k_center']}")
    print(f"  Final: {stats['final_count']}")
    print(f"  EL2N mean: {stats['el2n_mean']:.4f}")
    print(f"  SAM mean: {stats['sam_mean']:.3f}")

    print(f"\nSelected images:")
    for i, path in enumerate(selected_paths[:5]):
        print(f"  {i+1}. {Path(path).name[:50]}")
    if len(selected_paths) > 5:
        print(f"  ... and {len(selected_paths) - 5} more")

    print("Combined Selector: PASSED")
    return True


def test_k_center():
    """Test k-Center Greedy selection."""
    print("\n" + "="*60)
    print("TEST 4: k-Center Greedy")
    print("="*60)

    from AdvancedDatasetSelection.selection_methods.k_center_greedy import KCenterGreedy

    # Create synthetic features (simulating DINO output)
    np.random.seed(42)
    n_samples = 100
    features = np.random.randn(n_samples, 768)

    print(f"Testing k-Center on {n_samples} samples with 768-dim features")

    selector = KCenterGreedy(distance_metric="euclidean")

    # Select 20 diverse samples
    selected = selector.select(features, k=20, seed=42, show_progress=True)

    # Analyze diversity
    diversity = selector.analyze_diversity(features, selected)

    print(f"\nSelected {len(selected)} samples")
    print(f"Coverage radius: {diversity['coverage_radius']:.3f}")
    print(f"Mean pairwise distance: {diversity['mean_pairwise_distance']:.3f}")
    print(f"Min pairwise distance: {diversity['min_pairwise_distance']:.3f}")

    print("k-Center Greedy: PASSED")
    return True


def test_el2n():
    """Test EL2N scorer."""
    print("\n" + "="*60)
    print("TEST 5: EL2N Scorer")
    print("="*60)

    from AdvancedDatasetSelection.selection_methods.el2n_scorer import (
        EL2NScorer, compute_proxy_el2n_from_features
    )

    # Create synthetic features and labels
    np.random.seed(42)
    n_samples = 100
    features = np.random.randn(n_samples, 768)
    labels = np.random.randint(0, 2, n_samples)

    print(f"Computing EL2N scores for {n_samples} samples...")

    # Compute scores using proxy method
    scores = compute_proxy_el2n_from_features(features, labels)

    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"Mean score: {scores.mean():.4f}")
    print(f"Std score: {scores.std():.4f}")

    # Test ranking
    scorer = EL2NScorer()
    scorer.scores = {f"img_{i}": float(scores[i]) for i in range(n_samples)}

    hardest = scorer.rank_by_difficulty(keep_hard=True, top_k=5)
    easiest = scorer.rank_by_difficulty(keep_hard=False, top_k=5)

    print(f"\nHardest 5 samples: {[scorer.scores[k] for k in hardest[:5]]}")
    print(f"Easiest 5 samples: {[scorer.scores[k] for k in easiest[:5]]}")

    print("EL2N Scorer: PASSED")
    return True


def test_visualization():
    """Test visualization module."""
    print("\n" + "="*60)
    print("TEST 6: Visualization")
    print("="*60)

    from AdvancedDatasetSelection.utils.visualization import Visualizer
    import tempfile

    # Create temp output dir
    output_dir = tempfile.mkdtemp()
    visualizer = Visualizer(output_dir=output_dir)

    print(f"Output dir: {output_dir}")

    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    features = np.random.randn(n_samples, 768)
    selected_indices = list(range(0, 100, 5))  # Every 5th

    # Test PCA plot
    print("Generating PCA coverage plot...")
    visualizer.plot_pca_coverage(features, selected_indices)

    # Test difficulty distribution
    print("Generating difficulty distribution plot...")
    scores = {f"img_{i}": np.random.random() for i in range(n_samples)}
    selected_keys = [f"img_{i}" for i in selected_indices]
    visualizer.plot_difficulty_distribution(scores, selected_keys)

    # Check files created
    import glob as g
    created_files = g.glob(os.path.join(output_dir, "*.png"))
    print(f"Created {len(created_files)} visualization files")

    for f in created_files:
        print(f"  - {Path(f).name}")

    print("Visualization: PASSED")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# ADVANCED DATASET SELECTION PIPELINE - TEST SUITE")
    print("#"*60)

    tests = [
        ("Fourier Analyzer", test_fourier_analyzer),
        ("SAM Proxy", test_sam_proxy),
        ("k-Center Greedy", test_k_center),
        ("EL2N Scorer", test_el2n),
        ("Visualization", test_visualization),
        ("Combined Selector", test_combined_selector),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nALL TESTS PASSED!")
        return True
    else:
        print("\nSOME TESTS FAILED!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
