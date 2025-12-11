"""
FAISS GPU K-Means Worker Script.
Runs in conda environment to avoid numpy/faiss version conflicts.
Called via subprocess from cluster_selector.py.

Usage:
    conda_python faiss_clustering_worker.py <features_path> <n_clusters> <output_path> [niter] [seed]
"""

import sys
import numpy as np
import pickle
import time

def main():
    if len(sys.argv) < 4:
        print("Usage: python faiss_clustering_worker.py <features_path> <n_clusters> <output_path> [niter] [seed]")
        sys.exit(1)

    features_path = sys.argv[1]
    n_clusters = int(sys.argv[2])
    output_path = sys.argv[3]
    niter = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42

    print(f"[FAISS Worker] Loading features from {features_path}")
    with open(features_path, 'rb') as f:
        features = pickle.load(f)

    n_samples, d = features.shape
    print(f"[FAISS Worker] Data: {n_samples:,} samples x {d} features -> {n_clusters:,} clusters")

    # Import faiss
    try:
        import faiss
        n_gpus = faiss.get_num_gpus()
        print(f"[FAISS Worker] faiss imported, {n_gpus} GPU(s) available")
    except ImportError as e:
        print(f"[FAISS Worker] ERROR: Could not import faiss: {e}")
        sys.exit(1)

    # Convert to float32
    print(f"[FAISS Worker] Converting to float32...")
    features_f32 = np.ascontiguousarray(features.astype(np.float32))

    # Run K-means
    use_gpu = n_gpus > 0
    mode = "GPU" if use_gpu else "CPU"
    print(f"[FAISS Worker] Starting {mode} K-means...")
    print(f"{'='*60}")

    start_time = time.time()

    kmeans = faiss.Kmeans(
        d, n_clusters,
        niter=niter,
        gpu=use_gpu,
        seed=seed,
        verbose=True,
        spherical=False,
        min_points_per_centroid=1
    )

    kmeans.train(features_f32)

    elapsed_train = time.time() - start_time
    print(f"{'='*60}")
    print(f"[FAISS Worker] Training complete in {elapsed_train:.1f}s")

    # Get cluster assignments
    print(f"[FAISS Worker] Computing cluster labels...")
    _, labels = kmeans.index.search(features_f32, 1)
    labels = labels.flatten()

    # Get centers
    centers = kmeans.centroids

    elapsed_total = time.time() - start_time
    print(f"[FAISS Worker] Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")

    # Save results
    result = {
        'labels': labels,
        'centers': centers,
        'n_clusters': n_clusters,
        'n_gpus': n_gpus,
        'elapsed_seconds': elapsed_total
    }

    print(f"[FAISS Worker] Saving results to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)

    print(f"[FAISS Worker] Done!")

if __name__ == "__main__":
    main()
