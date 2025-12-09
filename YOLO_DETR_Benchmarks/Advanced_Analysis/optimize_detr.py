"""
DETR Ultra-Optimization Script
Test multiple configurations to beat YOLO
"""
import sys
import json
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Import existing modules
from run_inference import load_config, run_detr_inference

def test_configuration(config, conf_threshold, description):
    """Test DETR with specific configuration"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Confidence Threshold: {conf_threshold}")
    print(f"{'='*60}")

    # Update config
    config['inference_params']['confidence_threshold'] = conf_threshold
    output_suffix = description.replace(' ', '_').replace('.', '')
    config['output']['directory'] = f"./optimization_results/{output_suffix}"

    # Run inference
    try:
        run_detr_inference(config)

        # Evaluate
        gt_coco = COCO(config['dataset']['annotations_path'])
        pred_path = Path(config['output']['directory']) / "detr_predictions.json"

        if not pred_path.exists():
            print(f"SKIP: No predictions generated")
            return None

        pred_coco = gt_coco.loadRes(str(pred_path))

        coco_eval = COCOeval(gt_coco, pred_coco, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        results = {
            "description": description,
            "conf_threshold": conf_threshold,
            "mAP_0.5": float(coco_eval.stats[1]),
            "mAP_0.5:0.95": float(coco_eval.stats[0]),
            "mAP_0.75": float(coco_eval.stats[2]),
            "AR_max_100": float(coco_eval.stats[8]),
            "AR_large": float(coco_eval.stats[11]),
        }

        print(f"\n📊 RESULTS:")
        print(f"  mAP@0.5      = {results['mAP_0.5']*100:.2f}%")
        print(f"  mAP@0.5:0.95 = {results['mAP_0.5:0.95']*100:.2f}%")
        print(f"  AR@100       = {results['AR_max_100']*100:.2f}%")

        return results

    except Exception as e:
        print(f"ERROR: {e}")
        return None

def main():
    # Load base config
    config = load_config("config.yaml")

    # YOLO baseline to beat
    yolo_baseline = {
        "mAP_0.5": 0.8652,
        "mAP_0.5:0.95": 0.7988,
        "AR_max_100": 0.9460
    }

    print("\n" + "="*60)
    print("🎯 DETR ULTRA-OPTIMIZATION")
    print("="*60)
    print(f"\n📌 YOLO Baseline to Beat:")
    print(f"  mAP@0.5      = {yolo_baseline['mAP_0.5']*100:.2f}%")
    print(f"  mAP@0.5:0.95 = {yolo_baseline['mAP_0.5:0.95']*100:.2f}%")
    print(f"  AR@100       = {yolo_baseline['AR_max_100']*100:.2f}%")

    # Test configurations
    test_configs = [
        (0.5, "Baseline conf=0.5"),
        (0.4, "Lower conf=0.4"),
        (0.3, "Lower conf=0.3"),
        (0.2, "Lower conf=0.2"),
        (0.6, "Higher conf=0.6"),
    ]

    all_results = []

    for conf, desc in test_configs:
        result = test_configuration(config, conf, desc)
        if result:
            all_results.append(result)

    # Find best
    if all_results:
        print("\n" + "="*60)
        print("🏆 OPTIMIZATION RESULTS")
        print("="*60)

        best_map_05 = max(all_results, key=lambda x: x['mAP_0.5'])
        best_map_overall = max(all_results, key=lambda x: x['mAP_0.5:0.95'])
        best_ar = max(all_results, key=lambda x: x['AR_max_100'])

        print(f"\n🥇 Best mAP@0.5: {best_map_05['description']}")
        print(f"   Value: {best_map_05['mAP_0.5']*100:.2f}% (YOLO: {yolo_baseline['mAP_0.5']*100:.2f}%)")
        print(f"   Gap: {(best_map_05['mAP_0.5'] - yolo_baseline['mAP_0.5'])*100:+.2f}%")

        print(f"\n🥇 Best mAP@0.5:0.95: {best_map_overall['description']}")
        print(f"   Value: {best_map_overall['mAP_0.5:0.95']*100:.2f}% (YOLO: {yolo_baseline['mAP_0.5:0.95']*100:.2f}%)")
        print(f"   Gap: {(best_map_overall['mAP_0.5:0.95'] - yolo_baseline['mAP_0.5:0.95'])*100:+.2f}%")

        print(f"\n🥇 Best AR@100: {best_ar['description']}")
        print(f"   Value: {best_ar['AR_max_100']*100:.2f}% (YOLO: {yolo_baseline['AR_max_100']*100:.2f}%)")
        print(f"   Gap: {(best_ar['AR_max_100'] - yolo_baseline['AR_max_100'])*100:+.2f}%")

        # Save results
        output_file = Path("./optimization_results/optimization_summary.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump({
                "yolo_baseline": yolo_baseline,
                "detr_configurations": all_results,
                "best_mAP_0.5": best_map_05,
                "best_mAP_0.5:0.95": best_map_overall,
                "best_AR": best_ar
            }, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")

        # Analysis
        print("\n" + "="*60)
        print("📈 ANALYSIS")
        print("="*60)

        if best_map_05['mAP_0.5'] >= yolo_baseline['mAP_0.5']:
            print("\n✅ DETR WINS on mAP@0.5!")
        else:
            print("\n❌ YOLO still ahead on mAP@0.5")
            print(f"   Gap to close: {(yolo_baseline['mAP_0.5'] - best_map_05['mAP_0.5'])*100:.2f}%")

        if best_map_overall['mAP_0.5:0.95'] >= yolo_baseline['mAP_0.5:0.95']:
            print("\n✅ DETR WINS on mAP@0.5:0.95!")
        else:
            print("\n❌ YOLO still ahead on mAP@0.5:0.95")
            print(f"   Gap to close: {(yolo_baseline['mAP_0.5:0.95'] - best_map_overall['mAP_0.5:0.95'])*100:.2f}%")

if __name__ == "__main__":
    main()
