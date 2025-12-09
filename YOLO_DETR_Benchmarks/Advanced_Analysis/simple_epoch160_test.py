"""
Simple DETR Epoch 160 test with confidence threshold optimization
"""
import sys
import json
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Import existing modules
from run_inference import load_config, run_detr_inference, run_yolo_inference

def test_detr_configuration(config, conf_threshold, checkpoint_path, description):
    """Test DETR with specific configuration"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Confidence Threshold: {conf_threshold}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    # Update config
    config['inference_params']['confidence_threshold'] = conf_threshold
    config['models']['detr']['path'] = checkpoint_path
    output_suffix = description.replace(' ', '_').replace('.', '').replace('=', '')
    config['output']['directory'] = f"./benchmark_results_epoch160/{output_suffix}"

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
            "checkpoint": checkpoint_path,
            "mAP_0.5": float(coco_eval.stats[1]),
            "mAP_0.5:0.95": float(coco_eval.stats[0]),
            "mAP_0.75": float(coco_eval.stats[2]),
            "AR_max_100": float(coco_eval.stats[8]),
            "AR_small": float(coco_eval.stats[9]),
            "AR_medium": float(coco_eval.stats[10]),
            "AR_large": float(coco_eval.stats[11]),
        }

        print(f"\n📊 RESULTS:")
        print(f"  mAP@0.5      = {results['mAP_0.5']*100:.2f}%")
        print(f"  mAP@0.5:0.95 = {results['mAP_0.5:0.95']*100:.2f}%")
        print(f"  AR@100       = {results['AR_max_100']*100:.2f}%")

        return results

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_yolo_baseline_simple(config):
    """Run YOLO baseline"""
    print(f"\n{'='*60}")
    print("Running YOLO Baseline")
    print(f"{'='*60}")

    config['output']['directory'] = "./benchmark_results_epoch160/yolo_baseline"

    try:
        run_yolo_inference(config)

        # Evaluate
        gt_coco = COCO(config['dataset']['annotations_path'])
        pred_path = Path(config['output']['directory']) / "yolo_predictions.json"

        if not pred_path.exists():
            print(f"SKIP: No predictions generated")
            return None

        pred_coco = gt_coco.loadRes(str(pred_path))

        coco_eval = COCOeval(gt_coco, pred_coco, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        results = {
            "description": "YOLO Epoch 100",
            "mAP_0.5": float(coco_eval.stats[1]),
            "mAP_0.5:0.95": float(coco_eval.stats[0]),
            "mAP_0.75": float(coco_eval.stats[2]),
            "AR_max_100": float(coco_eval.stats[8]),
            "AR_small": float(coco_eval.stats[9]),
            "AR_medium": float(coco_eval.stats[10]),
            "AR_large": float(coco_eval.stats[11]),
        }

        print(f"\n📊 YOLO RESULTS:")
        print(f"  mAP@0.5      = {results['mAP_0.5']*100:.2f}%")
        print(f"  mAP@0.5:0.95 = {results['mAP_0.5:0.95']*100:.2f}%")
        print(f"  AR@100       = {results['AR_max_100']*100:.2f}%")

        return results

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Load base config
    config = load_config("config.yaml")

    print("\n" + "="*60)
    print("🎯 DETR EPOCH 160 OPTIMIZATION TEST")
    print("="*60)

    # First run YOLO baseline
    print("\n📌 Running YOLO Baseline...")
    yolo_results = run_yolo_baseline_simple(config)

    if not yolo_results:
        print("❌ YOLO baseline failed!")
        return

    print("\n📌 YOLO Baseline to Beat:")
    print(f"  mAP@0.5      = {yolo_results['mAP_0.5']*100:.2f}%")
    print(f"  mAP@0.5:0.95 = {yolo_results['mAP_0.5:0.95']*100:.2f}%")
    print(f"  AR@100       = {yolo_results['AR_max_100']*100:.2f}%")

    # Test DETR Epoch 160 with different thresholds
    checkpoint_160 = "../../DETR_Checkpoints/checkpoint_epoch_160.pth"

    test_configs = [
        (0.5, "DETR Epoch 160 conf=0.5"),
        (0.4, "DETR Epoch 160 conf=0.4"),
        (0.3, "DETR Epoch 160 conf=0.3"),
        (0.2, "DETR Epoch 160 conf=0.2"),
        (0.15, "DETR Epoch 160 conf=0.15"),
    ]

    all_results = []

    for conf, desc in test_configs:
        result = test_detr_configuration(config, conf, checkpoint_160, desc)
        if result:
            all_results.append(result)

    # Compare
    if all_results:
        print("\n" + "="*60)
        print("🏆 OPTIMIZATION RESULTS")
        print("="*60)

        best_map_05 = max(all_results, key=lambda x: x['mAP_0.5'])
        best_map_overall = max(all_results, key=lambda x: x['mAP_0.5:0.95'])
        best_ar = max(all_results, key=lambda x: x['AR_max_100'])

        print(f"\n🥇 Best mAP@0.5: {best_map_05['description']}")
        print(f"   Value: {best_map_05['mAP_0.5']*100:.2f}% (YOLO: {yolo_results['mAP_0.5']*100:.2f}%)")
        gap = (best_map_05['mAP_0.5'] - yolo_results['mAP_0.5'])*100
        print(f"   Gap: {gap:+.2f}%")
        if gap > 0:
            print("   🎉 DETR WINS!")
        else:
            print(f"   Still {abs(gap):.2f}% behind YOLO")

        print(f"\n🥇 Best mAP@0.5:0.95: {best_map_overall['description']}")
        print(f"   Value: {best_map_overall['mAP_0.5:0.95']*100:.2f}% (YOLO: {yolo_results['mAP_0.5:0.95']*100:.2f}%)")
        gap = (best_map_overall['mAP_0.5:0.95'] - yolo_results['mAP_0.5:0.95'])*100
        print(f"   Gap: {gap:+.2f}%")
        if gap > 0:
            print("   🎉 DETR WINS!")
        else:
            print(f"   Still {abs(gap):.2f}% behind YOLO")

        print(f"\n🥇 Best AR@100: {best_ar['description']}")
        print(f"   Value: {best_ar['AR_max_100']*100:.2f}% (YOLO: {yolo_results['AR_max_100']*100:.2f}%)")
        gap = (best_ar['AR_max_100'] - yolo_results['AR_max_100'])*100
        print(f"   Gap: {gap:+.2f}%")
        if gap > 0:
            print("   🎉 DETR WINS!")
        else:
            print(f"   Still {abs(gap):.2f}% behind YOLO")

        # Save results
        output_file = Path("./benchmark_results_epoch160/summary.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump({
                "yolo_baseline": yolo_results,
                "detr_epoch_160_configurations": all_results,
                "best_mAP_0.5": best_map_05,
                "best_mAP_0.5:0.95": best_map_overall,
                "best_AR": best_ar
            }, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")

        # Final verdict
        print("\n" + "="*60)
        print("📈 FINAL VERDICT")
        print("="*60)

        if best_map_05['mAP_0.5'] >= yolo_results['mAP_0.5']:
            print("\n✅ DETR EPOCH 160 BEATS YOLO ON mAP@0.5!")
        else:
            print("\n❌ YOLO still ahead on mAP@0.5")

        if best_map_overall['mAP_0.5:0.95'] >= yolo_results['mAP_0.5:0.95']:
            print("✅ DETR EPOCH 160 BEATS YOLO ON mAP@0.5:0.95!")
        else:
            print("❌ YOLO still ahead on mAP@0.5:0.95")

        if best_ar['AR_max_100'] >= yolo_results['AR_max_100']:
            print("✅ DETR EPOCH 160 BEATS YOLO ON AR@100!")
        else:
            print("❌ YOLO still ahead on AR@100")

if __name__ == "__main__":
    main()
