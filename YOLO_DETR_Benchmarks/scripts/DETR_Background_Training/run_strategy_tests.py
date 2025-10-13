#!/usr/bin/env python3
"""
RUN STRATEGY TESTS - Main Entry Point
====================================

Prosty script do uruchomienia kompletnego testowania strategii:
1. Jednorazowa DINO extraction z E:\PicsOnly
2. Testowanie różnych strategii na tym samym datasecie
3. Porównanie wyników i wybór najlepszej strategii

Usage: python run_strategy_tests.py
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

class StrategyTestRunner:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.python_exe = 'py -3.11'
        
        # Check if E:\PicsOnly exists
        self.pics_dir = Path("E:/PicsOnly")
        if not self.pics_dir.exists():
            print(f"ERROR: {self.pics_dir} not found!")
            print("Please make sure E:\\PicsOnly directory exists with images.")
            sys.exit(1)
        
        print(f"Found image directory: {self.pics_dir}")
        
    def run_dino_extraction(self):
        """Run one-time DINO extraction"""
        print("\n" + "="*60)
        print("STEP 1: DINO EXTRACTION FROM E:\\PicsOnly")
        print("="*60)
        
        # Check if dataset already exists
        dino_dataset = Path("test_dino_dataset")
        if dino_dataset.exists() and (dino_dataset / "interesting_frames").exists():
            print("DINO dataset already exists!")
            response = input("Use existing dataset? (y/n): ").lower()
            if response == 'y':
                print("Using existing DINO dataset.")
                return True
            else:
                print("Will recreate DINO dataset...")
        
        # Run extraction
        script_path = self.base_dir / "dino_one_time_extraction.py"
        if not script_path.exists():
            print(f"ERROR: {script_path} not found!")
            return False
        
        command = f'{self.python_exe} "{script_path}"'
        print(f"Running: {command}")
        
        try:
            result = subprocess.run(command, shell=True, cwd=str(self.base_dir))
            if result.returncode == 0:
                print("DINO extraction completed successfully!")
                return True
            else:
                print(f"DINO extraction failed with code {result.returncode}")
                return False
        except Exception as e:
            print(f"Error running DINO extraction: {e}")
            return False
    
    def test_priority_strategies(self):
        """Test the most promising strategies first"""
        print("\n" + "="*60)
        print("STEP 2: TESTING PRIORITY STRATEGIES")
        print("="*60)
        
        # Define priority strategies (most likely to succeed)
        priority_strategies = [
            'ultra_gentle',      # Ultra-low LR
            'freeze_backbone',   # Proven approach
            'tooltip_heavy_95',  # Minimal background
            'two_stage'          # Reinforcement first
        ]
        
        framework_script = self.base_dir / "strategy_testing_framework.py"
        if not framework_script.exists():
            print(f"ERROR: {framework_script} not found!")
            return False
        
        successful_strategies = []
        
        for i, strategy in enumerate(priority_strategies, 1):
            print(f"\n[{i}/{len(priority_strategies)}] Testing strategy: {strategy}")
            
            command = f'{self.python_exe} "{framework_script}" --strategy {strategy}'
            print(f"Running: {command}")
            
            try:
                result = subprocess.run(command, shell=True, cwd=str(self.base_dir))
                if result.returncode == 0:
                    print(f"✅ Strategy '{strategy}' completed successfully!")
                    successful_strategies.append(strategy)
                else:
                    print(f"❌ Strategy '{strategy}' failed")
            except Exception as e:
                print(f"Error testing strategy '{strategy}': {e}")
        
        print(f"\nPriority testing completed!")
        print(f"Successful strategies: {successful_strategies}")
        
        return len(successful_strategies) > 0
    
    def test_remaining_strategies(self):
        """Test remaining strategies if needed"""
        print("\n" + "="*60)
        print("STEP 3: TESTING REMAINING STRATEGIES (OPTIONAL)")
        print("="*60)
        
        # Ask user if they want to test more strategies
        response = input("Test remaining strategies? (y/n): ").lower()
        if response != 'y':
            print("Skipping remaining strategies.")
            return
        
        remaining_strategies = [
            'dynamic_freezing',
            'gradual_phases',
            'weighted_loss',
            'memory_buffer',
            'ewc_regularization'
        ]
        
        framework_script = self.base_dir / "strategy_testing_framework.py"
        
        for strategy in remaining_strategies:
            print(f"\nTesting strategy: {strategy}")
            
            command = f'{self.python_exe} "{framework_script}" --strategy {strategy}'
            
            try:
                result = subprocess.run(command, shell=True, cwd=str(self.base_dir))
                if result.returncode == 0:
                    print(f"✅ Strategy '{strategy}' completed")
                else:
                    print(f"❌ Strategy '{strategy}' failed")
            except Exception as e:
                print(f"Error testing strategy '{strategy}': {e}")
    
    def compare_results(self):
        """Compare all tested strategies"""
        print("\n" + "="*60)
        print("STEP 4: COMPARING STRATEGY RESULTS")
        print("="*60)
        
        framework_script = self.base_dir / "strategy_testing_framework.py"
        command = f'{self.python_exe} "{framework_script}" --compare'
        
        print(f"Running comparison: {command}")
        
        try:
            result = subprocess.run(command, shell=True, cwd=str(self.base_dir))
            if result.returncode == 0:
                print("✅ Strategy comparison completed!")
            else:
                print("❌ Strategy comparison failed")
        except Exception as e:
            print(f"Error running comparison: {e}")
    
    def show_recommendations(self):
        """Show final recommendations"""
        print("\n" + "="*60)
        print("FINAL RECOMMENDATIONS")
        print("="*60)
        
        # Try to load comparison results
        results_dir = Path("strategy_results")
        summary_file = results_dir / "strategy_comparison_summary.json"
        
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                
                print(f"Strategies tested: {summary.get('strategies_tested', 0)}")
                print(f"Successful: {summary.get('successful', 0)}")
                
                # Show best strategy
                results = summary.get('results', [])
                successful = [r for r in results if r.get('success', False)]
                
                if successful:
                    best = successful[0]  # Already sorted by framework
                    print(f"\n🏆 BEST STRATEGY: {best['strategy']}")
                    print(f"   Training time: {best.get('training_time_minutes', 0):.1f} minutes")
                    print(f"   Configuration: {json.dumps(best.get('config', {}), indent=6)}")
                    
                    print(f"\n📝 RECOMMENDATION:")
                    print(f"   Use strategy '{best['strategy']}' for production training.")
                    print(f"   Expected to prevent catastrophic forgetting.")
                else:
                    print("\n❌ NO SUCCESSFUL STRATEGIES FOUND")
                    print("   Consider:")
                    print("   - Even lower learning rates (1e-9, 1e-10)")
                    print("   - Stronger freezing (freeze more layers)")
                    print("   - Higher tooltip ratios (98/2, 99/1)")
                
            except Exception as e:
                print(f"Error reading results: {e}")
        else:
            print("No comparison results found.")
        
        print(f"\n📁 All results saved in: {results_dir}")
    
    def run(self):
        """Run complete strategy testing pipeline"""
        print("DETR BACKGROUND TRAINING - STRATEGY TESTING")
        print("="*60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Working directory: {self.base_dir}")
        print("="*60)
        
        # Step 1: DINO Extraction
        if not self.run_dino_extraction():
            print("❌ DINO extraction failed - cannot proceed")
            return
        
        # Step 2: Test priority strategies
        if not self.test_priority_strategies():
            print("❌ All priority strategies failed - trying more strategies")
            self.test_remaining_strategies()
        
        # Step 3: Test remaining strategies (optional)
        else:
            print("✅ Some priority strategies succeeded!")
            self.test_remaining_strategies()
        
        # Step 4: Compare results
        self.compare_results()
        
        # Step 5: Show recommendations
        self.show_recommendations()
        
        print("\n" + "="*60)
        print("STRATEGY TESTING COMPLETED!")
        print("="*60)

def main():
    print("="*60)
    print("DETR Background Training - Strategy Testing Suite")
    print("="*60)
    print()
    print("This script will:")
    print("1. Extract diverse frames from E:\\PicsOnly using DINO")
    print("2. Test multiple training strategies to prevent catastrophic forgetting")
    print("3. Compare results and recommend the best strategy")
    print("4. Show detailed logs and performance metrics")
    print()
    
    response = input("Continue? (y/n): ").lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    runner = StrategyTestRunner()
    runner.run()

if __name__ == "__main__":
    main()