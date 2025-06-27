"""
Quick Test of Comprehensive Stability Analysis

Test the comprehensive analysis framework with small sample sizes
to verify everything works before running the full analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from comprehensive_stability_analysis import ComprehensiveStabilityAnalyzer

def quick_test():
    """Quick test with small sample sizes"""
    print("="*50)
    print("QUICK TEST OF COMPREHENSIVE ANALYSIS")
    print("="*50)
    
    # Small parameter ranges for testing
    param_ranges = {
        'scale': (0.5, 3.0),
        'q': (0.3, 0.9),
        'cooperation_increase': (0.1, 0.4)
    }
    
    print("Test parameter ranges:")
    for param, (min_val, max_val) in param_ranges.items():
        print(f"  {param}: {min_val} - {max_val}")
    
    # Set seed
    np.random.seed(42)
    
    # Create analyzer
    analyzer = ComprehensiveStabilityAnalyzer(output_dir='test_comprehensive')
    
    try:
        # Run small analysis
        print(f"\nRunning quick test (5 samples, 3 reps each)...")
        df = analyzer.comprehensive_parameter_sweep(param_ranges, n_samples=5, n_reps=3)
        
        print(f"\nQuick test results:")
        print(f"Total data points: {len(df)}")
        print(f"Stability score range: {df['stability_mean'].min():.3f} - {df['stability_mean'].max():.3f}")
        
        # Show sample results
        print(f"\nSample results:")
        for param in param_ranges.keys():
            param_data = df[df['parameter'] == param]
            best_idx = param_data['stability_mean'].idxmax()
            best_row = param_data.loc[best_idx]
            print(f"  {param}: best at {best_row['value']:.2f} (stability: {best_row['stability_mean']:.3f})")
        
        # Test threshold finding
        thresholds = analyzer.find_parameter_thresholds(df, threshold_value=0.3)
        
        print(f"\nThreshold analysis:")
        for param, thresh in thresholds.items():
            if thresh is not None:
                print(f"  {param}: boundary at ~{thresh['boundary_estimate']:.2f}")
            else:
                print(f"  {param}: no clear boundary")
        
        # Save test results
        df.to_csv('test_comprehensive/quick_test_results.csv', index=False)
        
        print(f"\n✅ QUICK TEST SUCCESSFUL!")
        print(f"Framework is working correctly.")
        print(f"Ready to run full comprehensive analysis.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ QUICK TEST FAILED: {e}")
        return False

if __name__ == '__main__':
    success = quick_test()
    if success:
        print(f"\nTo run full analysis, execute:")
        print(f"python comprehensive_stability_analysis.py")
    else:
        print(f"\nFix issues before running full analysis.") 