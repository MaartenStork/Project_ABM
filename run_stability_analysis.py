"""
Run Parameter Stability Analysis

This script runs the comprehensive parameter stability analysis
with reasonable sample sizes for practical execution.
"""

import numpy as np
from parameter_stability_finder import StabilityAnalyzer

def main():
    """Run stability analysis with practical sample sizes"""
    
    print("="*60)
    print("RUNNING PARAMETER STABILITY ANALYSIS")
    print("="*60)
    
    # Define parameter ranges
    param_ranges = {
        'scale': (0.5, 4.0),
        'imitation_period': (1, 15),
        'cooperation_increase': (0.05, 0.5),
        'q': (0.2, 1.0),
        'trust_decrease': (0.05, 0.5)
    }
    
    print("Parameter ranges:")
    for param, (min_val, max_val) in param_ranges.items():
        print(f"  {param}: {min_val} - {max_val}")
    
    # Create analyzer
    analyzer = StabilityAnalyzer(param_ranges, output_dir='stability_analysis')
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    print(f"\nStarting analysis...")
    print(f"This will take approximately 30-45 minutes...")
    
    try:
        # Run analysis with practical sample sizes
        results = analyzer.run_full_analysis()
        
        print(f"\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Check the 'stability_analysis' directory for:")
        print(f"  - CSV files with raw results")
        print(f"  - PNG files with visualizations") 
        print(f"  - TXT file with stable parameter ranges")
        
        return results
        
    except KeyboardInterrupt:
        print(f"\nAnalysis interrupted by user")
        return None
    except Exception as e:
        print(f"\nError during analysis: {e}")
        return None

if __name__ == '__main__':
    results = main() 