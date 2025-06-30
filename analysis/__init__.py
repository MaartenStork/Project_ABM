"""
Analysis module for the Dynamic Fishery Cooperation Model.
Contains sensitivity analysis and multi-run simulation tools.
"""
import os
import sys

# Add the parent directory to path so analysis modules can import from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# These will be uncommented once you move specific files to this directory
# from .sens_anl import run_sensitivity_analysis
# from .morris_analysis_no_mpa import run_morris_analysis
# from .run_multiple_simulations import run_batch_simulations
