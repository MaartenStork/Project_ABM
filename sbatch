#!/bin/bash
#SBATCH --job-name=newt
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=1:00:00

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192

#SBATCH --output=$HOME/Logfile/%x-%j.out
#SBATCH --error =$HOME/Logfile/%x-%j.err
#SBATCH --open-mode=truncate

set -xeuo pipefail

### — paths — ###
LOG_DIR=$HOME/Logfile
REPO_DIR=$HOME/Project_ABM                 # <— note the capital “P” and underscore
SCRIPT_PATH=$REPO_DIR/sens_anl.py           # full path to your script
DATA_DIR=$REPO_DIR/Snelliusdata
### — end paths — ###

# If ~/Logfile exists as a file, remove it so it can become a directory
if [[ -e "$LOG_DIR" && ! -d "$LOG_DIR" ]]; then
  rm -f "$LOG_DIR"
fi

# Create necessary directories
mkdir -p "$LOG_DIR" "$DATA_DIR"

# Change into your repo
cd "$REPO_DIR"

# Load Python modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# Ensure pip is up to date and SALib is installed
python3 -m pip install --upgrade --user pip
python3 -m pip install --user SALib imageio

# Run your sensitivity-analysis script
python3 "$SCRIPT_PATH"

# (Optional) move any generated .png files into Snelliusdata
mv -- *.png "$DATA_DIR/" || true

echo "Job finished at $(date)"
