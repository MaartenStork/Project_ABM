#!/bin/bash
#SBATCH --job-name=newt
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=1:00:00

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192

# put Slurm’s own logs here
#SBATCH --output=$HOME/Logfile/%x-%j.out
#SBATCH --error =$HOME/Logfile/%x-%j.err

set -xeuo pipefail

### — paths & names — ###
HOME_DIR=$HOME
LOG_DIR=$HOME_DIR/Logfile
REPO_DIR=$HOME_DIR/project_abm       # where you’ve uploaded your repo
DATA_DIR=$REPO_DIR/Snelliusdata       # if you want your script’s outputs saved there
SCRIPT=sens_anl.py
### — end paths — ###

mkdir -p "$LOG_DIR" "$DATA_DIR"
cd "$REPO_DIR"

### — load Python & pip — ###
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

# upgrade pip (optional)
python3 -m pip install --upgrade --user pip

# install only the extra library your script needs:
#   SALib  ← for sobol_sample, morris_sample, sobol_analyze, morris_analyze
python3 -m pip install --user SALib
### — end pip installs — ###

# run your sensitivity‐analysis script
# it will read/write whatever files it needs (PNG’s, etc.)
python3 "$SCRIPT"

# done
echo "Job completed at $(date)"
