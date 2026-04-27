#!/bin/bash
# =============================================================================
# SLURM Array Template — Synthetic Time Series Experiments
# =============================================================================
# Adapt the lines marked <CONFIGURE> before submitting.
#
# Usage:
#   1. Run generate_jobs.sh to populate jobs/ and jobs/all_jobs.txt
#   2. Set the --array range to match the number of jobs generated
#   3. sbatch scripts/slurm_array_template.sh
# =============================================================================

#SBATCH --job-name=synth_tslib
#SBATCH --output=logs/slurm_%A_%a.out    # <CONFIGURE> log directory
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=1-4110%48               # <CONFIGURE> total jobs % max concurrency
#SBATCH --time=04:00:00                 # <CONFIGURE> wall time per job
#SBATCH --partition=YOUR_PARTITION      # <CONFIGURE>
#SBATCH --account=YOUR_ACCOUNT          # <CONFIGURE>
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# ---- environment setup ------------------------------------------------------
# <CONFIGURE> load your cluster modules and activate your virtual environment
# module load GCCcore/13.3.0
# module load Python/3.12.3
# source /path/to/venv/bin/activate

export TMPDIR=/tmp   # <CONFIGURE> scratch space for temp files

# ---- job dispatch -----------------------------------------------------------
JOB_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" jobs/all_jobs.txt)

if [ -f "${JOB_FILE}" ]; then
    bash "${JOB_FILE}"
else
    echo "Job file not found: ${JOB_FILE}"
    exit 1
fi
