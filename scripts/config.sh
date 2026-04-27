#!/bin/bash
# =============================================================================
# Experiment Configuration
# =============================================================================
# Central configuration for all synthetic time series experiments.
# Paper: "Synthetic Time Series for Training Forecasters"
# =============================================================================

# -----------------------------------------------------------------------------
# Directory Structure
# -----------------------------------------------------------------------------
# Dynamically resolve project root (2 levels up from this file: scripts/experiments/ -> project root)
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RESULTS_DIR="${PROJECT_ROOT}/results/experiments"
export LOGS_DIR="${PROJECT_ROOT}/logs/experiments"
export CHECKPOINTS_DIR="${PROJECT_ROOT}/checkpoints/experiments"

# Create directories
mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}" "${CHECKPOINTS_DIR}"

# -----------------------------------------------------------------------------
# Training Configuration
# -----------------------------------------------------------------------------
export TRAIN_EPOCHS=10
export BATCH_SIZE=32
export LEARNING_RATE=0.0001
export NUM_WORKERS=4
export PATIENCE=3  # For reference, not used with fixed epochs

# Phase 1: 3 seeds, Phase 2: 5 seeds
export SEEDS_PHASE1="2021 2022 2023"
export SEEDS_PHASE2="2021 2022 2023 2024 2025"

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
export MODELS="DLinear SegRNN TimesNet iTransformer PatchTST"

# -----------------------------------------------------------------------------
# Datasets Configuration
# -----------------------------------------------------------------------------
# Format: DATASET_NAME:ROOT_PATH:DATA_PATH:ENC_IN:FREQ:DATA_TYPE

export DATASETS_FULL="
ETTh1:./dataset/ETT-small:ETTh1.csv:7:h:ETTh1
ETTh2:./dataset/ETT-small:ETTh2.csv:7:h:ETTh2
ETTm1:./dataset/ETT-small:ETTm1.csv:7:t:ETTm1
ETTm2:./dataset/ETT-small:ETTm2.csv:7:t:ETTm2
Weather:./dataset/weather:weather.csv:21:h:custom
Electricity:./dataset/electricity:electricity.csv:321:h:custom
Traffic:./dataset/traffic:traffic.csv:862:h:custom
"

# Subset for ablation studies (faster iteration)
export DATASETS_ABLATION="
ETTh1:./dataset/ETT-small:ETTh1.csv:7:h:ETTh1
ETTm1:./dataset/ETT-small:ETTm1.csv:7:t:ETTm1
Weather:./dataset/weather:weather.csv:21:h:custom
Electricity:./dataset/electricity:electricity.csv:321:h:custom
"

# -----------------------------------------------------------------------------
# Prediction Horizons
# -----------------------------------------------------------------------------
export PRED_LENS="96 336"

# -----------------------------------------------------------------------------
# Synthetic Data Parameters
# -----------------------------------------------------------------------------
# Difficulty modes
export DIFFICULTY_MODES="uniform easy medium hard"

# Bundle types
export BUNDLE_TYPES="ST NR LM VE"

# Synth ratios for augmentation
export SYNTH_RATIOS="0.25 0.5 1.0"

# Synth ratios for low-resource (more aggressive)
export SYNTH_RATIOS_LOWRES="0.5 1.0 2.0"

# Sparsity levels
export SPARSITY_LEVELS="0.1 0.25 0.5"

# Latent factor probabilities
export LATENT_FACTORS="0.0 0.3 0.5 0.7"

# Annealing configurations
export ANNEAL_MODES="anneal anneali"
export ANNEAL_STRATEGIES="hard gradual"

# -----------------------------------------------------------------------------
# Model-Specific Hyperparameters (Paper Defaults)
# -----------------------------------------------------------------------------
# These functions return model-specific arguments

get_model_args() {
    local model=$1
    local enc_in=$2
    local pred_len=$3

    # Common args
    local common="--enc_in ${enc_in} --dec_in ${enc_in} --c_out ${enc_in}"

    case $model in
        "DLinear")
            # DLinear: Simple linear model
            # Paper: individual channels work better for most datasets
            echo "${common} --individual"
            ;;
        "SegRNN")
            # SegRNN: Segment-based RNN
            # Paper defaults: seg_len=48, d_model=512
            echo "${common} --seg_len 48 --d_model 512 --dropout 0.5"
            ;;
        "TimesNet")
            # TimesNet: 2D temporal variation modeling
            # Paper defaults: d_model=64, d_ff=64, e_layers=2, top_k=5
            echo "${common} --d_model 64 --d_ff 64 --e_layers 2 --top_k 5 --num_kernels 6"
            ;;
        "iTransformer")
            # iTransformer: Inverted transformer
            # Paper defaults: d_model=512, d_ff=512, e_layers=4, n_heads=8
            echo "${common} --d_model 512 --d_ff 512 --e_layers 4 --n_heads 8 --dropout 0.1"
            ;;
        "PatchTST")
            # PatchTST: Patch-based transformer
            # Paper defaults: d_model=128, d_ff=256, e_layers=3, n_heads=16, patch_len=16, stride=8
            echo "${common} --d_model 128 --d_ff 256 --e_layers 3 --n_heads 16 --patch_len 16 --stride 8 --dropout 0.2"
            ;;
        *)
            echo "${common}"
            ;;
    esac
}

# -----------------------------------------------------------------------------
# Result CSV Header
# -----------------------------------------------------------------------------
export CSV_HEADER="experiment_group,model,dataset,pred_len,seed,data_mode,synth_ratio,sparsity,difficulty_mode,bundle,p_latent_factor,anneal_strategy,anneal_epoch,mse,mae,train_time,timestamp"

init_results_csv() {
    local csv_file=$1
    if [ ! -f "${csv_file}" ]; then
        echo "${CSV_HEADER}" > "${csv_file}"
    fi
}

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

parse_dataset() {
    # Parse dataset string: NAME:ROOT:PATH:ENC_IN:FREQ:DATA_TYPE
    local ds_str=$1
    IFS=':' read -r DS_NAME DS_ROOT DS_PATH DS_ENC_IN DS_FREQ DS_DATA_TYPE <<< "${ds_str}"
    export DS_NAME DS_ROOT DS_PATH DS_ENC_IN DS_FREQ DS_DATA_TYPE
}

log_experiment() {
    local msg=$1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${msg}"
}

get_experiment_id() {
    # Generate unique experiment ID
    local model=$1
    local dataset=$2
    local pred_len=$3
    local seed=$4
    local suffix=$5
    echo "${model}_${dataset}_${pred_len}_s${seed}_${suffix}"
}
