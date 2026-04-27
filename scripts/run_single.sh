#!/bin/bash
# =============================================================================
# Single Experiment Runner
# =============================================================================
# Runs a single experiment and appends results to CSV.
# Usage: ./run_single.sh <args_file>
# Or source config.sh and call run_experiment directly
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# -----------------------------------------------------------------------------
# Resume Logic: Check if experiment already completed successfully
# -----------------------------------------------------------------------------
# Set FORCE_RERUN=1 to disable resume and re-run all experiments
# Example: FORCE_RERUN=1 ./launch_all.sh --group 1

is_experiment_completed() {
    # If FORCE_RERUN is set, always return false (run everything)
    if [ "${FORCE_RERUN:-0}" = "1" ]; then
        return 1
    fi
    local results_csv=$1
    local experiment_group=$2
    local model=$3
    local dataset=$4
    local pred_len=$5
    local seed=$6
    local data_mode=$7
    local synth_ratio=$8
    local sparsity=$9
    local difficulty_mode=${10}
    local bundle=${11}
    local p_latent_factor=${12}
    local anneal_strategy=${13}
    local anneal_epoch=${14}

    # If CSV doesn't exist, experiment not completed
    if [ ! -f "${results_csv}" ]; then
        return 1
    fi

    # Build pattern to match this exact experiment configuration
    # CSV format: experiment_group,model,dataset,pred_len,seed,data_mode,synth_ratio,sparsity,difficulty_mode,bundle,p_latent_factor,anneal_strategy,anneal_epoch,mse,mae,train_time,timestamp
    local pattern="^${experiment_group},${model},${dataset},${pred_len},${seed},${data_mode},${synth_ratio},${sparsity},${difficulty_mode},${bundle},${p_latent_factor},${anneal_strategy},${anneal_epoch},"

    # Check if pattern exists and MSE is not NA (field 14)
    # Using awk to verify MSE field is a valid number
    local match=$(grep -E "${pattern}" "${results_csv}" | awk -F',' '$14 != "NA" && $14 ~ /^[0-9.]+$/ {print; exit}')

    if [ -n "${match}" ]; then
        return 0  # Experiment completed successfully
    else
        return 1  # Not completed or failed (NA)
    fi
}

run_experiment() {
    # Arguments (all required)
    local experiment_group=$1
    local model=$2
    local dataset_str=$3
    local pred_len=$4
    local seed=$5
    local data_mode=$6
    local synth_ratio=$7
    local sparsity=$8
    local difficulty_mode=$9
    local bundle=${10}
    local p_latent_factor=${11}
    local anneal_strategy=${12}
    local anneal_epoch=${13}
    local results_csv=${14}

    # Decode cache_size encoded in bundle field (Group 9: bundle="cache0"/"cache100"/"cache500")
    # Preserve original bundle value for resume check and CSV recording
    local bundle_key="${bundle}"
    local cache_size=500
    if [[ "${bundle}" == cache* ]]; then
        cache_size="${bundle#cache}"
        bundle="all"
    fi

    # Parse dataset
    parse_dataset "${dataset_str}"

    # Generate experiment ID
    local exp_id=$(get_experiment_id "${model}" "${DS_NAME}" "${pred_len}" "${seed}" "${experiment_group}")

    # Resume logic: skip if already completed successfully
    if is_experiment_completed "${results_csv}" "${experiment_group}" "${model}" "${DS_NAME}" \
                               "${pred_len}" "${seed}" "${data_mode}" "${synth_ratio}" "${sparsity}" \
                               "${difficulty_mode}" "${bundle_key}" "${p_latent_factor}" \
                               "${anneal_strategy}" "${anneal_epoch}"; then
        log_experiment "SKIPPED (already completed): ${exp_id}"
        return 0
    fi

    log_experiment "Starting: ${exp_id}"

    # Get model-specific arguments
    local model_args=$(get_model_args "${model}" "${DS_ENC_IN}" "${pred_len}")

    # Build command
    local cmd="python -u ${PROJECT_ROOT}/run.py"
    cmd+=" --task_name long_term_forecast"
    cmd+=" --is_training 1"
    cmd+=" --root_path ${DS_ROOT}"
    cmd+=" --data_path ${DS_PATH}"
    cmd+=" --model_id ${exp_id}"
    cmd+=" --model ${model}"
    cmd+=" --data ${DS_DATA_TYPE}"
    cmd+=" --features M"
    cmd+=" --seq_len 96"
    cmd+=" --label_len 48"
    cmd+=" --pred_len ${pred_len}"
    cmd+=" --train_epochs ${TRAIN_EPOCHS}"
    cmd+=" --batch_size ${BATCH_SIZE}"
    cmd+=" --learning_rate ${LEARNING_RATE}"
    cmd+=" --num_workers ${NUM_WORKERS}"
    cmd+=" --freq ${DS_FREQ}"
    cmd+=" --seed ${seed}"
    cmd+=" --use_gpu True"
    cmd+=" --checkpoints ${CHECKPOINTS_DIR}"
    cmd+=" ${model_args}"

    # Add synthetic data parameters
    cmd+=" --data_mode ${data_mode}"

    if [ "${data_mode}" != "real" ]; then
        cmd+=" --synth_ratio ${synth_ratio}"
        cmd+=" --synth_difficulty ${difficulty_mode}"
        cmd+=" --synth_p_latent_factor ${p_latent_factor}"

        if [ "${bundle}" != "all" ]; then
            cmd+=" --synth_bundle ${bundle}"
        fi
    fi

    cmd+=" --sparsity ${sparsity}"
    cmd+=" --synth_cache_size ${cache_size}"

    if [ "${data_mode}" == "anneal" ] || [ "${data_mode}" == "anneali" ]; then
        cmd+=" --anneal_strategy ${anneal_strategy}"
        cmd+=" --anneal_epoch ${anneal_epoch}"
    fi

    # Run and capture output — include PID to avoid race when two partitions run same job
    local log_file="${LOGS_DIR}/${exp_id}_$$.log"
    local start_time=$(date +%s)

    ${cmd} > "${log_file}" 2>&1
    local exit_code=$?

    local end_time=$(date +%s)
    local train_time=$((end_time - start_time))

    # Extract metrics from log
    local mse="NA"
    local mae="NA"

    if [ ${exit_code} -eq 0 ]; then
        # Parse MSE and MAE from the log file
        # Expected format: "mse:0.xxx, mae:0.xxx"
        mse=$(grep -oP 'mse:\K[0-9.]+' "${log_file}" | tail -1)
        mae=$(grep -oP 'mae:\K[0-9.]+' "${log_file}" | tail -1)

        if [ -z "${mse}" ]; then
            mse="NA"
        fi
        if [ -z "${mae}" ]; then
            mae="NA"
        fi

        log_experiment "Completed: ${exp_id} | MSE: ${mse} | MAE: ${mae} | Time: ${train_time}s"
    else
        log_experiment "FAILED: ${exp_id} | Exit code: ${exit_code}"
    fi

    # Clean up checkpoint to save storage (metrics already extracted)
    rm -rf "${CHECKPOINTS_DIR}/${exp_id}"

    # Append to results CSV
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "${experiment_group},${model},${DS_NAME},${pred_len},${seed},${data_mode},${synth_ratio},${sparsity},${difficulty_mode},${bundle_key},${p_latent_factor},${anneal_strategy},${anneal_epoch},${mse},${mae},${train_time},${timestamp}" >> "${results_csv}"
}

# If called directly with args file
if [ "$1" != "" ] && [ -f "$1" ]; then
    source "$1"
    run_experiment "${EXPERIMENT_GROUP}" "${MODEL}" "${DATASET_STR}" "${PRED_LEN}" "${SEED}" \
                   "${DATA_MODE}" "${SYNTH_RATIO}" "${SPARSITY}" "${DIFFICULTY_MODE}" "${BUNDLE}" \
                   "${P_LATENT_FACTOR}" "${ANNEAL_STRATEGY}" "${ANNEAL_EPOCH}" "${RESULTS_CSV}"
fi
