#!/bin/bash
# =============================================================================
# Generate Parallel Jobs for Cluster Execution
# =============================================================================
# Creates individual job files that can be executed across nodes.
# Compatible with SLURM, PBS, or GNU parallel.
#
# Usage:
#   ./generate_parallel_jobs.sh [groups]     # Default: all groups
#   ./generate_parallel_jobs.sh 1,2,3        # Specific groups
#
# Output:
#   jobs/job_XXXX.sh files - individual experiment jobs
#   jobs/all_jobs.txt      - list of all job files for GNU parallel
#   jobs/slurm_array.sh    - SLURM array job script
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

JOBS_DIR="${PROJECT_ROOT}/jobs"
mkdir -p "${JOBS_DIR}"

# Parse groups argument
# Note: GROUPS is a reserved bash variable (user's GIDs), so use INPUT_GROUPS instead
INPUT_GROUPS="${1:-1,2,3,4,5,6,7,8,9}"
IFS=',' read -ra GROUP_ARRAY <<< "${INPUT_GROUPS}"

# Counter for job IDs
JOB_ID=0
ALL_JOBS_FILE="${JOBS_DIR}/all_jobs.txt"
> "${ALL_JOBS_FILE}"

log_experiment "Generating parallel jobs for groups: ${INPUT_GROUPS}"

# Function to generate a single job file
generate_job() {
    local job_id=$1
    local experiment_group=$2
    local model=$3
    local dataset_str=$4
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
    local results_csv=${15}

    local job_file="${JOBS_DIR}/job_$(printf '%04d' ${job_id}).sh"

    cat > "${job_file}" << EOF
#!/bin/bash
# Job ${job_id}: ${experiment_group} - ${model} - ${dataset_str%%:*}
# CUDA_VISIBLE_DEVICES is managed by SLURM via --gres=gpu:1

SCRIPT_DIR="${SCRIPT_DIR}"
source "\${SCRIPT_DIR}/config.sh"
source "\${SCRIPT_DIR}/run_single.sh"

run_experiment "${experiment_group}" "${model}" "${dataset_str}" "${pred_len}" "${seed}" \\
               "${data_mode}" "${synth_ratio}" "${sparsity}" "${difficulty_mode}" "${bundle}" \\
               "${p_latent_factor}" "${anneal_strategy}" "${anneal_epoch}" "${results_csv}"
EOF

    chmod +x "${job_file}"
    echo "${job_file}" >> "${ALL_JOBS_FILE}"
}

# -----------------------------------------------------------------------------
# Group 1: Baseline
# -----------------------------------------------------------------------------
if [[ " ${GROUP_ARRAY[*]} " =~ " 1 " ]]; then
    RESULTS_CSV="${RESULTS_DIR}/group1_baseline.csv"
    init_results_csv "${RESULTS_CSV}"

    for model in ${MODELS}; do
        for dataset_str in ${DATASETS_FULL}; do
            [ -z "${dataset_str}" ] && continue
            for pred_len in ${PRED_LENS}; do
                for seed in ${SEEDS_PHASE1}; do
                    generate_job $((++JOB_ID)) "baseline" "${model}" "${dataset_str}" "${pred_len}" "${seed}" \
                                 "real" "0.0" "1.0" "none" "all" "0.0" "none" "0" "${RESULTS_CSV}"
                done
            done
        done
    done
fi

# -----------------------------------------------------------------------------
# Group 2: Sparsity
# -----------------------------------------------------------------------------
if [[ " ${GROUP_ARRAY[*]} " =~ " 2 " ]]; then
    RESULTS_CSV="${RESULTS_DIR}/group2_sparsity.csv"
    init_results_csv "${RESULTS_CSV}"

    for model in ${MODELS}; do
        for dataset_str in ${DATASETS_FULL}; do
            [ -z "${dataset_str}" ] && continue
            for pred_len in ${PRED_LENS}; do
                for sparsity in ${SPARSITY_LEVELS}; do
                    for seed in ${SEEDS_PHASE1}; do
                        generate_job $((++JOB_ID)) "sparsity" "${model}" "${dataset_str}" "${pred_len}" "${seed}" \
                                     "real" "0.0" "${sparsity}" "none" "all" "0.0" "none" "0" "${RESULTS_CSV}"
                    done
                done
            done
        done
    done
fi

# -----------------------------------------------------------------------------
# Group 3: Augmentation
# -----------------------------------------------------------------------------
if [[ " ${GROUP_ARRAY[*]} " =~ " 3 " ]]; then
    RESULTS_CSV="${RESULTS_DIR}/group3_augmentation.csv"
    init_results_csv "${RESULTS_CSV}"

    for model in ${MODELS}; do
        for dataset_str in ${DATASETS_FULL}; do
            [ -z "${dataset_str}" ] && continue
            for pred_len in ${PRED_LENS}; do
                for synth_ratio in ${SYNTH_RATIOS}; do
                    for seed in ${SEEDS_PHASE1}; do
                        generate_job $((++JOB_ID)) "augmentation" "${model}" "${dataset_str}" "${pred_len}" "${seed}" \
                                     "mixed" "${synth_ratio}" "1.0" "uniform" "all" "0.0" "none" "0" "${RESULTS_CSV}"
                    done
                done
            done
        done
    done
fi

# -----------------------------------------------------------------------------
# Group 4: Low-Resource
# -----------------------------------------------------------------------------
if [[ " ${GROUP_ARRAY[*]} " =~ " 4 " ]]; then
    RESULTS_CSV="${RESULTS_DIR}/group4_lowresource.csv"
    init_results_csv "${RESULTS_CSV}"

    SPARSITY_LOWRES="0.1 0.25"
    for model in ${MODELS}; do
        for dataset_str in ${DATASETS_ABLATION}; do
            [ -z "${dataset_str}" ] && continue
            for pred_len in ${PRED_LENS}; do
                for sparsity in ${SPARSITY_LOWRES}; do
                    for synth_ratio in ${SYNTH_RATIOS_LOWRES}; do
                        for seed in ${SEEDS_PHASE1}; do
                            generate_job $((++JOB_ID)) "lowresource" "${model}" "${dataset_str}" "${pred_len}" "${seed}" \
                                         "mixed" "${synth_ratio}" "${sparsity}" "uniform" "all" "0.0" "none" "0" "${RESULTS_CSV}"
                        done
                    done
                done
            done
        done
    done
fi

# -----------------------------------------------------------------------------
# Group 5: Difficulty
# -----------------------------------------------------------------------------
if [[ " ${GROUP_ARRAY[*]} " =~ " 5 " ]]; then
    RESULTS_CSV="${RESULTS_DIR}/group5_difficulty.csv"
    init_results_csv "${RESULTS_CSV}"

    for model in ${MODELS}; do
        for dataset_str in ${DATASETS_ABLATION}; do
            [ -z "${dataset_str}" ] && continue
            for pred_len in ${PRED_LENS}; do
                for difficulty in ${DIFFICULTY_MODES}; do
                    for seed in ${SEEDS_PHASE1}; do
                        generate_job $((++JOB_ID)) "difficulty" "${model}" "${dataset_str}" "${pred_len}" "${seed}" \
                                     "mixed" "1.0" "1.0" "${difficulty}" "all" "0.0" "none" "0" "${RESULTS_CSV}"
                    done
                done
            done
        done
    done
fi

# -----------------------------------------------------------------------------
# Group 6: Bundle
# -----------------------------------------------------------------------------
if [[ " ${GROUP_ARRAY[*]} " =~ " 6 " ]]; then
    RESULTS_CSV="${RESULTS_DIR}/group6_bundle.csv"
    init_results_csv "${RESULTS_CSV}"

    for model in ${MODELS}; do
        for dataset_str in ${DATASETS_ABLATION}; do
            [ -z "${dataset_str}" ] && continue
            for pred_len in ${PRED_LENS}; do
                for bundle in ${BUNDLE_TYPES}; do
                    for seed in ${SEEDS_PHASE1}; do
                        generate_job $((++JOB_ID)) "bundle" "${model}" "${dataset_str}" "${pred_len}" "${seed}" \
                                     "mixed" "1.0" "1.0" "uniform" "${bundle}" "0.0" "none" "0" "${RESULTS_CSV}"
                    done
                done
            done
        done
    done
fi

# -----------------------------------------------------------------------------
# Group 7: Curriculum
# -----------------------------------------------------------------------------
if [[ " ${GROUP_ARRAY[*]} " =~ " 7 " ]]; then
    RESULTS_CSV="${RESULTS_DIR}/group7_curriculum.csv"
    init_results_csv "${RESULTS_CSV}"

    for model in ${MODELS}; do
        for dataset_str in ${DATASETS_ABLATION}; do
            [ -z "${dataset_str}" ] && continue
            for pred_len in ${PRED_LENS}; do
                for anneal_mode in ${ANNEAL_MODES}; do
                    for anneal_strategy in ${ANNEAL_STRATEGIES}; do
                        for seed in ${SEEDS_PHASE1}; do
                            generate_job $((++JOB_ID)) "curriculum" "${model}" "${dataset_str}" "${pred_len}" "${seed}" \
                                         "${anneal_mode}" "1.0" "1.0" "uniform" "all" "0.0" "${anneal_strategy}" "5" "${RESULTS_CSV}"
                        done
                    done
                done
            done
        done
    done
fi

# -----------------------------------------------------------------------------
# Group 8: Latent
# -----------------------------------------------------------------------------
if [[ " ${GROUP_ARRAY[*]} " =~ " 8 " ]]; then
    RESULTS_CSV="${RESULTS_DIR}/group8_latent.csv"
    init_results_csv "${RESULTS_CSV}"

    for model in ${MODELS}; do
        for dataset_str in ${DATASETS_ABLATION}; do
            [ -z "${dataset_str}" ] && continue
            for pred_len in ${PRED_LENS}; do
                for p_latent in ${LATENT_FACTORS}; do
                    for seed in ${SEEDS_PHASE1}; do
                        generate_job $((++JOB_ID)) "latent" "${model}" "${dataset_str}" "${pred_len}" "${seed}" \
                                     "mixed" "1.0" "1.0" "uniform" "all" "${p_latent}" "none" "0" "${RESULTS_CSV}"
                    done
                done
            done
        done
    done
fi

# -----------------------------------------------------------------------------
# Group 9: Cache Ablation (on-the-fly vs fixed cache)
# Encodes cache_size in the bundle field: "cache0", "cache100", "cache500"
# Models: PatchTST + iTransformer; Datasets: ETTh1, ETTh2, ETTm1 (small, feasible for on-the-fly)
# 2 models x 3 datasets x 3 cache sizes x 2 pred_lens x 3 seeds = 108 jobs
# -----------------------------------------------------------------------------
if [[ " ${GROUP_ARRAY[*]} " =~ " 9 " ]]; then
    RESULTS_CSV="${RESULTS_DIR}/group9_cacheablation.csv"
    init_results_csv "${RESULTS_CSV}"

    MODELS_G9="PatchTST iTransformer"
    DATASETS_G9="
ETTh1:./dataset/ETT-small:ETTh1.csv:7:h:ETTh1
ETTm1:./dataset/ETT-small:ETTm1.csv:7:t:ETTm1
Weather:./dataset/weather:weather.csv:21:h:custom
"
    CACHE_SIZES="0 100 500"

    for model in ${MODELS_G9}; do
        for dataset_str in ${DATASETS_G9}; do
            [ -z "${dataset_str}" ] && continue
            for pred_len in ${PRED_LENS}; do
                for cache_size in ${CACHE_SIZES}; do
                    for seed in ${SEEDS_PHASE1}; do
                        generate_job $((++JOB_ID)) "cacheablation" "${model}" "${dataset_str}" "${pred_len}" "${seed}" \
                                     "mixed" "1.0" "1.0" "uniform" "cache${cache_size}" "0.0" "none" "0" "${RESULTS_CSV}"
                    done
                done
            done
        done
    done
fi

log_experiment "Generated ${JOB_ID} job files in ${JOBS_DIR}"

# -----------------------------------------------------------------------------
# Generate SLURM array job scripts (chunked to respect MaxArraySize=1001)
# -----------------------------------------------------------------------------
CHUNK_SIZE=1000
CHUNK_NUM=0
SUBMIT_SCRIPT="${JOBS_DIR}/submit_all.sh"
echo "#!/bin/bash" > "${SUBMIT_SCRIPT}"
echo "# Submit all chunked SLURM arrays" >> "${SUBMIT_SCRIPT}"

for (( chunk_start=1; chunk_start<=JOB_ID; chunk_start+=CHUNK_SIZE )); do
    chunk_end=$(( chunk_start + CHUNK_SIZE - 1 ))
    if [ ${chunk_end} -gt ${JOB_ID} ]; then
        chunk_end=${JOB_ID}
    fi
    chunk_local_end=$(( chunk_end - chunk_start + 1 ))
    chunk_offset=$(( chunk_start - 1 ))
    CHUNK_NUM=$(( CHUNK_NUM + 1 ))
    chunk_file="${JOBS_DIR}/slurm_array_${CHUNK_NUM}.sh"

    cat > "${chunk_file}" << EOF
#!/bin/bash
#SBATCH --job-name=synth_ts_${CHUNK_NUM}
#SBATCH --output=${LOGS_DIR}/slurm_%A_%a.out
#SBATCH --error=${LOGS_DIR}/slurm_%A_%a.err
#SBATCH --array=1-${chunk_local_end}%48
#SBATCH --time=06:00:00
#SBATCH --partition=dp-dam
#SBATCH -A joaiml
#SBATCH --exclude=ml-gpu02,dp-dam16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Load modules and activate venv (module may not exist on all partitions — venv works regardless)
module load GCCcore/13.3.0 2>/dev/null || true
module load Python/3.12.3 2>/dev/null || true
source /p/scratch/joaiml/cazaux1/venv/bin/activate

# Offset maps local task ID (1-${chunk_local_end}) to global line in all_jobs.txt
LINE=\$(( SLURM_ARRAY_TASK_ID + ${chunk_offset} ))
JOB_FILE=\$(sed -n "\${LINE}p" ${JOBS_DIR}/all_jobs.txt)

if [ -f "\${JOB_FILE}" ]; then
    bash "\${JOB_FILE}"
else
    echo "Job file not found: \${JOB_FILE}"
    exit 1
fi
EOF

    echo "sbatch \${PARTITION_ARGS} ${chunk_file}" >> "${SUBMIT_SCRIPT}"
    log_experiment "Generated SLURM array script (chunk ${CHUNK_NUM}, jobs ${chunk_start}-${chunk_end}): ${chunk_file}"
done

chmod +x "${SUBMIT_SCRIPT}"
log_experiment "Generated submit script: ${SUBMIT_SCRIPT}"

# -----------------------------------------------------------------------------
# Generate GNU parallel runner
# -----------------------------------------------------------------------------
cat > "${JOBS_DIR}/run_parallel.sh" << 'EOF'
#!/bin/bash
# Run all jobs using GNU parallel
# Adjust -j to match available GPUs/nodes

JOBS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Option 1: Run with N parallel jobs locally
# parallel -j 4 bash {} < "${JOBS_DIR}/all_jobs.txt"

# Option 2: Run on multiple nodes via SSH (edit hostfile)
# parallel -j 1 --sshloginfile hostfile.txt bash {} < "${JOBS_DIR}/all_jobs.txt"

# Option 3: Simple sequential fallback
echo "Running ${JOB_COUNT} jobs..."
while read job_file; do
    bash "${job_file}"
done < "${JOBS_DIR}/all_jobs.txt"
EOF

sed -i "s/JOB_COUNT/${JOB_ID}/" "${JOBS_DIR}/run_parallel.sh"
chmod +x "${JOBS_DIR}/run_parallel.sh"

log_experiment "Generated parallel runner: ${JOBS_DIR}/run_parallel.sh"
log_experiment ""
log_experiment "To run with SLURM:"
log_experiment "  sbatch ${JOBS_DIR}/slurm_array.sh"
log_experiment ""
log_experiment "To run with GNU parallel (4 jobs):"
log_experiment "  parallel -j 4 bash {} < ${JOBS_DIR}/all_jobs.txt"
log_experiment ""
log_experiment "Total jobs: ${JOB_ID}"
