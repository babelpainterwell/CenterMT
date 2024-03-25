#!/bin/bash
#SBATCH --job-name=matlab_sim_multi_cpu
#SBATCH --partition=short
#SBATCH --time=72:00:00
#SBATCH --array=1-50%50
#SBATCH --cpus-per-task=1  # Request 1 CPU core per task
#SBATCH -o /path/to/logs/output_%A_%a.txt  # %A is job ID, %a is array task ID
#SBATCH -e /path/to/logs/error_%A_%a.txt   # Standard error file
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G 

module load MATLAB/2023a

# Determine the unique directory for this array job
SIM_DIR="/Curved_branch/simulation_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${SIM_DIR}"

# Define unique directories for samples and labels for this task
export SAMPLES_DIR="${SIM_DIR}/samples"
export LABELS_DIR="${SIM_DIR}/labels"
mkdir -p "${SAMPLES_DIR}"
mkdir -p "${LABELS_DIR}"

# Function to run MATLAB simulation
run_matlab() {
    matlab -nodesktop -nosplash -r "try, sqrsum, catch ME, disp(ME.message), end, exit"
}

# Loop to repeat the MATLAB simulation and restart every 3600 seconds
SECONDS=0
MAX_DURATION=$((72 * 3600))  # 72 hours in seconds
RESTART_INTERVAL=3600  # Restart every 3600 seconds (1 hour)
REPEAT_COUNT=0
MAX_REPEATS=100  # Maximum number of times to repeat the MATLAB simulation

while [ "$SECONDS" -lt "$MAX_DURATION" ] && [ "$REPEAT_COUNT" -lt "$MAX_REPEATS" ]; do
    START_TIME=$SECONDS 
    run_matlab
    END_TIME=$SECONDS
    DURATION=$((END_TIME - START_TIME))
    
    # Increment the repeat counter
    REPEAT_COUNT=$((REPEAT_COUNT + 1))
    
    # Calculate remaining time before the next restart
    SLEEP_TIME=$((RESTART_INTERVAL - DURATION))
    if [ "$SLEEP_TIME" -gt 0 ]; then
        sleep "$SLEEP_TIME"
    fi
done




