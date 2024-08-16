#!/bin/bash
#
# Batch script for submitting UVW reordering jobs on CSD3
# Adjust requested resources and variables as necessary
#
#SBATCH --job-name=uvw_reorder
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:05:00
#SBATCH --partition=icelake
#SBATCH --account=SKA-SDHP-SL2-CPU
#SBATCH --signal=B:TERM@90

###############################################################################
# Variables to adjust
###############################################################################
INPUT_DIR="/home/hpcmore1/rds/rds-sdhp-S7lLL7eOZIg/hpcmore1/PI23/uvw_reorder_tests/output_0064ch_4.0h"

# Just to save dask logs and reports
OUTPUT_DIR="/home/hpcmore1/rds/rds-sdhp-S7lLL7eOZIg/hpcmore1/PI23/uvw_reorder_tests/output_0064ch_4.0h/read_test"

# Activation command for the Python env in which the pipeline is installed
# You may have to replace this with the equivalent command
# for the virtualenv manager you are using
VENV_ACTIVATION_COMMAND="source ${HOME}/python_venvs/cip/bin/activate"

# Dask options
DASK_WORKERS_PER_NODE=16

###############################################################################
# Anything below should not be edited
###############################################################################

# Load required base modules for icelake + recent Python
module purge
module load rhel8/default-icl
module load python/3.11.0-icl

set -x

# Immediately ensure output dir exists
mkdir -p ${OUTPUT_DIR}

# Fetch list of nodes
NODES=($(scontrol show hostnames))
HEAD_NODE="$(hostname)"

echo "Allocated nodes: ${NODES[*]}"
echo "Head node: $HEAD_NODE"

DASK_SCHEDULER_PORT=8786
DASK_SCHEDULER_ADDRESS="${HEAD_NODE}:${DASK_SCHEDULER_PORT}"
DASK_RESOURCES_ARGUMENT="--resources processing_slots=1"
DASK_WORKER_COMMAND="DASK_DISTRIBUTED__WORKER__DAEMON=False dask worker ${DASK_SCHEDULER_ADDRESS} --nworkers ${DASK_WORKERS_PER_NODE} ${DASK_RESOURCES_ARGUMENT}"
DASK_LOGS_DIR=${OUTPUT_DIR}

##### Launch dask scheduler on head node #####
${VENV_ACTIVATION_COMMAND}
dask scheduler --port ${DASK_SCHEDULER_PORT} >$DASK_LOGS_DIR/scheduler_$HEAD_NODE.log 2>&1 &
echo "Started dask scheduler on $DASK_SCHEDULER_ADDRESS"

##### Start dask workers on all nodes via ssh (including on head node, it's fine) #####
# We need to make a shell script to:
# 1. Activate the same python environment on all nodes
# 2. Launch the dask workers
DASK_WORKER_LAUNCH_SCRIPT=${OUTPUT_DIR}/launch_dask_worker.sh
echo "#!/bin/bash" >$DASK_WORKER_LAUNCH_SCRIPT
echo $VENV_ACTIVATION_COMMAND >>$DASK_WORKER_LAUNCH_SCRIPT
echo $DASK_WORKER_COMMAND >>$DASK_WORKER_LAUNCH_SCRIPT
chmod u+x $DASK_WORKER_LAUNCH_SCRIPT

for node in "${NODES[@]}"; do
    logfile=$DASK_LOGS_DIR/worker_$node.log
    echo "Starting dask worker on $node"
    ssh $node ${DASK_WORKER_LAUNCH_SCRIPT} >$logfile 2>&1 &
done

echo "Waiting for workers to start"
sleep 60

### Launch program
# Execute from output dir, because app saves dask reports in the working
# directory
cd ${OUTPUT_DIR}
ska-sdp-cip-read-tile-chunks ${INPUT_DIR} -d ${DASK_SCHEDULER_ADDRESS}
