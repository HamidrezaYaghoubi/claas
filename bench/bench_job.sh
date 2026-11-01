#!/bin/bash
#SBATCH --job-name=claas-bench
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --gres=gpu:rtxa6000:1   # baseline loads two 12.8 GB models; the demo was deployed on an A6000 (src/sbatch/demo_llm.sh)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH --output=/fs/nexus-projects/pc_driving/yaghoubi/claas/bench/logs/%x.out.%j
#SBATCH --error=/fs/nexus-projects/pc_driving/yaghoubi/claas/bench/logs/%x.err.%j

# Latency / GPU-memory benchmark of CLAAS inference (demo request + eval sample).
# Usage: sbatch bench/bench_job.sh
# Real-model run: CLAAS_ADAPTER_12B=/path/to/adapter RES=$PWD/bench/results_real EXTRA=--natural_len \
#                 STEPS="demo_asis demo_onemodel" sbatch bench/bench_job.sh
set -uo pipefail

BENCH=/fs/nexus-projects/pc_driving/yaghoubi/claas/bench
ENV=/fs/nexus-scratch/yaghoubi/anaconda3/envs/claas
PY=$ENV/bin/python
RES=${RES:-$BENCH/results}
EXTRA=${EXTRA:-}  # e.g. EXTRA=--natural_len to let outputs stop where the model stops
STEPS=${STEPS:-"eval_asis eval_prefix eval_4b_prefix demo_asis demo_onemodel demo_onemodel_prefix demo_4b_prefix"}

export HF_HOME=/fs/nexus-projects/pc_driving/yaghoubi/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_XET_CACHE=$HF_HOME/xet
export HF_ASSETS_CACHE=$HF_HOME/assets
# not offline: unsloth 2025.3.19 calls HfFileSystem.glob() on the repo while loading (weights are already cached)
export TRITON_CACHE_DIR=/tmp/$USER/triton-$SLURM_JOB_ID
export TMPDIR=/fs/nexus-scratch/yaghoubi/tmp
export PYTHONUNBUFFERED=1
module load gcc/11.2.0 2>/dev/null || true

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "=== steps: $STEPS"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv
$PY -c "import torch,transformers,unsloth;print('torch',torch.__version__,'cuda',torch.version.cuda,'| transformers',transformers.__version__,'| unsloth',unsloth.__version__)"
$ENV/bin/pip list 2>/dev/null | grep -i -E "^(unsloth_zoo|bitsandbytes|peft|trl|xformers|triton|accelerate) "

cd $BENCH  # unsloth writes unsloth_compiled_cache/ into the cwd
mkdir -p $RES

echo "=== [adapters] $(date)"
$PY make_adapters.py unsloth/gemma-3-12b-it unsloth/gemma-3-4b-it 2>&1 | tail -40

for step in $STEPS; do
    wl=${step%%_*}; var=${step#*_}
    echo "=== [$step] start $(date)"
    rm -f $RES/${step}.json  # a failed step must not leave an older result behind
    $PY bench_claas.py --workload $wl --variant $var --seeds 2 --out $RES/${step}.json $EXTRA \
        > $BENCH/logs/${step}.$SLURM_JOB_ID.log 2>&1
    echo "=== [$step] exit=$? end $(date)"
    tail -3 $BENCH/logs/${step}.$SLURM_JOB_ID.log
done

echo "=== summary"
$PY summarize.py $RES
echo "=== done $(date)"
