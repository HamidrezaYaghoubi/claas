#!/bin/bash
#SBATCH --job-name=claas-step0-control
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --gres=gpu:rtxa6000:1   # the as-is app needs >24 GB (two 12B models)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=01:00:00
#SBATCH --output=/fs/nexus-projects/pc_driving/yaghoubi/claas/bench/logs/%x.out.%j
#SBATCH --error=/fs/nexus-projects/pc_driving/yaghoubi/claas/bench/logs/%x.err.%j

# Step 0 of plan.md: real trained adapter, real app code/prompts/posters/questions, natural lengths.
# as-is app vs one-model app on the same GPU. Shared project files are only read (checked below).
set -uo pipefail

BENCH=/fs/nexus-projects/pc_driving/yaghoubi/claas/bench
SHARED=/fs/nexus-projects/health_sim_ai
PY=/fs/nexus-scratch/yaghoubi/anaconda3/envs/claas/bin/python
RES=$BENCH/results_step0
LOGS=$BENCH/logs

export HF_HOME=/fs/nexus-projects/pc_driving/yaghoubi/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub HF_XET_CACHE=$HF_HOME/xet HF_ASSETS_CACHE=$HF_HOME/assets
export TRITON_CACHE_DIR=/tmp/$USER/triton-$SLURM_JOB_ID
export TMPDIR=/fs/nexus-scratch/yaghoubi/tmp
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
module load gcc/11.2.0 2>/dev/null || true

echo "=== job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv

checksums() {
    (cd $SHARED && sha256sum src/unsloth/gemma-3-12b-it_task1_1_epochs_test_neutral_partialTraits/* \
        src/demo_task1_sph.py src/utils.py src/configs/prompts.yaml src/configs/task1_demo_sph.yaml \
        data/survey_responses_screened.csv stimuli/{COPD1.1,HD1,HIV1.1,MH1.1,N2.1,SA4.1,SP7.1,V7.1,CF1.1}.png)
}
mkdir -p $RES
MARK=$LOGS/step0_start.$SLURM_JOB_ID
touch $MARK
checksums > $LOGS/step0_shared_sha256_before.$SLURM_JOB_ID

cd $BENCH  # unsloth writes unsloth_compiled_cache/ into the cwd
echo "=== [control] start $(date)"
$PY step0_control.py > $LOGS/step0_control.$SLURM_JOB_ID.log 2>&1
echo "=== [control] exit=$? end $(date)"
grep "^{" $LOGS/step0_control.$SLURM_JOB_ID.log

echo "=== shared files unchanged?"
checksums > $LOGS/step0_shared_sha256_after.$SLURM_JOB_ID
if cmp -s $LOGS/step0_shared_sha256_before.$SLURM_JOB_ID $LOGS/step0_shared_sha256_after.$SLURM_JOB_ID; then
    echo "checksums identical"
else
    echo "CHECKSUMS DIFFER"
fi
echo "files modified in shared src/data/stimuli during the job:"
find $SHARED/src $SHARED/data $SHARED/stimuli -newer $MARK -print 2>/dev/null | head -20
echo "(end of list)"

echo "=== done $(date)"
