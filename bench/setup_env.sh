#!/bin/bash
# Builds the CLAAS env with the original pins (requirements.txt: torch 2.6.0, transformers 4.50.0,
# unsloth 2025.3.19, unsloth_zoo 2025.3.17) + era-matched deps for unsloth 2025.3.x.
set -euxo pipefail
export TMPDIR=/fs/nexus-scratch/yaghoubi/tmp
ENV=/fs/nexus-scratch/yaghoubi/anaconda3/envs/claas
source /fs/nexus-scratch/yaghoubi/anaconda3/etc/profile.d/conda.sh
[ -d "$ENV" ] || conda create -y -p "$ENV" python=3.12
PIP="$ENV/bin/pip install --no-cache-dir"
$PIP torch==2.6.0 torchvision==0.21.0 xformers==0.0.29.post3
$PIP "unsloth==2025.3.19" "unsloth_zoo==2025.3.17" "transformers==4.50.0" "trl==0.15.2" \
     "peft==0.14.0" "accelerate==1.5.2" "bitsandbytes==0.45.3" "datasets==3.4.1" \
     "pillow==11.2.1" pandas pyyaml termcolor nvidia-ml-py hf_transfer
$ENV/bin/pip list | grep -i -E "^(torch|torchvision|xformers|transformers|unsloth|unsloth_zoo|trl|peft|accelerate|bitsandbytes|triton|datasets) "
