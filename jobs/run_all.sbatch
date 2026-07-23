#!/bin/bash
# Full pipeline (Stage 0-11) on a GPU node.  Submit from the repo root:
#   sbatch jobs/run_all.sbatch
#SBATCH --job-name=if-runall
#SBATCH --partition=gen_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=AlmaLinux9   # venv is alma9-built; an AlmaLinux8 GPU node dies on GLIBC_2.34
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=jobs/logs/runall_%j.out
#SBATCH --error=jobs/logs/runall_%j.err

set -euo pipefail
REPO="/hpf/projects/msalter/sghazis/Imaging/ED/For Shahrzad/if-pipeline"
cd "$REPO"
module load python/3.11.15
source .venv/bin/activate

echo "=== node: $(hostname) ==="
python -c "import torch; print('cuda.is_available =', torch.cuda.is_available())"
python run_all.py --config config/config.yaml
echo "=== RUN_ALL JOB DONE ==="
