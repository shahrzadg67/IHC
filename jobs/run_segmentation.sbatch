#!/bin/bash
# Stage 3 + Stage 4 on a GPU node (SickKids hpf / Slurm).
# Submit from the repo root:  sbatch jobs/run_segmentation.sbatch
#SBATCH --job-name=if-segment
#SBATCH --partition=gen_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=AlmaLinux9   # venv is alma9-built; an AlmaLinux8 GPU node dies on GLIBC_2.34
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=jobs/logs/segment_%j.out
#SBATCH --error=jobs/logs/segment_%j.err

set -euo pipefail
REPO="/hpf/projects/msalter/sghazis/Imaging/ED/For Shahrzad/if-pipeline"
cd "$REPO"

module load python/3.11.15          # needed for libpython on the venv interpreter
source .venv/bin/activate
# torch cu124 wheels bundle their own CUDA runtime; the node's GPU driver is used directly.

echo "=== node: $(hostname) ==="
nvidia-smi -L || echo "(nvidia-smi not found)"
python -c "import torch; print('torch', torch.__version__, '| cuda.is_available =', torch.cuda.is_available())"
echo

python -m src.s3_segment_nuclei --config config/config.yaml
echo
python -m src.s4_cell_bodies   --config config/config.yaml
echo "=== SEGMENTATION JOB DONE ==="
