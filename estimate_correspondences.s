#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --time=5:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=salsa-train
#SBATCH --mail-type=END
#SBATCH --mail-user=ekp252@nyu.edu
#SBATCH --output=run_est_corr.out
#SBATCH --error=error_est_corr.err
#SBATCH --account=torch_pr_1026_general
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh;
conda activate /scratch/ekp252/conda_envs/salsa
export PATH=/scratch/ekp252/conda_envs/salsa/bin:$PATH;
hash -r
cd /scratch/ekp252/salsa

# For debug-display only
echo
echo "which python: $(which python)"
python --version
echo "CONDA_PREFIX=$CONDA_PREFIX"
# python -c "import sys; print(sys.executable); print('\n'.join(sys.path))"
# python -c "import pytorch_lightning; print(pytorch_lightning.__file__)"

MODEL_NAME=$1
CUDA_VISIBLE_DEVICES=0 python -m experiments.ex_dcase24 cmd_generate_embeddings with \
data_loader.batch_size_eval=32 \
audio_features.segment_length=10 \
audio_features.model=passt \
sentence_features.model=roberta-large \
load_parameters=$MODEL_NAME
