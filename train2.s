#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --time=5:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=salsa-train-stage-2
#SBATCH --mail-type=END
#SBATCH --mail-user=ekp252@nyu.edu
#SBATCH --output=run_train2.out
#SBATCH --error=error_train2.err
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
CUDA_VISIBLE_DEVICES=0 python -m experiments.ex_dcase24 with \
data_loader.batch_size=64 \
data_loader.batch_size_eval=32 \
audio_features.segment_length=10 \
audio_features.model=passt \
sentence_features.model=roberta-large \
lr_audio_encoder=2e-5 \
lr_audio_project=2e-5 \
lr_sentence_encoder=2e-5 \
lr_sentence_project=2e-5 \
rampdown_type=cosine \
max_epochs=20 \
rampdown_stop=15 \
warmup_length=1 \
rampdown_start=1 \
train_on=clothov2 \
load_parameters=$MODEL_NAME \
load_last=best \
loss_weight=0.0 \
distill_weight=1.0 \
distill_from=$MODEL_NAME \
seed=523528930
