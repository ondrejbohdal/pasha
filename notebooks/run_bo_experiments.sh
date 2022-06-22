#!/bin/sh
#SBATCH -N 1  # nodes requested
#SBATCH -n 1  # tasks requested
#SBATCH --job-name=hpo
#SBATCH --cpus-per-task=4
#SBATCH --mem=14000  # memory in Mb
#SBATCH --time=0-22:00:00
#SBATCH --array=1-90%30

# =====================
# Logging information
# =====================

echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
source ~/.bashrc

# Make script stop after first error
set -e

MY_ID=$(whoami)

source /home/${MY_ID}/miniconda3/bin/activate pasha

echo ${CONFIGS[SLURM_ARRAY_TASK_ID-1]}

echo "start training"

DATASET_NAME=(
cifar10
cifar10
cifar10
cifar10
cifar10
cifar10
cifar10
cifar10
cifar10
cifar10
cifar10
cifar10
cifar10
cifar10
cifar10
cifar10
cifar10
cifar10
cifar10
cifar10
cifar10
cifar10
cifar10
cifar10
cifar10
cifar10
cifar10
cifar10
cifar10
cifar10
cifar100
cifar100
cifar100
cifar100
cifar100
cifar100
cifar100
cifar100
cifar100
cifar100
cifar100
cifar100
cifar100
cifar100
cifar100
cifar100
cifar100
cifar100
cifar100
cifar100
cifar100
cifar100
cifar100
cifar100
cifar100
cifar100
cifar100
cifar100
cifar100
cifar100
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
ImageNet16-120
)

APPROACH=(
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
pasha-bo
asha-bo
)

RANDOM_SEED=(
31415927
31415927
0
0
1234
1234
3458
3458
7685
7685
31415927
31415927
0
0
1234
1234
3458
3458
7685
7685
31415927
31415927
0
0
1234
1234
3458
3458
7685
7685
31415927
31415927
0
0
1234
1234
3458
3458
7685
7685
31415927
31415927
0
0
1234
1234
3458
3458
7685
7685
31415927
31415927
0
0
1234
1234
3458
3458
7685
7685
31415927
31415927
0
0
1234
1234
3458
3458
7685
7685
31415927
31415927
0
0
1234
1234
3458
3458
7685
7685
31415927
31415927
0
0
1234
1234
3458
3458
7685
7685
)

NB201_RANDOM_SEED=(
0
0
0
0
0
0
0
0
0
0
1
1
1
1
1
1
1
1
1
1
2
2
2
2
2
2
2
2
2
2
0
0
0
0
0
0
0
0
0
0
1
1
1
1
1
1
1
1
1
1
2
2
2
2
2
2
2
2
2
2
0
0
0
0
0
0
0
0
0
0
1
1
1
1
1
1
1
1
1
1
2
2
2
2
2
2
2
2
2
2
)

python run_bo_experiments.py --dataset_name ${DATASET_NAME[SLURM_ARRAY_TASK_ID-1]} --nb201_random_seed ${NB201_RANDOM_SEED[SLURM_ARRAY_TASK_ID-1]} --random_seed ${RANDOM_SEED[SLURM_ARRAY_TASK_ID-1]} --scheduler ${APPROACH[SLURM_ARRAY_TASK_ID-1]}

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Training ended: $dt"