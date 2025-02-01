#!/bin/bash

#SBATCH --job-name=binsegm
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=48:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-1
#SBATCH --exclude=sh00,sh[10-19]

current_dir=`pwd`
export PATH=$PATH:~/.local/bin

echo "Session " ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}

echo "Running on " $(hostname)

echo "Copying the source directory and data"
date
mkdir $TMPDIR/code
rsync -r --exclude logs --exclude logslurms --exclude configs . $TMPDIR/code

echo "Checking out the correct version of the code commit_id 1499de7cb483d24778e385c94d024ef22916537f"
cd $TMPDIR/code
git checkout 1499de7cb483d24778e385c94d024ef22916537f


echo "Setting up the virtual environment"
python3 -m venv $TMPDIR/venv
source $TMPDIR/venv/bin/activate

# Install the library
python -m pip install -e .
pip install pandas
pip install tabulate

echo "Training"
python -m torchtmpl.main train /usr/users/sdim/sdim_34/lora-llm/configs/tmpx2lweo86-config.yml

if [[ $? != 0 ]]; then
    exit -1
fi
