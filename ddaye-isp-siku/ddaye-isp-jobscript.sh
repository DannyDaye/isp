#!/bin/bash
#ABATCH --account=an-ddaye
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=0-1:00:00
#SBATCH --output=%N-%j.out

# Loading modules
module load python/3.11.5 scipy-stack

# Creating and activating the virtual environment
virtualenv --no-download ENV
source ENV/bin/activate

# Installing packages
pip install --no-index --upgrade pip
pip install --no-index scikit-learn openpyxl

# Launching the python files
python stat_compare.py
python hof_classification.py
