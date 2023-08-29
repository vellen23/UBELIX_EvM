#!/bin/bash
#SBATCH --job-name=parallel_nmf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2  # Adjust this based on your cluster's resources
#SBATCH --mem=16GB  # Adjust this based on your cluster's resources

module load Anaconda3
eval "$(conda shell.bash hook)"
# conda env create -f venv_nmf.yml
conda activate venv_nmf
# conda env create -f env_base.yml
# module list
# Activate virtual environment if needed
# conda activate your_environment
# cd /storage/homefs/ev19z018/Code/
# source venv/bin/activate
# which pythony

echo "Activated venv"


# go to Folder with python scripts
# cd ./UBELIX_EvM/NMF/
cd ./NMF/
echo "Change path"
# Run the Python script
python main_BM.py --inputfolder=/storage/homefs/ev19z018/Data/BM_CR/ --parallel=0