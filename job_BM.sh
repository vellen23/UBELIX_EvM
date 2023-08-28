#!/bin/bash
#SBATCH --job-name=parallel_nmf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2  # Adjust this based on your cluster's resources
#SBATCH --mem=64GB  # Adjust this based on your cluster's resources

# Load any necessary modules
module load Anaconda3
module list
# Activate virtual environment if needed
# conda activate your_environment
cd /storage/homefs/ev19z018/Code/
source venv/bin/activate
which python
echo "Activated venv"


# go to Folder with python scripts
cd ./UBELIX_EvM/NMF/

# Run the Python script
python3 main_BM.py --inputfolder=/storage/homefs/ev19z018/Data/BM_CR/ --parallel=1