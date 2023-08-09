#!/bin/bash
#SBATCH --job-name=parallel_nmf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2  # Adjust this based on your cluster's resources
#SBATCH --mem=16GB  # Adjust this based on your cluster's resources

# Load any necessary modules
module load Anaconda3

# Activate virtual environment if needed
# conda activate your_environment
source venv/bin/activate

# go to Folder with python scripts
cd ./NMF/

# Run the Python script
python3 main_BM.py --inputfolder=/storage/homefs/ev19z018/Data/BM_CR/ --parallel=1