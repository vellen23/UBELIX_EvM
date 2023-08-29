import argparse
import os
import pandas as pd
import h5py
import numpy as np
import sys
import json
from os import environ as cuda_environment
import multiprocessing

sys.path.append('./functions/')
import NMF_funcs
import BM_CR_funcs as BMf


def read_h5(file):
    with h5py.File(os.path.join(file), 'r') as hf:
        # Read W and H datasets
        W = hf['W'][:]
        H = hf['H'][:]

        # Read the Cluster dataset (assignment_array)
        assignment_array = hf['Cluster'][:]
        clusters = {}

        # Convert the assignment_array back into a dictionary
        for entry in assignment_array:
            cluster_idx = entry['cluster_idx']
            channels = entry['channels']
            clusters[cluster_idx] = channels.tolist()

    # Print the read cluster assignments
    for cluster_idx, channels in clusters.items():
        print(f"Cluster {cluster_idx}: Channels {channels}")

    # Read cluster assignments from the JSON file
    with open(os.path.join(output_folder, output_filename_json), 'r') as json_file:
        column_cluster_assignments = json.load(json_file)

    return clusters, W, H


def process_file(inputfolder, output_folder, filename):
    print(f"Starting: {filename}")
    con_trial = pd.read_csv(os.path.join(inputfolder, filename))
    clusters, W, H, con_trial = BMf.run_NMF(con_trial, mode='stab', k0=3, k1=10)

    output_filename = filename.replace(".csv", "_nmf.h5")
    with h5py.File(os.path.join(output_folder, output_filename), 'w') as hf:
        hf.create_dataset('W', data=W)
        hf.create_dataset('H', data=H)
        # hf.create_dataset('Cluster', data=assignment_array)
    output_filename = filename.replace(".csv", "_nmf_cluster.json")
    with open(os.path.join(output_folder, output_filename), 'w') as json_file:
        json.dump(clusters, json_file)

    # update con trial
    con_trial.to_csv(os.path.join(output_folder, filename.replace(".csv", "_cluster.csv")), header=True,
                     index=False)
    print(f"Processed: {filename}")


def conNMF(inputfolder, output_folder, filename):
    print(f"Starting: {filename}")
    con_trial = pd.read_csv(os.path.join(inputfolder, filename))

    experiment_dir = os.path.join(output_folder, filename[0:6] + 'conNMF')

    con_trial = BMf.run_conNMF(con_trial, experiment_dir=experiment_dir, k0=4, k1=10)

    # update con trial
    con_trial.to_csv(os.path.join(output_folder, filename.replace(".csv", "_cluster.csv")), header=True, index=False)

    print(f"Processed: {filename}")


def main(inputfolder, parallel=1):
    print('main')
    output_folder = os.path.join(inputfolder, "NMF_output")
    os.makedirs(output_folder, exist_ok=True)

    files_to_process = [
        filename
        for filename in os.listdir(inputfolder)
        if filename.endswith(".csv") and filename.startswith("EL")
    ]

    if parallel:
        # Use multiprocessing to parallelize the processing
        num_processes = multiprocessing.cpu_count()
        with multiprocessing.Pool(num_processes) as pool:
            pool.starmap(conNMF, [(inputfolder, output_folder, filename) for filename in files_to_process])
    else:
        for filename in files_to_process:
            conNMF(inputfolder, output_folder, filename)
    print('---DONE -----')

#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CSV files and apply NMF.')
    parser.add_argument('--inputfolder', type=str,
                        default='C:\\Users\\i0328442\Documents\EVM\EL_Experiment\\UBELIX_Cluster\\Data\\BM',
                        help='Input folder path')

    # parser.add_argument('--inputfolder', type=str,
    #                     default='/Users/ellenvanmaren/Desktop/Insel/EL_experiment/Codes/Cluster_scripts/Data/BM_CR',
    #                     help='Input folder path')

    parser.add_argument('--subj_parallel', type=int, default=0, help='Parallel (0/1)')
    args = parser.parse_args()
    # multiprocessing.freeze_support()
    main(args.inputfolder, args.subj_parallel)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Process CSV files and apply NMF.')
#     # parser.add_argument('--inputfolder', type=str, default='/Users/ellenvanmaren/Desktop/Insel/EL_experiment/Codes/Cluster_scripts/Data/BM_CR', help='Input folder path')
#     # parser.add_argument('--stab_NMF', type=int, default=0, help='Binary flag for stability NMF')
#     # parser.add_argument('--k0', type=int, default=2, help='Min Number of clusters for NMF')
#     # parser.add_argument('--k1', type=int, default=10, help='Max Number of clusters for NMF')
#     # args = parser.parse_args()
#     #
#     # main(args.inputfolder, args.stab_NMF, args.k0, args.k1)
#     parser.add_argument('--inputfolder', type=str, default='/Users/ellenvanmaren/Desktop/Insel/EL_experiment/Codes/Cluster_scripts/Data/BM_CR', help='Input folder path')
#     args = parser.parse_args()
#
#     main(args.inputfolder)
