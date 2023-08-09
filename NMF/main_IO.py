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
import IO_funcs as IOf


def process_file(inputfolder, output_folder, filename):
    print(f"Starting: {filename}")
    con_trial = pd.read_csv(os.path.join(inputfolder, filename))
    clusters, W, H, con_trial = IOf.run_NMF(con_trial, mode='2level', k0=4, k1=8)

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


def main(inputfolder, parallel=1):
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
            pool.starmap(process_file, [(inputfolder, output_folder, filename) for filename in files_to_process])
    else:
        for filename in files_to_process:
            process_file(inputfolder, output_folder, filename)
    print('---DONE -----')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CSV files and apply NMF.')
    parser.add_argument('--inputfolder', type=str,
                        default='/Users/ellenvanmaren/Desktop/Insel/EL_experiment/Codes/Cluster_scripts/Data/IO',
                        help='Input folder path')
    parser.add_argument('--parallel', type=int, default=0, help='Parallel (0/1)')
    args = parser.parse_args()

    main(args.inputfolder, args.parallel)

# def main(inputfolder, stab_NMF, k0, k1):
#     # Create output folder
#     output_folder = os.path.join(inputfolder, "NMF_output")
#     os.makedirs(output_folder, exist_ok=True)
#     print(output_folder)
#     # Iterate over .csv files in input folder
#     for filename in os.listdir(inputfolder):
#         if filename.endswith(".csv") and filename.startswith("EL"):
#             print(filename)
#             # Load data
#             # Load data
#             con_trial = pd.read_csv(os.path.join(inputfolder, filename))
#
#             # Apply NMF
#             if stab_NMF:
#                 print('running stability NMF')
#                 # run stability NMF for different ranks
#                 ## FW code for hierarchical NMF
#                 # todo: add hierarchical code and get different clusters
#                 _, instability = NMF_funcs.stabNMF(V, num_it=100, k0=k0, k1=k1, init='nndsvda', it=2000)
#                 # select rank with lowest instability value
#                 ranks = np.arange(k0, k1 + 1)
#                 k = ranks[np.argmin(instability)]
#
#                 # rerun NMF with chosen best rank
#                 print('running NMF with a chosen rank of ' + str(k))
#                 W, H = NMF_funcs.get_nnmf(V, k, init='nndsvda', it=2000)
#             else:
#                 k = k0
#                 print('running NMF with a rank of ' + str(k))
#                 W, H = NMF_funcs.get_nnmf(V, n_components=k)
#             # Save output
#             output_filename = filename.replace(".csv", "_nmf.h5")
#             with h5py.File(os.path.join(output_folder, output_filename), 'w') as hf:
#                 hf.create_dataset('W', data=W)
#                 hf.create_dataset('H', data=H)
#                 hf.create_dataset('k', data=k)
#
#             # Assigning the clusters to each connection
#             clusters = np.argmax(W, axis=1)
#
#             # Create a dictionary with connection_id as keys and cluster as values
#             cluster_dict = {i: cluster for i, cluster in enumerate(clusters)}
#             # Add the 'Cluster' column to the DataFrame
#             con_trial['Cluster'] = con_trial['Con_ID'].map(cluster_dict)
#             con_trial.to_csv(os.path.join(output_folder, filename.replace(".csv", "_cluster.csv")), header=True,
#                              index=False)
#
#     print('---DONE -----')
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Process CSV files and apply NMF.')
#     parser.add_argument('--inputfolder', type=str, help='Input folder path')
#     parser.add_argument('--stab_NMF', type=int, default=0, help='Binary flag for stability NMF')
#     parser.add_argument('--k0', type=int, default=2, help='Min Number of clusters for NMF')
#     parser.add_argument('--k1', type=int, default=10, help='Max Number of clusters for NMF')
#     args = parser.parse_args()
#
#     main(args.inputfolder, args.stab_NMF, args.k0, args.k1)
