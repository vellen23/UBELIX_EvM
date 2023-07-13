import argparse
import os
import pandas as pd
import h5py
import numpy as np
import sys
from sklearn.decomposition import NMF
from os import environ as cuda_environment
sys.path.append('./functions/')
import NMF_funcs

def main(inputfolder, stab_NMF, k0, k1):
    # Create output folder
    output_folder = os.path.join(inputfolder, "NMF_output")
    os.makedirs(output_folder, exist_ok=True)
    print(output_folder)
    # Iterate over .csv files in input folder
    for filename in os.listdir(inputfolder):
        if filename.endswith(".csv") and filename.startswith("EL"):
            print(filename)
            # Load data
            # Load data
            con_trial = pd.read_csv(os.path.join(inputfolder, filename))
            con_trial = con_trial[(con_trial.Sig >-1)&(con_trial.Artefact < 2)]
            con_trial = con_trial.reset_index(drop=True)
            ## 1. Add unique connection label for each StimxChan combination: Con_ID
            con_trial['Con_ID'] = con_trial.groupby(['Stim', 'Chan']).ngroup()
            con_trial.insert(5, 'LL_sig', con_trial.LL *con_trial.Sig)
            ## normalize LL based on the mean of LL_pre per Chan
            con_trial['LL_norm'] = con_trial.groupby('Chan').apply(lambda x: x['LL_sig'] / x['LL_pre'].mean()).reset_index(0, drop=True)
            ## fill nan with mean of specifc Con_ID
            con_trial['LL_norm'].fillna(con_trial.groupby('Con_ID')['LL_norm'].transform('mean'), inplace=True)
            con_trial_block  = con_trial.groupby(['Con_ID','Stim', 'Chan', 'Block'])['LL_norm'].mean().reset_index()
            df_pivot = con_trial_block.pivot(index='Con_ID', columns='Block', values='LL_norm')
            # If there are still missing values after pivot, you might want to fill them with the global mean
            df_pivot.fillna(con_trial['LL_norm'].mean(), inplace=True)
            V = df_pivot.values

            # Apply NMF
            if stab_NMF:
                print('running stability NMF')
                # run stability NMF for different ranks
                _, instability = NMF_funcs.stabNMF(V, num_it=100, k0=k0, k1=k1, init='nndsvda',it=2000)
                # select rank with lowest instability value
                ranks = np.arange(k0, k1+1)
                k = ranks[np.argmin(instability)]

                # rerun NMF with chosen best rank
                print('running NMF with a chosen rank of '+str(k))
                W, H = NMF_funcs.get_nnmf(V, k, init='nndsvda',it=2000)
            else:
                k = k0
                print('running NMF with a rank of '+str(k))
                W, H = NMF_funcs.get_nnmf(V, n_components=k)
            # Save output
            output_filename = filename.replace(".csv", "_nmf.h5")
            with h5py.File(os.path.join(output_folder, output_filename), 'w') as hf:
                hf.create_dataset('W', data=W)
                hf.create_dataset('H', data=H)
                hf.create_dataset('k', data=k)

            # Assigning the clusters to each connection
            clusters = np.argmax(W, axis=1)

            # Create a dictionary with connection_id as keys and cluster as values
            cluster_dict = {i: cluster for i, cluster in enumerate(clusters)}
            # Add the 'Cluster' column to the DataFrame
            con_trial['Cluster'] = con_trial['Con_ID'].map(cluster_dict)
            con_trial.to_csv(os.path.join(output_folder, filename.replace(".csv", "_cluster.csv")), header=True, index=False)


    print('---DONE -----')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CSV files and apply NMF.')
    parser.add_argument('--inputfolder', type=str, help='Input folder path')
    parser.add_argument('--stab_NMF', type=int, default=0, help='Binary flag for stability NMF')
    parser.add_argument('--k0', type=int, default=2, help='Min Number of clusters for NMF')
    parser.add_argument('--k1', type=int, default=10, help='Max Number of clusters for NMF')
    args = parser.parse_args()

    main(args.inputfolder, args.stab_NMF, args.k0, args.k1)