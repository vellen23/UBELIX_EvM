import argparse
import os
import pandas as pd
import h5py
import numpy as np
import sys
from sklearn.decomposition import NMF

# Replace this with your own NMF function
def apply_nmf(data, n_components=5):
    model = NMF(n_components=n_components, init='random', random_state=0)
    W = model.fit_transform(data)
    H = model.components_
    return W, H

def main(inputfolder, stab_NMF, k):
    # Create output folder
    output_folder = os.path.join(inputfolder, "NMF_output")
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over .csv files in input folder
    for filename in os.listdir(inputfolder):
        if filename.endswith(".csv") and filename.startswith("EL"):
            # Load data
            tab = pd.read_csv(os.path.join(inputfolder, filename))
            print(filename)
            data = np.random.rand(10,200)
            # Apply NMF
            if stab_NMF:
                W, H = apply_nmf(data, n_components=k)
            else:
                W, H = apply_nmf(data, n_components=k)
            # Save output
            output_filename = filename.replace(".csv", "_nmf.h5")
            with h5py.File(os.path.join(output_folder, output_filename), 'w') as hf:
                hf.create_dataset('W', data=W)
                hf.create_dataset('H', data=H)
                hf.create_dataset('k', data=k)

    print('---DONE -----')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CSV files and apply NMF.')
    parser.add_argument('inputfolder', type=str, help='Input folder path')
    parser.add_argument('--stab_NMF', type=int, default=0, help='Binary flag for stability NMF')
    parser.add_argument('--k', type=int, default=5, help='Number of clusters for NMF')
    args = parser.parse_args()

    main(args.inputfolder, args.stab_NMF, args.k)