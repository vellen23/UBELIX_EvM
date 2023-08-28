import argparse
import os
from os import environ as cuda_environment
import shutil
import glob
from distutils.dir_util import copy_tree

sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab
shared_path = "C:\\Users\\i0328442\\Documents\\EVM\\EL_Experiment\\UBELIX_Cluster\\Data"
patient_path = "X:\\4 e-Lab\\EvM\\Projects\\EL_experiment\\Analysis\\Patients\\"

subjs = ["EL004", "EL005", "EL010", "EL011", "EL012", "EL013", "EL014", "EL015", "EL016", "EL017", "EL019", "EL020",
         "EL021", "EL022", "EL025", "EL026"]

subjs = ["EL024"]
protocols = ['InputOutput', 'BrainMapping']
protocols_short = ['IO', 'BM']
cond_folder = 'CR'


def archie2T_contrial(subjs, protocols, protocols_short):
    for subj in subjs:
        for prot, prot_s in zip(protocols, protocols_short):
            os.makedirs(os.path.join(shared_path, prot_s), exist_ok=True)  # Create directories if they don't exist
            path_patient_analysis = patient_path + subj
            file_con = path_patient_analysis + '\\' + prot + '\\' + cond_folder + '/data/con_trial_all.csv'
            if os.path.isfile(file_con):
                file_new = os.path.join(shared_path, prot_s, subj + '_con_trial.csv')
                shutil.copyfile(file_con, file_new)
                print(file_new + ' --- saved', end='\r')


def T2archie_NMF(subjs, protocols, protocols_short, shared_path, patient_path):
    # save UBELIX NMF output on Archie in subject specific folder
    for subj in subjs:
        for prot, prot_s in zip(protocols, protocols_short):
            # Subject specific new path
            path_patient_analysis = os.path.join(patient_path, subj, prot, 'CR', 'NMF')
            os.makedirs(path_patient_analysis, exist_ok=True)  # Create directories if they don't exist

            # All files in shared folder that start with subj (subject's ID)
            files = glob.glob(os.path.join(shared_path, prot_s, 'NMF_output', subj + '*'))
            for file in files:
                file_new = os.path.join(path_patient_analysis, os.path.basename(file))
                shutil.copyfile(file, file_new)

def T2archie_conNMF(subjs, protocols, protocols_short, shared_path, patient_path):
    # save UBELIX NMF output on Archie in subject specific folder
    for subj in subjs:
        for prot, prot_s in zip(protocols, protocols_short):
            # Subject specific new path
            path_pat = os.path.join(patient_path, subj, prot, 'CR', 'NMF', 'conNMF')
            if not os.path.isdir(path_pat):
                os.makedirs(path_pat, exist_ok=True)  # Create directories if they don't exist

            current_path = os.path.join(shared_path, prot_s, 'NMF_output', subj+'_conNMF')
            # shutil.copytree(current_path, path_pat)
            copy_tree(current_path, path_pat)

subjs = ["EL013", "EL014", "EL015", "EL016", "EL017", "EL019", "EL020",
         "EL021", "EL022", "EL025", "EL026"]
protocols = ['BrainMapping']
protocols_short = ['BM']
T2archie_conNMF(subjs, protocols, protocols_short, shared_path, patient_path)

archie2T_contrial(subjs, protocols, protocols_short)
T2archie_NMF(subjs, protocols, protocols_short, shared_path, patient_path)
