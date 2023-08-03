import argparse
import os
from os import environ as cuda_environment
import shutil

sub_path = 'X:\\4 e-Lab\\'  # y:\\eLab
shared_path = "T:\\EL_experiment\\Codes\\UBELIX_Data\\"
patient_path = "X:\\4 e-Lab\\EvM\\Projects\\EL_experiment\\Analysis\\Patients\\"

subjs = ["EL004", "EL005", "EL0010", "EL011", "EL012", "EL013", "EL014", "EL015", "EL016", "EL017", "EL019", "EL020",
         "EL021", "EL022", "EL025", "EL026"]
protocols = ['InputOutput', 'BrainMapping']
protocols_short = ['IO', 'BM']
cond_folder = 'CR'
for subj in subjs:
    for prot, prot_s in zip(protocols, protocols_short):
        path_patient_analysis = patient_path + subj
        file_con = path_patient_analysis + '\\' + prot + '\\' + cond_folder + '/data/con_trial_all.csv'
        if os.path.isfile(file_con):
            file_new = os.path.join(shared_path, prot_s, subj + '_con_trial.csv')
            shutil.copyfile(file_con, file_new)
            print(file_new + ' --- saved', end='\r')
