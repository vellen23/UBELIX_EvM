import numpy as np
import sys
import sklearn

sys.path.append('./functions/')
sys.path.append('./FW/')
import ccnmf as NMF_funcs


def get_V(con_trial, m='LL_sig'):
    if not "Ictal" in con_trial:
        con_trial['Ictal'] = 0
    con_trial = con_trial[(con_trial.Ictal==0)&(con_trial.LL<50)&(con_trial.P2P<6000)&(con_trial.Artefact<1)&(con_trial.Sig>-1)].reset_index(drop=True)
    con_trial['Con_ID'] = con_trial.groupby(['Stim', 'Chan']).ngroup()
    con_trial['LL_sig'] = con_trial['LL']*con_trial['Sig']

    m = 'LL_sig' #m+'_BL'
    data_plot = con_trial.groupby(['Con_ID','Stim', 'Chan', 'Block'], as_index=False)[[m, 'Sig']].mean().reset_index(drop=True)

    sig_con = data_plot.groupby(['Con_ID'], as_index=False)['Sig'].mean()
    sig_con = sig_con.loc[(sig_con.Sig>0.1), 'Con_ID'].values
    data_plot = data_plot[np.isin(data_plot.Con_ID, sig_con)].reset_index(drop=True)
    con_IDs = np.unique(data_plot.Con_ID)

    df_pivot = data_plot.pivot(index='Con_ID', columns='Block', values=m)
    df_pivot = df_pivot.apply(lambda row: row.fillna(row.mean()), axis=1)
    df_pivot = df_pivot.fillna(df_pivot.median(axis=0))
    V = df_pivot.values
    ## normalize LL based on the mean of LL_pre per Chan
    #con_trial['LL_norm'] = con_trial.groupby('Chan').apply(
    #    lambda x: x['LL_sig'] / x['LL_pre'].median()).reset_index(0, drop=True)
    ## fill nan with mean of specifc Con_ID
    # con_trial['LL_norm'].fillna(con_trial.groupby('Con_ID')['LL_norm'].transform('mean'), inplace=True)
    # con_trial_block = con_trial.groupby(['Con_ID', 'Stim', 'Chan', 'Block'])['LL_norm'].mean().reset_index()
    # df_pivot = con_trial_block.pivot(index='Con_ID', columns='Block', values='LL_norm')
    # # If there are still missing values after pivot, you might want to fill them with the global mean
    # df_pivot.fillna(con_trial['LL_norm'].mean(), inplace=True)
    # V = df_pivot.values
    return V, con_trial


def run_NMF(con_trial, mode='stab', k0=3, k1=10):
    V, con_trial = get_V(con_trial)
    if mode == 'stab':
        print('running stability NMF')
        # run stability NMF for different ranks
        _, instability = NMF_funcs.stabNMF(V, num_it=20, k0=k0, k1=k1, init='nndsvda', it=2000)
        # select rank with lowest instability value
        ranks = np.arange(k0, k1 + 1)
        k = ranks[np.argmin(instability)]

        # rerun NMF with chosen best rank
        print('running NMF with a chosen rank of ' + str(k))
        W, H = NMF_funcs.get_nnmf(V, k, init='nndsvda', it=2000)
    elif mode == 'hier_stab':
        k_range = np.arange(k0, k1 + 1)
        clusters = NMF_funcs.hier_staNMF(V, k_range, max_clusters=20)
    else:  # single
        k = k0
        print('running NMF with a rank of ' + str(k))
        W, H = NMF_funcs.get_nnmf(V, n_components=k)

    clusters = NMF_funcs.get_clusters(W)
    # # Assigning the clusters to each connection
    # clusters = np.argmax(W, axis=1)
    #
    # # Create a dictionary with connection_id as keys and cluster as values
    # cluster_dict = {i: cluster for i, cluster in enumerate(clusters)}
    # # Add the 'Cluster' column to the DataFrame
    # con_trial['Cluster'] = con_trial['Con_ID'].map(cluster_dict)
    # # con_trial.to_csv(os.path.join(output_folder, filename.replace(".csv", "_cluster.csv")), header=True,
    # #                  index=False)
    return clusters, W, H, con_trial


def run_conNMF(con_trial, experiment_dir=None, k0=3, k1=6):

    V, con_trial = get_V(con_trial)
    labels = None
    runs_per_rank = 100
    if k1 > np.min(V.shape):
        k1 = np.min(V.shape) - 1

    V = sklearn.preprocessing.normalize(V).T # transpose?
    NMF_funcs.parallel_nmf_consensus_clustering(V, (k0, k1), runs_per_rank, experiment_dir, target_clusters=labels, save_connectivity = 0)
    return con_trial