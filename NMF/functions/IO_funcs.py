import numpy as np
import sys

sys.path.append('./functions/')
import NMF_funcs


def get_V(con_trial):
    con_trial['SleepState'] = 'Wake'
    con_trial.loc[(con_trial.Sleep > 1) & (con_trial.Sleep < 4), 'SleepState'] = 'NREM'
    con_trial.loc[(con_trial.Sleep == 1), 'SleepState'] = 'NREM1'
    con_trial.loc[(con_trial.Sleep == 6), 'SleepState'] = 'SZ'
    con_trial.loc[(con_trial.Sleep == 4), 'SleepState'] = 'REM'
    con_trial = con_trial[con_trial.SleepState != 'NREM1']

    # remove bad trials
    con_trial = con_trial[con_trial.Artefact < 1]
    con_trial = con_trial.reset_index(drop=True)
    ## normalize LL based on the mean of LL_pre per Chan
    con_trial['LL_norm'] = con_trial.groupby('Chan').apply(lambda x: x['LL'] / x['LL_BL'].mean()).reset_index(0,
                                                                                                              drop=True)
    ## fill nan with mean of specifc Con_ID
    con_trial['LL_norm'].fillna(con_trial.groupby(['Stim', 'Chan', 'Int'])['LL_norm'].transform('mean'),
                                inplace=True)

    df_pivot = con_trial.pivot(index='Chan', columns='Num', values='LL_norm')
    # If there are still missing values after pivot, you might want to fill them with the global mean
    df_pivot.fillna(con_trial['LL_norm'].mean(), inplace=True)
    V = df_pivot.values
    return V, con_trial


def run_NMF(con_trial, mode='stab', k0=5, k1=10):
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
        clusters = NMF_funcs.get_clusters(W)
    elif mode == '2level':
        # 2 -level
        k_range1 = np.arange(k0, k1 + 1)
        k_range2 = np.arange(np.floor(k0/2).astype('int'), np.ceil(k1/2 + 1).astype('int'))
        # clusters = NMF_funcs.hier_staNMF(V, k_range, max_clusters=20)
        clusters, W, H = NMF_funcs.hier2_staNMF(V, k_range1,k_range2, stab_it=20)
    else:  # single
        k = k0
        print('running NMF with a rank of ' + str(k))
        W, H = NMF_funcs.get_nnmf(V, n_components=k)
        clusters = NMF_funcs.get_clusters(W)

    return clusters, W, H, con_trial
