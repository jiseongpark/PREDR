import os
import numpy as np
import pandas as pd

from ast import literal_eval
from sklearn.model_selection import train_test_split


def create_dataset(dataset_path, save_path):
    # Load raw data
    ChG = pd.read_csv(os.path.join(dataset_path, 'ChG_relation.csv'))
    DG = pd.read_csv(os.path.join(dataset_path, 'DG_relation.csv'))
    Ch_feat = pd.read_csv(os.path.join(dataset_path, 'Drug_feature.csv'))
    D_feat = pd.read_csv(os.path.join(dataset_path, 'Disease_feature_raw.csv'))
    G_feat = pd.read_csv(os.path.join(dataset_path, 'Gene_feature.csv'))
    
    # Build Drug-Gene datset
    ChG_dataset = ChG.merge(Ch_feat, how='inner', on='Drug').merge(G_feat, how='inner', on='Gene').loc[:, ['Drug', 'Gene', 'ECFP_PCA', 'PCA_feat']]
    ecfps = np.array(ChG_dataset['ECFP_PCA'].apply(lambda x: literal_eval(x)).tolist())
    gfeat = np.array(ChG_dataset['PCA_feat'].apply(lambda x: literal_eval(x)).tolist())
    ChG_positive = np.concatenate([ecfps, gfeat], axis=1)
    
    drugs = ChG_dataset['Drug'].unique()
    Genes = ChG_dataset['Gene'].unique()
    pairs = ChG_dataset.loc[:, ['Drug', 'Gene']].apply(lambda x: (x[0], x[1])).to_numpy()
    neg_pairs = []
    while len(neg_pairs) < len(pairs):
        temp = [np.random.choice(drugs, 1)[0], np.random.choice(genes, 1)[0]]
        if temp not in pairs and temp not in neg_pairs:
            neg_pairs.append(temp)
    pd_neg_data = pd.DataFrame(neg_pairs, columns=['Drug', 'Gene'])
    pd_neg_dataset = pd_neg_data.merge(Ch_feat, how='inner', on='Drug').merge(G_feat, how='inner', on='Gene').loc[:, ['Drug', 'Gene', 'ECFP_PCA', 'PCA_feat']]
    ecfps_neg = np.array(pd_neg_dataset['ECFP_PCA'].apply(lambda x: literal_eval(x)).tolist())
    gfeat_neg = np.array(pd_neg_dataset['PCA_feat'].apply(lambda x: literal_eval(x)).tolist())
    ChG_negative = np.concatenate([ecfps_neg, gfeat_neg], axis=1)
    
    X_chg = np.concatenate([ChG_positive, ChG_negative], axis=0)
    y_chg = np.array([[0,1] * len(ChG_positive) + [[1,0] * len(ChG_negative)]])
    
    # Build Gene-Disease dataset
    D_feat = D_feat.apply(lambda x: [x[1], [x[2], x[3], x[4], x[5], x[6]]], axis=1, result_type='expand')
    D_feat.rename(columns={0: 'Disease', 1:'feat'}, inplace=True)
    DG_dataset = DG.merge(D_feat, how='inner', on='Disease').merge(G_feat, how='inner', on='Gene').loc[:, ['Disease', 'Gene', 'feat', 'PCA_feat']]
    
    dfeat = np.array(DG_dataset['feat'].tolist())
    gfeat = np.array(DG_dataset['PCA_feat'].apply(lambda x: literal_eval(x)).tolist())
    DG_positive = np.concatenate([gfeat, dfeat], axis=1)
    
    diseases = DG_dataset['Disease'].unique()
    genes = DG_dataset['Gene'].unique()
    pairs = DG_dataset.loc[:, ['Disease', 'Gene']].apply(lambda x: (x[0], x[1]), axis=1).to_numpy()
    DG_neg_pairs = []
    while len(DG_neg_pairs) < 21086:
        temp = [np.random.choice(diseases, 1)[0] , np.random.choice(genes, 1)[0]]
        if temp not in pairs and temp not in DG_neg_pairs:
            DG_neg_pairs.append(temp)
    DG_neg_data = pd.DataFrame(DG_neg_pairs, columns=['Disease', 'Gene'])
    DG_neg_dataset = DG_neg_data.merge(D_feature_raw, how='inner', on='Disease').merge(G_feat, how='inner', on='Gene').loc[:, ['Disease', 'Gene', 'feat', 'PCA_feat']]
    dfeat_neg = np.array(DG_neg_dataset['feat'].tolist())
    gfeat_neg = np.array(DG_neg_dataset['PCA_feat'].apply(lambda x: literal_eval(x)).tolist())
    DG_negative = np.concatenate([gfeat_neg, dfeat_neg], axis=1)
    
    X_dg = np.concatenate([DG_positive, DG_negative], axis=0)
    y_dg = np.array([[0, 1]] * len(DG_positive) + [[1, 0]] * len(DG_negative))
    
    # Save the dataset
    npys = ['X_chg', 'y_chg', 'X_dg', 'y_dg']
    for npy in npys:
        with open(os.path.join(save_path, f'{npy}.npy'), 'wb') as f_npy:
            np.save(f_npy, eval(npy))
    
    return X_chg, y_chg, X_dg, y_dg
    
    
def load_dataset(save_path):
    npys = ['X_chg', 'y_chg', 'X_dg', 'y_dg']
    loaded = []
    
    for npy in npys:
        with open(os.path.join(save_path, f'{npy}.npy'), 'rb') as f_npy:
            loaded.append(np.load(f_npy))
    
    return loaded