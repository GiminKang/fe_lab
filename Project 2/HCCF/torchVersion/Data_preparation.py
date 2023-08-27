import torch
import pickle
import numpy as np
from Model import Model
import pandas as pd
from scipy import sparse

def data_prep(data, prefix=''):
    if prefix == 'train':
        name = 'trn'
    elif prefix == 'valid':
        name = 'val'
    elif prefix == 'test':
        name = 'tst'

    org = data[prefix+'_data']
    idx = data[prefix+'_indptr']

    temp = np.zeros((data['n_users'], data['n_items']))

    for i in range(len(idx)-1):
        for x in range(idx[i],idx[i+1]):
            if org[2][x] == 1:
                j = org[1][x]
                temp[i][j] = 1
    #print(temp)

    data = sparse.coo_matrix(temp)
    print(data)

    with open('Data/stocks/'+name+'Mat.pkl', 'wb') as f:
        pickle.dump(data, f)
    f.close()

if __name__ == '__main__':
    """
    with open('Data/data.pkl', 'rb') as f:
        data = pickle.load(f)
    f.close()

    data_prep(data, 'train')
    data_prep(data, 'valid')
    data_prep(data, 'test')
    """
    with open('Data/stocks/tstMat.pkl', 'rb') as f:
        data = pickle.load(f)
    f.close()

    ckp = torch.load('../Models/tem.mod')
    model = ckp['model']
    opt = torch.optim.Adam(model.parameters())

    print(model.parameters())