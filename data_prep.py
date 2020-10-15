import numpy as np

# help functions that clean and organize data for ML algorithms

def train_test_split_data(data, ratio):
    idxs = np.random.permutation(data.shape[0])
    train_idx, test_idx = idxs[:ratio], idxs[ratio:]
    train, test = data[train_idx], data[test_idx]
    return train, test






