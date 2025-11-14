import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances


def unit_dist(x, y):
    i_th = x
    j_th = y
    
    n_features = arr[i].shape[0]
    
    dist = 0.0
    for k in range(n_features):
        dist += wass(i_th[k], j_th[k])
    
    return dist


def parallel_aligned_distance(batch):
    i, j, arr = batch
    
    n_feat = len(arr)
    
    dists = []
    for feat in range(n_feat): 
        s_i = arr[feat][i]
        s_j = arr[feat][j]
        
        if len(s_i) < 1 or len(s_j) < 1:
            continue
        
        dists.append(wasserstein_distance(s_i, s_j))
    
    return np.sum(dists)


def back_to_time_list(binarized):
    t = np.where(binarized == 1)[0]
    return list(t)


def aligned_distance(x, y):
    # x, y = binary repr of spikes (L,)
    
    t_x = back_to_time_list(x)
    t_y = back_to_time_list(y)
    
    return time_wasserstein(t_x, t_y)


def parallel_featurewise_pdist(X_k):
    return squareform(pdist(X_k, aligned_distance))


def parallel_featurewise_pdist2(X_k):
    return squareform(pdist(X_k, time_wasserstein))


def parallel_featurewise_pdist_test(X_k):
    return pairwise_distances(X_k, metric=time_wasserstein2, n_jobs=-1)


def time_wasserstein(t1, t2, skip_none=False):
    if len(t1) < 1 and len(t2) < 1:
        return 0.0
    
    if len(t1) < 1:
        if skip_none:
            return 0.0
        return np.sum(t2) / len(t2)
    
    if len(t2) < 1:
        if skip_none:
            return 0.0
        return np.sum(t1) / len(t1)
    
    return wasserstein_distance(t1, t2)


def time_wasserstein2(x, y):
    
    t1 = back_to_time_list(x)
    t2 = back_to_time_list(y)
    
    if len(t1) < 1 and len(t2) < 1:
        return 0.0
    
    if len(t1) < 1:
        return np.sum(t2) / len(t2)
    
    if len(t2) < 1:
        return np.sum(t1) / len(t1)
    
    n = len(t1)
    m = len(t2)
    
    if n > m:
        return time_wasserstein2(y, x)
    
    dist_a = 0.0
    for i in range(n):
        dist_a += abs(t1[i] - t2[i])
    
    dist_a /= n
    
    dist_b = 0.0
    for j in range(n, m):
        dist_b += t2[j]
    if m > n:
        dist_b /= (m-n)
    
    return dist_a + dist_b


def parallel_aligned_distance2(batch):
    i, j, X = batch
    
    n_windows = X.shape[0]
    n_feats = X.shape[1]
    
    d_ij = 0.0
    
    for feat in range(n_feats):
        s_i = back_to_time_list(X[i, feat])
        s_j = back_to_time_list(X[j, feat])
        
        if len(s_i) < 1 and len(s_j) < 1:
            continue
        
        if len(s_i) < 1:
            d_ij += np.sum(s_j) / len(s_j)
            continue
        
        if len(s_j) < 1:
            d_ij += np.sum(s_i) / len(s_i)
            continue
            
        d_ij += wasserstein_distance(s_i, s_j)
    
    return d_ij


def parallel_aligned_distance_test(batch):
    i, j, X = batch
    
    n_windows = X.shape[0]
    n_feats = X.shape[1]
    
    d_ij = 0.0
    scale = 1
    precision = 6
    
    for feat in range(n_feats):
        s_i = back_to_time_list(X[i, feat])
        s_j = back_to_time_list(X[j, feat])
        
        if len(s_i) < 1 and len(s_j) < 1:
            continue
        
        if len(s_i) < 1:
            d_ij += np.round(np.sum(s_j) / (len(s_j)*scale), precision)
            continue
        
        if len(s_j) < 1:
            d_ij += np.round(np.sum(s_i) / (len(s_i)*scale), precision)
            continue
            
        d_ij += np.round(wasserstein_distance(s_i, s_j) / scale, precision)
    
    return (i, j, d_ij)


def parallel_mixed_distance(batch):
    i, j, arr = batch
    
    n_feat = len(arr)
    
    dists = []

    for feat_k in range(n_feat):
        for feat_l in range(n_feat):
            if feat_k == feat_l:
                continue

            win_i_spike_k = arr[feat_k][i]
            win_j_spike_l = arr[feat_l][j]

            if len(win_i_spike_k) < 1 or len(win_j_spike_l) < 1:
                appended = np.hstack((win_i_spike_k,win_j_spike_l))
                d = np.sum(appended)
                d /= len(appended)               
                dists.append(d)
            else:
                dists.append(wasserstein_distance(win_i_spike_k, win_j_spike_l))
    
    return np.sum(dists)