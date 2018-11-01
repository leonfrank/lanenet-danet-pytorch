import torch
import random
import numpy as np
from sklearn.cluster import KMeans
def seg2points(X, K):
    '''Pick K centroids with K-means++ algorithm.
    Args:
      X: (tensor) data, sized [N,D].
      K: (int) number of clusters.
    ReturnsL
      (tensor) picked centroids, sized [K,D].
    '''
    C,H,W = X.size()
    M = X.permute(1,2,0).view(-1, C).cpu().numpy()
    
    cls = KMeans(n_clusters = K)
    kmeans = cls.fit(M)
    labels = kmeans.labels_
    labels = labels.reshape((H,W))
    idx, count = np.unique(labels,return_counts = True)
    dtype = [('idx',np.int32),('count',np.int32)]
    arr = list(zip(list(idx),list(count)))
    arr = np.array(arr,dtype = dtype)
    arr = np.sort(arr,order='count')
    arr = arr[:-1]
    ret = {}
    ret["Lanes"] = [[]]*len(arr)
    
    for rowid,row in enumerate(labels):    
        for idx,count in arr:
            x_ = np.mean(np.where(row == idx)[0])
            ret["Lanes"][idx].append({"y":float(rowid),"x":x_})
    return ret

