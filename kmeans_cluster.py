import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import warnings
warnings.simplefilter('ignore')



def find_nearest_centroid(x, y, c_x, c_y):
    distance = 1000000000
    cluster = -1
    for i in range(len(c_x)):
        dist = int(np.sqrt((c_x[i] - x) ** 2 + (c_y[i] - y) ** 2))
        if dist < distance:
            distance = dist
            cluster = i

    return cluster


def get_new_centroids(m, data):
    centroidsx = []
    centroidsy = []
    for i in range(0, 10):
        indices = np.where(m == i)
        cluster = data.iloc[indices[0],:]
        centroidsx.append(np.mean(cluster[2]))
        centroidsy.append(np.mean(cluster[3]))

    return centroidsx, centroidsy


def wc_ssd():
    print('')

def sc():
    print('')

def nmi():
    print('')



def kmeans(dataFileName, K=10, max_iter=50):
    data = pd.read_csv(dataFileName, header=None)
    centroids = np.random.randint(0, data.shape[0], size=K)
    centroidsx = []
    centroidsy = []
    i = 0
    for c in centroids:
        centroidsx.append(data.loc[c, 2])
        centroidsy.append(data.loc[c, 3])
        i += 1
    del i
    #print(centroidsx, centroidsy)

    m = np.zeros(shape=(data.shape[0],), dtype=int)


    for i in range(0, max_iter):
        clusters = {}
        for j in range(0, K):
            clusters[j] = []

        for index, row in data.iterrows():
             m[index] = find_nearest_centroid(row[2], row[3], c_x=centroidsx, c_y=centroidsy)

        centroidsx, centroidsy = get_new_centroids(m, data)

    print(centroidsx, centroidsy)





np.random.seed(0)
kmeans(dataFileName='./digits-embedding.csv')



