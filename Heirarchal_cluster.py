import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import spatial
from scipy.cluster import hierarchy
from kmeans_cluster import wc_ssd, sc, get_new_centroids, nmi


import warnings
warnings.simplefilter('ignore')


def hierarchal_cluster(dataFileName, type):
    data = pd.read_csv(dataFileName, header=None)
    sampled_data = pd.DataFrame()
    for i in range(0, 10):
        smple_data = data.loc[data[1] == i]
        values = np.random.randint(low=0, high=smple_data.shape[0], size=10)
        smple_data = smple_data.iloc[values]
        sampled_data = pd.concat([sampled_data, smple_data], ignore_index=True)

    locations = sampled_data[[2, 3]]

    y = spatial.distance.pdist(locations.to_numpy(dtype=float, copy=True))
    if type == 1:
        Z = hierarchy.single(y)
    elif type == 2:
        Z = hierarchy.complete(y)
    elif type == 3:
        Z = hierarchy.average(y)
    else:
        print('Please choose valid type')
        return

    plt.figure(figsize=(10, 6))
    dn1 = hierarchy.dendrogram(Z, orientation='top')

    #hierarchy.set_link_color_palette(None)  # reset to default after use
    plt.xlabel('Cluster Number (In the new dataset)')
    plt.ylabel('Distance between clusters')
    plt.show()

    K = [2, 4, 8, 16, 32]
    #clustered = hierarchy.cut_tree(Z, n_clusters=16)

    wc_lst = []
    sc_lst = []

    for k in K:
        clustered = hierarchy.cut_tree(Z, n_clusters=k)
        m = np.zeros(shape=(clustered.shape[0],), dtype=int)
        index = 0
        for i in clustered:
            m[index] = i[0]
            index += 1
        cx, cy = get_new_centroids(m, sampled_data)
        wc_lst.append(wc_ssd(c_x=cx, c_y=cy, m=m, data=sampled_data))
        sc_lst.append(sc(c_x=cx, c_y=cy, m=m, data=sampled_data))

    plt.plot(K, wc_lst, color='blue', label="WC-SSD")
    plt.xlabel('# number of Clusters')
    plt.ylabel('WC-SSD score')
    plt.legend()
    plt.show()

    plt.plot(K, sc_lst[::-1], color='red', label='SC')
    plt.xlabel('# number of Clusters')
    plt.ylabel('SC score')
    plt.legend()
    plt.show()

    # Ran it before and k=4 was the best in all three cases
    clustered = hierarchy.cut_tree(Z, n_clusters=4)
    m = np.zeros(shape=(clustered.shape[0],), dtype=int)
    index = 0
    for i in clustered:
        m[index] = i[0]
        index += 1

    print("NMI: {0}".format(nmi(sampled_data[1], m)))


def calcNMI(dataFileName, type):
    data = pd.read_csv(dataFileName, header=None)
    sampled_data = pd.DataFrame()
    for i in range(0, 10):
        smple_data = data.loc[data[1] == i]
        values = np.random.randint(low=0, high=smple_data.shape[0], size=10)
        smple_data = smple_data.iloc[values]
        sampled_data = pd.concat([sampled_data, smple_data], ignore_index=True)

    locations = sampled_data[[2, 3]]

    y = spatial.distance.pdist(locations.to_numpy(dtype=float, copy=True))
    if type == 1:
        Z = hierarchy.single(y)
    elif type == 2:
        Z = hierarchy.complete(y)
    elif type == 3:
        Z = hierarchy.average(y)
    else:
        print('Please choose valid type')
        return

    clustered = hierarchy.cut_tree(Z, n_clusters=16)
    m = np.zeros(shape=(clustered.shape[0],), dtype=int)
    index = 0
    for i in clustered:
        m[index] = i[0]
        index += 1

    print("NMI: {0}".format(nmi(sampled_data[1], m)))


if __name__ == '__main__':
    np.random.seed(0)
    #hierarchal_cluster('./digits-embedding.csv', type=1)
    #hierarchal_cluster('./digits-embedding.csv', type=2)
    #hierarchal_cluster('./digits-embedding.csv', type=3)
    calcNMI('./digits-embedding.csv', type=1)
    calcNMI('./digits-embedding.csv', type=2)
    calcNMI('./digits-embedding.csv', type=3)






