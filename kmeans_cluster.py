import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt

import warnings
warnings.simplefilter('ignore')

def find_nearest_centroid(x, y, c_x, c_y):
    distance = 1000000000
    cluster = -1
    for i in range(len(c_x)):
        dist = np.sqrt((c_x[i] - x) ** 2 + (c_y[i] - y) ** 2)
        if dist < distance:
            distance = dist
            cluster = i

    return cluster


def get_new_centroids(m, data):
    centroidsx = []
    centroidsy = []
    for i in set(m):
        indices = np.where(m == i)
        cluster = data.iloc[indices[0],:]
        centroidsx.append(np.mean(cluster[2]))
        centroidsy.append(np.mean(cluster[3]))

    return centroidsx, centroidsy


def wc_ssd(c_x, c_y, data, m):
    distance_sum = 0

    i = 0
    for index, row in data.iterrows():
        cluster = m[i]
        distance_sum += (c_x[cluster] - row[2]) ** 2 + (c_y[cluster] - row[3]) ** 2
        i += 1

    return distance_sum



def sc(c_x, c_y, data, m):
    S_i = 0
    i_ = 0
    for index, row in data.iterrows():
        A, B = 0, 0
        indices = np.where(m == m[i_])

        same_cluster = data.iloc[indices[0], :]
        for i in same_cluster:
            A += np.sqrt((row[2] - data.iloc[i][2]) ** 2 + (row[3] - data.iloc[i][3]) ** 2)

        A = A / len(same_cluster)

        nearest_cluster = -1
        nearest_cluster_distance = 1000000000

        for i in range(len(c_x)):
            if i != m[i_]:
                cluster_dist = np.sqrt((row[2] - c_x[i]) ** 2 + (row[3] - c_y[i]) ** 2)
                if cluster_dist < nearest_cluster_distance:
                    nearest_cluster_distance = cluster_dist
                    nearest_cluster = i

        indices = np.where(m == nearest_cluster)
        other_cluster = data.iloc[indices[0], :]

        for i in other_cluster:
            B += np.sqrt((row[2] - data.iloc[i][2]) ** 2 + (row[3] - data.iloc[i][3]) ** 2)

        B = B / len(other_cluster)

        S_i += (B - A) / max(B, A)
        i_ += 1

    return S_i / data.shape[0]



def calc_entorpy(data):
    values, counts = np.unique(data, return_counts = True)
    probs = []
    entropy = 0
    for i in range(len(values)):
        probs.append(counts[i]/data.shape[0])

    pk = np.array(probs)

    return sp.stats.entropy(pk=pk, base=2)


def calc_mutual_info(c, g):
    data = pd.DataFrame({'c': c, 'g': g}, columns=['c', 'g'])
    c_cnts = data['c'].value_counts()
    g_cnts = data['g'].value_counts()

    mi = 0
    pts = data.shape[0]
    for i in c_cnts.keys():
        cnts = data[data['c'] == i]['g'].value_counts()
        i_ = 0
        #print(cnts)
        for index, value in cnts.items():
            mi += value/pts * np.log2((value / pts) / (c_cnts[i] / pts * g_cnts[index] / pts))
            i_ += 1

    return mi


def nmi(c, g):
    ent_c = calc_entorpy(c)
    ent_g = calc_entorpy(g)
    #print(ent_c, ent_g)
    mi = calc_mutual_info(c, g)
    #print(mi)

    return mi / (ent_c + ent_g)



def kmeans(dataFileName, K=10, max_iter=50, type=0):
    data = pd.read_csv(dataFileName, header=None)
    if type == 2:
        data = data[(data[1] == 2) | (data[1] == 4) | (data[1] == 6) | (data[1] == 7)]
    if type == 3:
        data = data[(data[1] == 6) | (data[1] == 7)]

    centroids = np.random.choice(data.shape[0], K, replace=False)
    centroidsx = []
    centroidsy = []
    i = 0
    for c in centroids:
        centroidsx.append(data.iloc[c, 2])
        centroidsy.append(data.iloc[c, 3])
        i += 1
    del i
    del centroids

    m = np.zeros(shape=(data.shape[0],), dtype=int)


    for i in range(0, max_iter):
        clusters = {}
        m = np.zeros(shape=(data.shape[0],), dtype=int)
        for j in range(0, K):
            clusters[j] = []

        i = 0
        for index, row in data.iterrows():
            m[i] = find_nearest_centroid(row[2], row[3], c_x=centroidsx, c_y=centroidsy)
            i += 1

        centroidsx, centroidsy = get_new_centroids(m, data)

    metric1 = wc_ssd(centroidsx, centroidsy, data, m)
    metric2 = sc(centroidsx, centroidsy, data, m)
    metric3 = nmi(data[1].to_numpy(copy=True, dtype=int), m)
    if type == 0:
        print("WC-SSD: {0:3f}".format(metric1))
        print("SC: {0:3f}".format(metric2))
        print("NMI: {0:3f}".format(metric3))
    else:
        return metric1, metric2, metric3, m


def change_k(type):
    K = [2, 4, 8, 16, 32]
    wc_lst = []
    s_c_lst = []
    for k in K:
        wc, s_c, _, _ = kmeans(dataFileName='./digits-embedding.csv', K=k, max_iter=50, type=type)
        wc_lst.append(wc)
        s_c_lst.append(s_c)

    plt.plot(K, wc_lst, color='blue', label="WC-SSD")
    plt.xlabel('# number of Clusters')
    plt.ylabel('WC-SSD score')
    plt.legend()
    plt.show()

    plt.plot(K, s_c_lst, color='red', label='SC')
    plt.xlabel('# number of Clusters')
    plt.ylabel('SC score')
    plt.legend()
    plt.show()

    wc_dict = {}
    sc_dict = {}

    for k in K:
        wc_dict[k] = []
        sc_dict[k] = []

    for i in range(0,10):
        np.random.seed()
        for k in K:
            wc, s_c, _, _ = kmeans(dataFileName='./digits-embedding.csv', K=k, max_iter=50, type=type)
            wc_dict[k].append(wc)
            sc_dict[k].append(s_c)

    wc_dict_avg = []
    wc_dict_sd = []
    sc_dict_avg = []
    sc_dict_sd = []
    for k in K:
        wc_dict_avg.append(np.mean(wc_dict[k]))
        sc_dict_avg.append(np.mean(sc_dict[k]))
        wc_dict_sd.append(np.std(wc_dict[k]))
        sc_dict_sd.append(np.std(sc_dict[k]))

    print('WC-SSD AVG for each K: {0}'.format(wc_dict_avg))
    print('WC-SSD STD for each K: {0}'.format(wc_dict_sd))

    print('SC     AVG for each K: {0}'.format(sc_dict_avg))
    print('SC     STD for each K: {0}'.format(sc_dict_sd))


def visualize(dataFileName, K, type):
    _, _, nmi, m = kmeans(dataFileName,K=K, type=type)
    print("NMI: {0:3f}".format(nmi))

    values = np.random.randint(low=0, high=m.shape[0], size=1000)

    data = pd.read_csv(dataFileName, header=None)
    if type == 2:
        data = data[(data[1] == 2) | (data[1] == 4) | (data[1] == 6) | (data[1] == 7)]
    if type == 3:
        data = data[(data[1] == 6) | (data[1] == 7)]

    x = []
    y = []
    labels = {}
    for val in values:
        label = m[val]
        if '{0}x'.format(label) not in labels.keys():
            labels['{0}x'.format(label)] = []
            labels['{0}y'.format(label)] = []
        labels['{0}x'.format(label)].append(data.iloc[val, 2])
        labels['{0}y'.format(label)].append(data.iloc[val, 3])
        x.append(data.iloc[val, 2])
        y.append(data.iloc[val, 3])

    class_labels = [*range(K)]
    print(class_labels)
    for i in class_labels:
        if '{0}x'.format(i) not in labels.keys():
            labels['{0}x'.format(i)] = []
            labels['{0}y'.format(i)] = []
    plt.figure(figsize=(12, 8))
    #print(colors, class_labels, labels.keys())
    for cl in class_labels:
        plt.scatter(labels['{0}x'.format(cl)], labels['{0}y'.format(cl)], s=2, label=str(cl)) #c='tab:{0}'.format(co),

    plt.legend()
    plt.show()









if __name__ == '__main__':

    np.random.seed(0)
    #kmeans(dataFileName='./digits-embedding.csv', K=10, max_iter=50, type=0)
    visualize(dataFileName='./digits-embedding.csv', K=8, type=1)

#change_k(type=1)
#change_k(type=2)
#change_k(type=3)



