import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random

import warnings
warnings.simplefilter('ignore')




def grayscale():
    data = pd.read_csv('digits-raw.csv', header=None)
    image_id = data[0].to_numpy(copy=True)
    class_label = data[1].to_numpy(copy=True)
    data = data.drop([0, 1], axis=1)
    for i in range(0, 10):
        index = random.choice(np.argwhere(class_label == i))
        arr = np.reshape(data.loc[index].to_numpy(copy=True), (28, 28))

        plt.imshow(arr, cmap='gray')
        plt.title('GreyScale #{0} from digits-raw.csv'.format(i))
        plt.show()


def visualize_points():
    data = pd.read_csv('digits-embedding.csv', header=None)
    values = np.random.randint(low=0, high=data.shape[0], size=1000)

    x = []
    y = []
    labels = {}
    for val in values:
        label = data.loc[val, 1]
        if '{0}x'.format(label) not in labels.keys():
            labels['{0}x'.format(label)] = []
            labels['{0}y'.format(label)] = []
        labels['{0}x'.format(label)].append(data.loc[val, 2])
        labels['{0}y'.format(label)].append(data.loc[val, 3])
        x.append(data.loc[val, 2])
        y.append(data.loc[val, 3])

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    class_labels = [0,1,2,3,4,5,6,7,8,9]
    plt.figure(figsize=(12, 8))
    for co, cl in zip(colors, class_labels):
        plt.scatter(labels['{0}x'.format(cl)], labels['{0}y'.format(cl)], c='tab:{0}'.format(co), s=2, label=str(cl))
    #plt.scatter(labels['0x'], labels['0y'], c='tab:blue', s=2, label='blue')
    #plt.scatter(x[label == 1], y[label == 1], c='tab:orange', s=2, label='orange')
    plt.legend()
    plt.show()












np.random.seed(0)
#grayscale()
visualize_points()
