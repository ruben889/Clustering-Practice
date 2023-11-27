# This is a sample Python script.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# depending on the scatterpoints distance from the cluster give it a certain alpha value
# make a list of random color values, when a cluster gets a value remove it from the list#

class Kmeans:
    def __init__(self, k=3):
        self.k = k
        self.centroids = 0

    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point)**2, axis=1))

    def fit(self, data_set, iterations):
        self.centroids = np.random.uniform(np.amin(data_set, axis=0), np.amax(data_set, axis=0),
                                           size=(self.k, data_set.shape[1]))

        for _ in range(iterations):
            clusterList = []

            for point in data_set:
                dist = Kmeans.euclidean_distance(point, self.centroids)
                cluster_num = np.argmin(dist)
                clusterList.append(cluster_num)

            clusterList = np.array(clusterList)
            cluster_indices = []

            for i in range(self.k):
                cluster_indices.append(np.argwhere(clusterList == i))

            centers = []

            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    centers.append(self.centroids[i])
                else:
                    centers.append(np.mean(data_set[indices], axis=0)[0])

            if np.max(self.centroids - np.array(centers)) < 0.0001:
                break
            else:
                self.centroids = np.array(centers)

        return clusterList


if __name__ == '__main__':

    dataset = pd.read_csv(r"C:\Users\icand\Desktop\General Projects Directory\DataPractice02\banknote_authen.csv")

    print(dataset.shape[1])

    dataX = dataset['V1']
    dataY = dataset['V2']
    tempdata = []

    for x, y in zip(dataX, dataY):
        temp = [x, y]
        tempdata.append(temp)

    datapoints = np.array(tempdata)

    kmeans = Kmeans(k=4)

    labels = kmeans.fit(datapoints,500)
    labels = np.array(labels)
    print(labels)
    print(len(labels))

    outlierMax = dataset['V2'].max()
    outlierMin = dataset['V2'].min()

    print(outlierMax)
    print(outlierMin)

    meanx = np.mean(dataX)
    meany = np.mean(dataY)

    var_x = np.var(dataX)
    var_y = np.var(dataY)

    stdX = np.std(dataX)
    stdY = np.std(dataY)

    print("Variance of V1: " + str(var_x))
    print("Variance of V2: " + str(var_y))
    print("MeanX : " + str(meanx) + " MeanY: " + str(meany))
    print("Standard Deviation X: " + str(stdX) + "Standard Deviation Y:" + str(stdY))

    # label min V1 and maximum value V1
    # mean (red dot) labelled standard deviation of bright red 1 to -1 and weaker red 2 standard deviation 2 to -2

    # Blend the colors together, depending on their distance to the nearest centroid
    # Make your own Hashtable for colors


    plt.scatter(datapoints[:, 0], datapoints[:, 1], c=labels, alpha=0.75)
    plt.scatter(meanx, meany, c='black', label='mean')
    plt.text(meanx, meany, 'Mean', color='Red', fontsize=10)
    plt.axhline(y=outlierMax, color='red', alpha=.75, linestyle="dashed")
    plt.axhline(y=outlierMin, color='red', alpha=.75, linestyle="dashed")
    plt.scatter(kmeans.centroids[:,0], kmeans.centroids[:, 1], c=range(len(kmeans.centroids)), s=200, edgecolor='black')


    plt.legend(['V1,V2', 'Mean', 'maxV2 Value', 'minV2 Value'])
    plt.title('Banknote Authentication Wavelets')

    plt.xlabel("V1 Values")
    plt.ylabel('V2 Values')
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
