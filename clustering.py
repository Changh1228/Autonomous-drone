#!/usr/bin/env python
from __future__ import print_function
import pandas as pd
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt

LABELS = ["airport", "residential", "dangerous_curve_left", "dangerous_curve_right", "junction", "road_narrows_from_left",
"road_narrows_from_right", "roundabout_warning", "follow_left", "follow_right", "no_bicycle", "no_heavy_truck", "no_parking",
"no_stopping_and_parking", "stop"]

class Clustering:
    """
    Localizer provides the function used to localize a detected object in the map frame.
    It has to be instantiated once when the node is created (i.e. not everytime the callback
    is invoked.).
    """
    def __init__(self): 
        self.path = "cluster_data.csv"

    def cluster(self):
        cluster_dataframe = pd.read_csv(self.path, skiprows = 0)
        d = {'Label': [],'ID': [], 'X': [], 'Y': [], 'Z': []}
        definitive_clusters = pd.DataFrame(data=d)

        begin = 0
        for label in LABELS:
            index_entry = 0
            data_sign = cluster_dataframe[cluster_dataframe.ID == label]
            #cluster_signs=[] # to include all the signs detected with their current cluster
            #d = {'ID': [], 'X': [], 'Y': [], 'Z': []}
            #cluster_signs = pd.DataFrame(data=d)
            average_clusters = []
            for index, element in data_sign.iterrows():
                clusters_index = 0
                if index_entry == 0:
                    cluster_signs = pd.DataFrame(data={'ID': [clusters_index], 'X': [element.x], 'Y': [element.y],
                                                       'Z': [element.z]}) # 0 for the first ID
                    average_clusters.append([clusters_index, element.x, element.y, element.z, 1]) # 1 because it has 1 element
                    index_entry = 1
                else:
                    flag_add_cluster = 1
                    for cluster in average_clusters:
                        distance = np.sqrt(np.power(element[1] - cluster[1], 2) + np.power(element[2] - cluster[2], 2) +
                                          np.power(element[3] - cluster[3], 2))
                        if distance < 1: # is less than 1 m apart from cluster average position
                            cluster_signs = cluster_signs.append({'ID': cluster[0], 'X': element.x, 'Y': element.y, 'Z':
                                element.z}, ignore_index=True)
                            average_clusters[cluster[0]][4] += 1 # add one element to the cluster
                            # we should update the average of the cluster
                            average_clusters[cluster[0]][1] = (element.x + average_clusters[cluster[0]][1])/ \
                                                              average_clusters[cluster[0]][4]
                            average_clusters[cluster[0]][2] = (element.y + average_clusters[cluster[0]][2]) / \
                                                              average_clusters[cluster[0]][4]
                            average_clusters[cluster[0]][3] = (element.z + average_clusters[cluster[0]][3]) / \
                                                              average_clusters[cluster[0]][4]
                            flag_add_cluster = 0
                            break
                            # TODO: Check if something better than break as could be another one closer
                    if flag_add_cluster == 1: # no other cluster is close to
                        clusters_index += 1
                        cluster_signs = cluster_signs.append({'ID': clusters_index, 'X': element.x, 'Y': element.y, 'Z': element.z}, ignore_index=True)  # 0 for the first ID
                        average_clusters.append([clusters_index, element.x, element.y, element.z, 1])


            
            # check number of clusters and how spread are they
            for index, elements_average in enumerate(average_clusters):
                if elements_average[4] > 2: # if seen more than 5 times
                    signs = cluster_signs[cluster_signs.ID == index]
                    for index_element, element_sign in signs.iterrows():
                        definitive_clusters = definitive_clusters.append({'Label': label, 'ID': index, 'X': element_sign.X,
                                                    'Y': element_sign.Y, 'Z': element_sign.Z}, ignore_index=True)

        sns.lmplot(x="X", y="Y", data=definitive_clusters, fit_reg=False, hue='Label', legend=False)
        plt.legend(loc='lower right')

"""
def main(args):
    algorithm = Clustering()
    algorithm.cluster()


if __name__ == '__main__':
    main(sys.argv)
"""


