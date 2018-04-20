#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 15:51:54 2018

@author: shanthakumarp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Mall_Customers.csv")
X1= dataset.iloc[:,[3,4]].values


# using dendrogram to find optimum no. of cluster
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X1, method='ward'))

plt.title('Dendrogram')
plt.ylabel('Euclidean Distance')
plt.xlabel('customers')
plt.show()

# Fitting dataset using hierachical Agglomerative clustering

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean",linkage='ward')
y1_hc = hc.fit_predict(X1)

# visualizing datapoints and clusters
plt.scatter(X1[y1_hc==0,0], X1[y1_hc==0,1], s=100, c="red", label="cluster1")
plt.scatter(X1[y1_hc==1,0], X1[y1_hc==1,1], s=100, c="blue", label="cluster2")
plt.scatter(X1[y1_hc==2,0], X1[y1_hc==2,1], s=100, c="green", label="cluster3")
plt.scatter(X1[y1_hc==3,0], X1[y1_hc==3,1], s=100, c="black", label="cluster4")
plt.scatter(X1[y1_hc==4,0], X1[y1_hc==4,1], s=100, c="cyan", label="cluster5")
plt.title("HC clustering")
plt.ylabel("Amound spending")
plt.xlabel("Annual Income")
plt.legend()
plt.show()




