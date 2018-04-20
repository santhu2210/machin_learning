#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 20:37:47 2018

@author: shanthakumarp
"""

#reset -f

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
X1 = dataset.iloc[:, [3,4]].values


from sklearn.cluster import KMeans 
wcss = []

# finding no. of k-cluster using wcss
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', n_init=10, max_iter=300, random_state=0)
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss)
plt.title('The Elbow method')
plt.xlabel('Number of cluster')
plt.ylabel('WCSS')
plt.show()

# apply cluster value to predit appropriate cluster
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10,max_iter=300, random_state=0)
y1_means = kmeans.fit_predict(X1)

# plot the clusters
plt.scatter(X1[y1_means==0, 0],X1[y1_means==0, 1],s=100, color='red', label='cluster1')
plt.scatter(X1[y1_means==1, 0],X1[y1_means==1, 1],s=100, color='black', label='cluster2')
plt.scatter(X1[y1_means==2, 0],X1[y1_means==2, 1],s=100, color='green', label='cluster3')
plt.scatter(X1[y1_means==3, 0],X1[y1_means==3, 1],s=100, color='blue', label='cluster4')
plt.scatter(X1[y1_means==4, 0],X1[y1_means==4, 1],s=100, color='cyan', label='cluster5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s=300, color='yellow', label='centroid')
plt.title('K-means Clustering')
plt.xlabel('Annual income')
plt.ylabel('Spending score')
plt.legend()
plt.show()











