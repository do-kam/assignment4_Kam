#sources https://medium.com/@sametgirgin/hierarchical-clustering-model-in-5-steps-with-python-6c45087d4318
#https://github.com/do-kam/assignment4_Kam
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
daten = pd.read_csv('input.csv', delimiter =";", header =None, skiprows =2)
#if Excel is in German and the Decimalpoint is a "," instead of ".": use this
#daten = pd.read_csv('input.csv', delimiter =";", header =None, decimal = ",", skiprows =2)
X = np.array(daten)

#create a scatter plot
plt.figure()
plt.scatter(X[:,0],X[:,1], color='red')
plt.title('Original data')
plt.xlabel('x-axis')
plt.ylabel('distances')
plt.show()

#fitting hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage ='ward')
y_hc=hc.fit_predict(X)

#visualizing the clusters
plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=100, c='green', label ='Cluster 3')
plt.title('Hierarchical Clustering Model')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()


#using dendogram to find the optimal numbers of clusters (in our case we know we want 3 clusters)

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X, method  = "ward"))
plt.title('Dendrogram')
plt.ylabel('Euclidean distances')
plt.show()
