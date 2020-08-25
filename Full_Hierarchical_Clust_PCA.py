#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 00:55:59 2020

@author: jcs
"""


from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import InterclusterDistance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 

from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
#Import of required libraries 
import os, sys, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import epitopepredict as ep
import matplotlib.patches as mpatches 
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering 

#The following functions outline different encoding methods
#Note that as of now epitopepredict does not function on Windows OS
#Encoders are from http://dmnfarrell.github.io/bioinformatics/mhclearning

#One hot encode matrix 

def one_hot_encode(seq):
    codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    o = list(set(codes) - set(seq))
    s = pd.DataFrame(list(seq))    
    x = pd.DataFrame(np.zeros((len(seq),len(o)),dtype=int),columns=o)    
    a = s[0].str.get_dummies(sep=',')
    a = a.join(x)
    a = a.sort_index(axis=1)
    e = a.values.flatten()
    return e

#NLF matrix 

def nlf_encode(seq):    
    nlf = pd.read_csv('NLF.csv',index_col=0)
    x = pd.DataFrame([nlf[i] for i in seq]).reset_index(drop=True)  
    e = x.values.flatten()
    return e

#Blosum 62 matrix 

def blosum_encode(seq):
    
    blosum = ep.blosum62
    s=list(seq)
    x = pd.DataFrame([blosum[i] for i in seq]).reset_index(drop=True)
    e = x.values.flatten()    
    return e

#######CREATING DATAFRAME FOR CLUSTERING#########
df = pd.read_csv("aa_comp.csv") 
X = df.Sequence.apply(lambda x: pd.Series(one_hot_encode(x)),1) #place holder
title = 'Kmeans One Hot'

plt.clf()


##########Initial Dendogram############
plt.figure( figsize = (15,12))
plt.title (title+ " Non-PCA Dendrogram")
dend = shc.dendrogram(shc.linkage(X,method='ward'))
plt.savefig(title+ ' Dendrogram (Raw Data)')

#########CLUSTERING#############
H_cluster = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean',
                                  linkage='ward')
H_cluster.fit_predict(X)
cluster=H_cluster.labels_
clust_lst = []

##########PCA VARIANCE Analysis##########

#np.savetxt('X_std.csv',X_std, delimiter=',')
pca = PCA()
principalcomps = pca.fit_transform(X)


##########Cumulative Explained Variance#########
plt.clf()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.savefig('cumulative explained variance.png')
# plt.clf()
# feats = range(pca.n_components_)
# plt.bar(feats,pca.explained_variance_ratio_, color='black')
# plt.xlabel('PCA features')
# plt.ylabel('Variance %')
# plt.xticks(feats)
# plt.savefig("PCA_variance(just_sequence).png")

pca_df = pd.DataFrame(principalcomps)

#########PCA 2D plot no clustering######
plt.clf()

plt.scatter(pca_df[0],pca_df[1],alpha = .1)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D Principal Components')

plt.savefig(title+' PCA_plot_2d(no_clusters).png')

########PCA 3d plot no clustering ######
plt.clf()

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(pca_df[0], pca_df[1], pca_df[2], alpha = .1)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set(title = '3D Principal Components')
    
plt.savefig(title+' affimer_3D_pca(no clusters).png', dpi=300)

#######PCA 2D plot clustered from raw data#####
plt.clf()

plt.scatter(pca_df[0],pca_df[1], c =cluster,cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Principal Component Clustering')

plt.savefig(title+' PCA_plot_2d(norm_clusters).png')

#########PCA 3d plot from raw data######
clusters = 4



pca= PCA(3)
pca.fit(X)
pca_data=pd.DataFrame(pca.transform(X))


colors = list(zip(*sorted((
                        tuple(mcolors.rgb_to_hsv(
                              mcolors.to_rgba(color)[:3])), name)
                        for name, color in dict(
                                mcolors.BASE_COLORS,**mcolors.CSS4_COLORS
                                ).items())))[1]
skips = math.floor(len(colors[5 : -5])/clusters)
cluster_colors = colors[3: -3:skips]
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(pca_data[0], pca_data[1], pca_data[2],
            c = list(map(lambda label : cluster_colors[label],H_cluster.labels_)))
str_labels = list(map(lambda label: '% s' % label, H_cluster.labels_))

    
plt.savefig(title+' affimer_3D_pca(just_sequence).png', dpi=300)

#######PCA DF Slicing########
pca_df10 = pca_df.iloc[:,0:10]
print(pca_df10)

######PCA Dendrogram#########
plt.clf()
plt.figure( figsize = (15,12))
plt.title (title+ "PCA Dendrogram")
dend = shc.dendrogram(shc.linkage(pca_df10,method='ward'))
plt.savefig(title+ ' Dendrogram (PCA Data)')

########PCA Clustering######
kmeans = KMeans(n_clusters=3)
kmeans.fit(pca_df10)
cluster_pca=kmeans.labels_
clust_lst_pca = []
#######PCA 2D plot clustered from raw data#####
plt.clf()
colormap = np.array(['steelblue', 'purple','green'])
clust0 = mpatches.Patch(color = 'steelblue',label = 'Cluster 0')
clust1 = mpatches.Patch(color = 'purple',label = 'Cluster 1')
clust2 = mpatches.Patch(color = 'green',label = 'Cluster 2')
clust3 = mpatches.Patch(color = 'peru',label = 'Cluster 3')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Principal Component Clustering')
plt.scatter(pca_df[0],pca_df[1], c =colormap[cluster_pca], alpha = .1)
plt.legend(handles = [clust0,clust1,clust2,clust3], loc = 'upper right')

plt.savefig(title+' PCA_plot_2d(PCA_clusters).png')

#########PCA 3d plot Using PCA Clusters######
plt.clf()


fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
colormap = np.array(['steelblue', 'purple','green'])
clust0 = mpatches.Patch(color = 'steelblue',label = 'Cluster 0')
clust1 = mpatches.Patch(color = 'purple',label = 'Cluster 1')
clust2 = mpatches.Patch(color = 'green',label = 'Cluster 2')
clust3 = mpatches.Patch(color = 'peru',label = 'Cluster 3')
ax.scatter(pca_data[0], pca_data[1], pca_data[2], c = colormap[cluster_pca],
            alpha = .1)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set(title = '3D Principal Component Clustering')
ax.legend(handles = [clust0,clust1,clust2,clust3])
    
plt.savefig(title+' affimer_3D_pca(PCA_clusters).png', dpi=300)



#########################Cluster Groups in Raw data##########
for i in cluster:
    clust_lst.append(i)
print(cluster)
print(clust_lst)
df['Cluster Assignment']=clust_lst
for i in cluster_pca:
    clust_lst_pca.append(i)
df['PCA Cluster Assignment']= clust_lst_pca
c0 = 0
c1 = 0
c2 = 0
c3 = 0
seqCount = []
for c in clust_lst_pca:
    if c == 0:
        c0 = c0 + 1
    elif c== 1:
        c1 = c1 + 1
    elif c == 2:
        c2 = c2 + 1
    else:
        c3 = c3 + 1
seqCount.append(c0)
seqCount.append(c1)
seqCount.append(c2)
seqCount.append(c3)
print (seqCount)

df.to_csv(title+' Clustering Data.csv')







