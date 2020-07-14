# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 13:44:47 2020

@author: user
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

class KMedians(KMeans):

    def _e_step(self, X):
        self.labels_ = manhattan_distances(X, self.cluster_centers_).argmin(axis=1)

    def _average(self, X):
        return np.median(X, axis=0)




# dataset load
df = pd.read_csv("C:\\ITWILL\\Work\\Final_Project\\clustering\\data\\ratio_df_dropna.csv", encoding='euc-kr')
df.drop("Unnamed: 0", axis=1, inplace=True)
df.info()

df_cluster_data = df.drop(['종목코드', '회사명', '업종', '업종명', '결산기준일', '보고서종류', 
               '자산총계', '유동자산', '비유동자산', '부채총계', '유동부채', '비유동부채', 
               '자본총계', '이익잉여금(결손금)'], axis=1)
df_cluster_data.info()






# kmedian 객체  생성(n_clusters=9)
kmedian = KMedians(n_clusters=9, n_init = 10, verbose=1)
kmedian.fit(np.array(df_cluster_data))

y_pred = kmedian.labels_
y_pred



# sillhouette & inertia
kmedian.inertia_  # 1713.4949185140924

silhouette_score(df_cluster_data, kmedian.labels_)  #  0.2414476385255036








# kmedian&silhouette&inertia for문(nclusters=2~20)
ki = []
type(ki)
ksil = []
type(ksil)
for i in range(2, 31) :
    print(type(ki))
    n_clusters = i
    kmedian=KMedians(n_clusters=n_clusters, n_init=10)
    kmedian.fit(np.array(df_cluster_data))
    print(type(ki))
    
    ki.append(float(kmedian.inertia_))
    print(type(ki))
    ksil.append(silhouette_score(df_cluster_data, kmedian.labels_))
    
    print(i, "번째 이너셔 :", kmedian.inertia_)
    print(i, '번째 실루엣 :', silhouette_score(df_cluster_data, kmedian.labels_))
    
ki
ksil

plt.plot(ki)
plt.show()
plt.plot(ksil)
plt.show()


