# -*- coding: utf-8 -*-
"""
k-means 클러스터 개수(silhouette) 찾기
"""

import numpy as np
from sklearn.metrics import silhouette_samples
from matplotlib import cm
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd




################################### 클러스터 별 실루엣 밀도그래프 ##################################

def plotSilhouette(X, y_km):
    cluster_labels = np.unique(y_km)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(X, y_km, metric="euclidean")
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []    
    
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(i / n_clusters)
        
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
                 edgecolor='none', color=color)
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_silhouette_vals)
    
    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color='red', linestyle='--')
    plt.yticks(yticks, cluster_labels+1)
    plt.ylabel("클러스터")
    plt.xlabel("실루엣 계수")
    plt.show()
    
    


# 1. dataset load
df = pd.read_csv("C:\\ITWILL\\Work\\Final_Project\\clustering\\data\\ratio_df_dropna_cluster.csv", encoding='euc-kr')
df.drop("Unnamed: 0", axis=1, inplace=True)
df.info()
df = df.drop(['종목코드', '회사명', '업종', '업종명', '결산기준일', '보고서종류', 
               '자산총계', '유동자산', '비유동자산', '부채총계', '유동부채', '비유동부채', 
               '자본총계', '이익잉여금(결손금)'], axis=1)


# 2. X, y(y는 아직 어떻게 설정할지 모르겠음)
X = df.copy()
X.drop(["회사명", "결산기준일"], axis=1, inplace=True)

# X 정규화
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)
X = pd.DataFrame(X)
X.describe()
'''
                  0             1             2
count  1.163300e+04  1.163300e+04  1.163300e+04
mean  -3.017790e-17  9.626351e-18  1.189344e-17
std    1.000043e+00  1.000043e+00  1.000043e+00
min   -1.273885e-02 -2.175222e-02 -1.166490e-02
25%   -1.257058e-02 -1.585058e-02 -1.159100e-02
50%   -1.238132e-02 -1.539452e-02 -1.148452e-02
75%   -1.167017e-02 -1.401871e-02 -1.103963e-02
max    1.078223e+02  1.076787e+02  1.078377e+02
'''

# 3. clustering_temp
n_clusters = range(2,10)
# X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5,
#                   shuffle=True, random_state=0)

km = KMeans(n_clusters=5, random_state=0)
y_km = km.fit_predict(X)
pd.DataFrame(y_km).head(30)
pd.Series(y_km).value_counts()


for n in n_clusters :
    km = KMeans(n_clusters=n)
    y_km = km.fit_predict(X)
    vc = pd.Series(y_km).value_counts()
    
    print(plotSilhouette(X, y_km))


# create silhouette plot
plotSilhouette(X, y_km)















########################################### 실루엣/클러스터 표 #####################################







n_clusters = range(2,20)
#n_clusters = 5

# km = KMeans(n_clusters=2, random_state=0)
# y_km = km.fit_predict(X)




def SilhouettePerCluster(X, y_km):
        
    silhouette_vals = silhouette_samples(X, y_km, metric="euclidean")
    silhouette_avg = np.mean(silhouette_vals)
    
    return silhouette_avg
    
    
silhouette_per_cluster = []
for n in n_clusters :
        
    km = KMeans(n_clusters=n, random_state=0)
    y_km = km.fit_predict(X)
    
    sil_result = SilhouettePerCluster(X, y_km)
    silhouette_per_cluster.append(sil_result)
    print(n, "번째 실루엣 확인 완료")

print(silhouette_per_cluster)

plt.plot(n_clusters, silhouette_per_cluster)


















