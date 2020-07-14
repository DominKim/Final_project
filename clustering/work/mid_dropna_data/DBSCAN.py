# -*- coding: utf-8 -*-
from sklearn.cluster import DBSCAN
import pandas as pd

# dataset load
df = pd.read_csv("C:\\ITWILL\\Work\\Final_Project\\clustering\\data\\ratio_df_dropna.csv", encoding='euc-kr')
df.drop("Unnamed: 0", axis=1, inplace=True)
df.info()

df_cluster_data = df.drop(['종목코드', '회사명', '업종', '업종명', '결산기준일', '보고서종류', 
               '자산총계', '유동자산', '비유동자산', '부채총계', '유동부채', '비유동부채', 
               '자본총계', '이익잉여금(결손금)'], axis=1)
df_cluster_data.info()



# DBSCAN 모델 객체
dbscan = DBSCAN(eps=1, min_samples=5)
dbscan
'''
DBSCAN(algorithm='auto', eps=0.1, leaf_size=30, metric='euclidean',
       metric_params=None, min_samples=5, n_jobs=None, p=None)
'''
dbscan.fit(df_cluster_data)
dbscan.labels_  #  array([-1, -1, -1, ..., -1, -1, -1], dtype=int64)
pd.Series(dbscan.labels_).value_counts()
pd.Series(dbscan.labels_).value_counts()[-1]


len(dbscan.core_sample_indices_)  # 3553

dbscan.components_



# Hyper Parameters


# 1) params 설정

eps = [0.01,0.05,0.1,0.15, 0.2, 0.25, 0.3,0.5,1]
min_samples = [5,10,20,30,40,50]
outliers = []

for ep in eps :
    for min_sample in min_samples :
        print('eps =', ep)
        print('min_samples =', min_sample)
        
        dbscan = DBSCAN(eps = ep, min_samples=min_sample)
        dbscan.fit(df_cluster_data)
        
        outliers.append(pd.Series(dbscan.labels_).value_counts()[-1])
        print("outliers(list) :", outliers)

'''
epc=1
min_samples=5 일 때,
outlier 15개로 최소!
'''
        
import matplotlib.pyplot as plt        
plt.plot(outliers)
outliers































