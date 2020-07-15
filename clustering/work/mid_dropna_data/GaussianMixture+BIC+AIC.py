# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 14:35:21 2020

@author: user
"""

import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


# dataset load
df = pd.read_csv("C:\\ITWILL\\Work\\Final_Project\\clustering\\data\\ratio_df_dropna.csv", encoding='euc-kr')
df.drop("Unnamed: 0", axis=1, inplace=True)
df.info()

df_cluster_data = df.drop(['종목코드', '회사명', '업종', '업종명', '결산기준일', '보고서종류', 
               '자산총계', '유동자산', '비유동자산', '부채총계', '유동부채', '비유동부채', 
               '자본총계', '이익잉여금(결손금)'], axis=1)
df_cluster_data.info()



# model 객체
gm = GaussianMixture(n_components=9, n_init=10)
gm.fit(df_cluster_data)

gm.means_

gm.weights_  # array([1.74803466e-01, 4.68721015e-02, 5.66001794e-03, 2.66358874e-04,1.19680086e-02, 3.10963062e-01, 5.91927528e-03, 5.19763572e-03,4.38350073e-01])
gm.weights_.sum()  #  0.9999999999999993
max(gm.weights_)  # 0.4383500733638386
min(gm.weights_)  # 0.0002663588741842895
gm.weights_.mean()

pd.DataFrame(gm.weights_).boxplot()

gm.converged_  # True
gm.n_iter_  # 32

y_pred_cluster_9 = gm.predict(df_cluster_data)
y_pred_cluster_10 = gm.predict(df_cluster_data)
y_pred_cluster_11 = gm.predict(df_cluster_data)
y_pred_cluster_12 = gm.predict(df_cluster_data)
y_pred_cluster_13 = gm.predict(df_cluster_data)
y_pred_cluster_14 = gm.predict(df_cluster_data)
y_pred_cluster_15 = gm.predict(df_cluster_data)
y_pred_cluster_16 = gm.predict(df_cluster_data)
y_pred_proba = gm.predict_proba(df_cluster_data)



# bic & aic
gm.bic(df_cluster_data)  # -408864.3325768434
gm.aic(df_cluster_data)  # -411231.6894664668

bic = []
aic = []
for i in range(2,101) :
    n_components = i
    gm = GaussianMixture(n_components=n_components, n_init=10)
    gm.fit(df_cluster_data)
    
    bic.append(gm.bic(df_cluster_data))
    aic.append(gm.aic(df_cluster_data))
    
    print(i,'번째 bic :', gm.bic(df_cluster_data))
    print(i,'번째 aic :', gm.bic(df_cluster_data))


plt.plot(bic)
plt.title("BIC minimum rage when n_cluster=11(2~100")
plt.plot(aic)
plt.title("AIC's differential coefficient starts to decrease from spot n_cluster=16")
'''
[해석]
bic는 cluster 10개 근방에서 최소,
aic는 cluster 16개 근방에서 기울기 급감'''












#################################### 시각화 #######################################
df.info()
df['cluster_9'] = y_pred_cluster_9
df['cluster_10'] = y_pred_cluster_10
df['cluster_11'] = y_pred_cluster_11
df['cluster_12'] = y_pred_cluster_12
df['cluster_13'] = y_pred_cluster_13
df['cluster_14'] = y_pred_cluster_14
df['cluster_15'] = y_pred_cluster_15
df['cluster_16'] = y_pred_cluster_16


df.to_csv("C:\\ITWILL\\Work\\Final_Project\\clustering\\data\\ratio_df_dropna_cluster_with_GMM_9_16.csv", encoding='euc-kr')


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure( figsize=(6,6))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.scatter(df['자본총계'],df['부채총계'],df['자산총계'],c=df['cluster_16'],alpha=0.5)
ax.set_xlabel('자본총계1')
ax.set_ylabel('부채총계2')
ax.set_zlabel('자산총계3')
plt.show()














