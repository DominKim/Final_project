# -*- coding: utf-8 -*-
from sklearn.decomposition import PCA
import pandas as pd

df = pd.read_csv("C:\\ITWILL\\Work\\Final_Project\\clustering\\data\\ratio_df_dropna.csv", encoding='euc-kr')
df.drop("Unnamed: 0", axis=1, inplace=True)
df.info()


df_cluster_data = df.drop(['종목코드', '회사명', '업종', '업종명', '결산기준일', '보고서종류', 
               '자산총계', '유동자산', '비유동자산', '부채총계', '유동부채', '비유동부채', 
               '자본총계', '이익잉여금(결손금)'], axis=1)
df_cluster_data.info()
'''
 0   유동자산per자산총계        11263 non-null  float64
 1   비유동자산per자산총계       11263 non-null  float64
 2   유동부채per자산총계        11263 non-null  float64
 3   비유동부채per자산총계       11263 non-null  float64
 4   부채총계per자산총계        11263 non-null  float64
 5   이익잉여금(결손금)per자산총계  11263 non-null  float64
 6   자본총계per자산총계        11263 non-null  float64
 '''

len(df_cluster_data['유동자산per자산총계'])  # 11263



pca = PCA(n_components=2)
model = pca.fit_transform(df_cluster_data)
model

plt.scatter(model[:, 0], model[:, 1], c=y_pred, cmap=plt.cm.Set1)
plt.show()
