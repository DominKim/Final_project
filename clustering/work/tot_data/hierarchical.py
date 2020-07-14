# -*- coding: utf-8 -*-

import pandas as pd  # dataframe 생성
from sklearn.datasets import load_iris  # 5째 컬럼이 숫자로 되어있는.
from scipy.cluster.hierarchy import linkage, dendrogram  # 군집분석 관련 tool
import matplotlib.pyplot as plt  # 군집분석 결과를 산점도로 시각화



# 1. dataset load
df = pd.read_csv("C:/ITWILL/Work/Final_Project/clustering/data/final_data_tot_only_untuk_clustering.csv", encoding='euc-kr')
df.drop("Unnamed: 0", axis=1, inplace=True)
df.info()

df[df['회사명']=='유아이디']  # 7602(인덱스)
df.drop(7602, axis=0,inplace=True)

df_cluster_data = df.drop(["회사명", "결산기준일"], axis=1)
df_cluster_data.info()

df_cluster_data.drop(7602, axis=0,inplace=True)
len(pd.DataFrame(df_cluster_data.iloc[0]))

len(df_cluster_data['자산총계'])
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
df_cluster_data = ss.fit_transform(df_cluster_data)


# 2. 계층적 군집분석
clusters = linkage(y = df_cluster_data, method = 'complete', metric = 'euclidean')
'''
method = 'single'  :  단순연결
method = 'complete'  :  완전연결
method = 'average'  :  평균연결
'''
clusters.shape  # (11631, 4)  >> 클러스터를 생성할 때마다 하나씩 수가 줄어드는것은 1번째 관측치를 기준으로 나머지 관측치를,
                #              두번째 관측치를 기준으로 나머지 관측치에 대한 거리를 계산하기 때문
                #              유클리디안 거리 계산 결과는 이등변삼각형 모양으로 나타남.

clusters
'''
array([[1.01000000e+02, 1.42000000e+02, 0.00000000e+00, 2.00000000e+00],
       [7.00000000e+00, 3.90000000e+01, 1.00000000e-01, 2.00000000e+00],
       .
       .
       .]
'''



# 3. 덴드로그램 시각화
plt.figure(figsize=(100,20))
dendrogram(Z=clusters, leaf_font_size=20,)  # 끝에 쉼표 찍어줄 것.
plt.show()




# 4. 클러스터 자르기/평가 : 3단계 덴드로그램 결과로 판단한다
from scipy.cluster.hierarchy import fcluster  # cluster 자르기

# 1) 클러스터 자르기
cluster = fcluster(clusters, t=2, criterion='distance')
# t는 자를 클러스터 수, criterion은 자르는 기준(distance는 유클리디안 거리)
cluster
'''
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2,
       2, 3, 2, 3, 3, 2, 2, 2, 2, 3, 2, 3, 2, 3, 2, 2, 3, 3, 2, 2, 2, 2,
       2, 3, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 3], dtype=int32)
'''
pd.Series(cluster).value_counts()
len(cluster)

# 2) DF 칼럼 추가
df['cluster'] = cluster
df_cluster_data = pd.DataFrame(df_cluster_data)
df_cluster_data['cluster'] = cluster
df.info()
df.head()
df.tail()
df_cluster_data.info()
df_cluster_data.head()

# 3) 산점도 시각화 -> 1번째 칼럼을 x축, 3번째 컬럼을 y축으로, cluster별 색상지정하려고 함.

plt.scatter(x=df_cluster_data.iloc[:,0], y=df_cluster_data.iloc[:,1],
            c=df_cluster_data['cluster'], marker='o')
# plt.ylim()
# plt.xlim()
plt.legend(loc='best')
plt.show()


df[df['cluster']==8]

# 4) 관측치(species) vs 예측치(cluster라고 치자)
cross = pd.crosstab(df['회사명'], df['cluster'])
cross
'''
cluster   1   2   3
species            
0        50   0   0
1         0   0  50
2         0  34  16
'''

# 5) 군집별 특성분석
# DF.groupby('집단변수')

cluster_grp = iris_df.groupby('cluster')
cluster_grp.size()
'''
1    50
2    34
3    66
dtype: int64
'''
cluster_grp.mean()
'''     <-------------------- 집단들 ----------------------------->    집단
         sepal length (cm)  sepal width (cm)  ...  petal width (cm)   species
cluster                                       ...                            
1                 5.006000          3.428000  ...          0.246000  0.000000
2                 6.888235          3.100000  ...          2.123529  2.000000
3                 5.939394          2.754545  ...          1.445455  1.242424
'''



































