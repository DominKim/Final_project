# -*- coding: utf-8 -*-
import numpy as np
import pylab as pl
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans as KMeansGood
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("C:/ITWILL/Work/Final_Project/clustering/data/final_data_tot_only_untuk_clustering.csv", encoding='euc-kr')
df.drop("Unnamed: 0", axis=1, inplace=True)
df.info()

df[df['회사명']=='유아이디']  # 7602(인덱스)
df.drop(7602, axis=0,inplace=True)

df_cluster_data = df.drop(["회사명", "결산기준일"], axis=1)
df_cluster_data.info()


len(df_cluster_data['자산총계'])
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
df_cluster_data = ss.fit_transform(df_cluster_data)
#df_cluster_data.describe()

class KMeans(BaseEstimator):

    def __init__(self, k, max_iter=100, random_state=0, tol=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol

    def _e_step(self, X):
        self.labels_ = euclidean_distances(X, self.cluster_centers_,
                                     squared=True).argmin(axis=1)

    def _average(self, X):
        return X.mean(axis=0)

    def _m_step(self, X):
        X_center = None
        for center_id in range(self.k):
            center_mask = self.labels_ == center_id
            if not np.any(center_mask):
                # The centroid of empty clusters is set to the center of
                # everything
                if X_center is None:
                    X_center = self._average(X)
                self.cluster_centers_[center_id] = X_center
            else:
                self.cluster_centers_[center_id] = \
                    self._average(X[center_mask])

    def fit(self, X, y=None):
        n_samples = X.shape[0]
        vdata = np.mean(np.var(X, 0))

        random_state = check_random_state(self.random_state)
        self.labels_ = random_state.permutation(n_samples)[:self.k]
        self.cluster_centers_ = X[self.labels_]

        for i in range(self.max_iter):
            centers_old = self.cluster_centers_.copy()

            self._e_step(X)
            self._m_step(X)

            if np.sum((centers_old - self.cluster_centers_) ** 2) < self.tol * vdata:
                break

        return self

class KMedians(KMeans):

    def _e_step(self, X):
        self.labels_ = manhattan_distances(X, self.cluster_centers_).argmin(axis=1)

    def _average(self, X):
        return np.median(X, axis=0)

class FuzzyKMeans(KMeans):

    def __init__(self, k, m=2, max_iter=100, random_state=0, tol=1e-4):
        """
        m > 1: fuzzy-ness parameter
        The closer to m is to 1, the closter to hard kmeans.
        The bigger m, the fuzzier (converge to the global cluster).
        """
        self.k = k
        assert m > 1
        self.m = m
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol

    def _e_step(self, X):
        D = 1.0 / euclidean_distances(X, self.cluster_centers_, squared=True)
        D **= 1.0 / (self.m - 1)
        D /= np.sum(D, axis=1)[:, np.newaxis]
        # shape: n_samples x k
        self.fuzzy_labels_ = D
        self.labels_ = self.fuzzy_labels_.argmax(axis=1)

    def _m_step(self, X):
        weights = self.fuzzy_labels_ ** self.m
        # shape: n_clusters x n_features
        self.cluster_centers_ = np.dot(X.T, weights).T
        self.cluster_centers_ /= weights.sum(axis=0)[:, np.newaxis]

    def fit(self, X, y=None):
        n_samples, n_features = X.shape
        vdata = np.mean(np.var(X, 0))

        random_state = check_random_state(self.random_state)
        self.fuzzy_labels_ = random_state.rand(n_samples, self.k)
        self.fuzzy_labels_ /= self.fuzzy_labels_.sum(axis=1)[:, np.newaxis]
        self._m_step(X)

        for i in range(self.max_iter):
            centers_old = self.cluster_centers_.copy()

            self._e_step(X)
            self._m_step(X)

            if np.sum((centers_old - self.cluster_centers_) ** 2) < self.tol * vdata:
                break

        return self
    

kmedians = KMedians(k=5)
kmedians.fit(np.array(df_cluster_data))

y_pred = kmedians.labels_
df_cluster_data = pd.DataFrame(df_cluster_data)
df_cluster_data["cluster"] = y_pred
print(pd.Series(y_pred).value_counts())


import matplotlib.pyplot as plt

plt.scatter(df_cluster_data.iloc[:,0], df_cluster_data.iloc[:, 1], c= df_cluster_data["cluster"])








# 그룹화 >> 규모가 너무 큰 기업 삭제

cluster_grp = df_cluster_data.groupby('cluster')
cluster_grp.size()
'''
1    50
2    34
3    66
dtype: int64
'''
cluster_grp.describe()
'''
          count      mean       std  ...       50%       75%        max
cluster                              ...                               
0        2265.0 -0.078789  0.030400  ... -0.092321 -0.062330   0.089142
1         770.0  0.155162  0.121301  ...  0.115324  0.248671   0.761200
2        8197.0 -0.136577  0.008252  ... -0.141562 -0.135560  -0.092944
3         157.0  5.907458  6.016319  ...  3.278131  4.477865  24.596822
4         243.0  1.033071  0.443547  ...  0.956686  1.403887   3.064398
'''


#########################################################################################

# 클러스터 3에 있는 초우량 기업은 데이터에서 삭제 후, 다시 분석하기로 한다.
df_cluster_data_new = df_cluster_data.copy()
df_cluster_data_new.groupby('cluster')
df_cluster_data_new = df_cluster_data_new[df_cluster_data_new['cluster'] != 3]
df_cluster_data_new.groupby('cluster').describe()


# 분석 다시.
kmedians = KMedians(k=5)
kmedians.fit(np.array(df_cluster_data_new))

y_pred = kmedians.labels_
df_cluster_data_new = pd.DataFrame(df_cluster_data_new)
df_cluster_data_new["cluster"] = y_pred
print(pd.Series(y_pred).value_counts())


import matplotlib.pyplot as plt

plt.scatter(df_cluster_data_new.iloc[:,0], df_cluster_data_new.iloc[:, 1], c= df_cluster_data_new["cluster"])


cluster_grp_new = df_cluster_data_new.groupby('cluster')
cluster_grp_new.size()
'''
0     797
1    1609
2    5100
3    1014
4    2955
'''
cluster_grp_new.describe()
'''
              0                      ...         2                    
          count      mean       std  ...       50%       75%       max
cluster                              ...                              
0         797.0 -0.137390  0.001233  ... -0.143007 -0.141238 -0.137475
1        1609.0 -0.138626  0.002237  ... -0.138443 -0.135695 -0.122299
2        5100.0 -0.135182  0.010064  ... -0.141826 -0.132091 -0.092944
3        1014.0  0.365408  0.445897  ...  0.190331  0.429194  3.064398
4        2955.0 -0.093406  0.037444  ... -0.104406 -0.073789  0.089142
'''



#########################################################################################

# 클러스터 3에 있는 우량 기업은 데이터에서 삭제 후, 다시 분석하기로 한다.
df_cluster_data_latest = df_cluster_data_new.copy()
df_cluster_data_latest.groupby('cluster')
df_cluster_data_latest = df_cluster_data_latest[df_cluster_data_latest['cluster'] != 3]
df_cluster_data_latest.groupby('cluster').describe()


# 분석 다시.
kmedians = KMedians(k=5)
kmedians.fit(np.array(df_cluster_data_latest))

y_pred = kmedians.labels_
df_cluster_data_latest = pd.DataFrame(df_cluster_data_latest)
df_cluster_data_latest["cluster"] = y_pred
print(pd.Series(y_pred).value_counts())


import matplotlib.pyplot as plt

plt.scatter(df_cluster_data_latest.iloc[:,0], df_cluster_data_latest.iloc[:, 1], c= df_cluster_data_latest["cluster"])

cluster_grp_latest = df_cluster_data_latest.groupby('cluster')
cluster_grp_latest.size()
'''
0    5100
1     544
2    1241
3     621
4    2955
'''

cluster_grp_latest.describe()
'''
              0                      ...         2                    
          count      mean       std  ...       50%       75%       max
cluster                              ...                              
0        5100.0 -0.135182  0.010064  ... -0.141826 -0.132091 -0.092944
1         544.0 -0.140033  0.001213  ... -0.142605 -0.141678 -0.139567
2        1241.0 -0.136804  0.001238  ... -0.139554 -0.136616 -0.126104
3         621.0 -0.139450  0.001864  ... -0.138056 -0.135502 -0.122299
4        2955.0 -0.093406  0.037444  ... -0.104406 -0.073789  0.089142
'''





#########################################################################################

# 클러스터 3에 있는 준우량 기업은 데이터에서 삭제 후, 다시 분석하기로 한다.
df_cluster_data_final = df_cluster_data_latest.copy()
df_cluster_data_final.groupby('cluster')
df_cluster_data_final = df_cluster_data_final[df_cluster_data_final['cluster'] != 3]
df_cluster_data_final.groupby('cluster').describe()


# 분석 다시.
kmedians = KMedians(k=5)
kmedians.fit(np.array(df_cluster_data_final))

y_pred = kmedians.labels_
df_cluster_data_final = pd.DataFrame(df_cluster_data_final)
df_cluster_data_final["cluster"] = y_pred
print(pd.Series(y_pred).value_counts())


import matplotlib.pyplot as plt

plt.scatter(df_cluster_data_final.iloc[:,0], df_cluster_data_final.iloc[:, 1], c= df_cluster_data_final["cluster"])

cluster_grp_final = df_cluster_data_final.groupby('cluster')
cluster_grp_final.size()
'''
0    1435
1    1546
2    1177
3    2955
4    2727
'''

cluster_grp_final.describe()
'''
          count      mean       std  ...       50%       75%       max
cluster                              ...                              
0        1435.0 -0.128308  0.005210  ... -0.138394 -0.134988 -0.129600
1        1546.0 -0.137257  0.001450  ... -0.140604 -0.137461 -0.126104
2        1177.0 -0.123962  0.005288  ... -0.124041 -0.118224 -0.092944
3        2955.0 -0.093406  0.037444  ... -0.104406 -0.073789  0.089142
4        2727.0 -0.144172  0.002506  ... -0.145392 -0.143123 -0.130141
'''




#########################################################################################

# 클러스터 3에 있는 중 규모 기업은 데이터에서 삭제 후, 다시 분석하기로 한다.
df_cluster_data_end = df_cluster_data_final.copy()
df_cluster_data_end.groupby('cluster')
df_cluster_data_end = df_cluster_data_end[df_cluster_data_end['cluster'] != 3]
df_cluster_data_end.groupby('cluster').describe()


# 분석 다시.
kmedians = KMedians(k=5)
kmedians.fit(np.array(df_cluster_data_end))

y_pred = kmedians.labels_
df_cluster_data_end = pd.DataFrame(df_cluster_data_end)
df_cluster_data_end["cluster"] = y_pred
print(pd.Series(y_pred).value_counts())


import matplotlib.pyplot as plt

plt.scatter(df_cluster_data_end.iloc[:,0], df_cluster_data_end.iloc[:, 1], c= df_cluster_data_end["cluster"])

cluster_grp_end = df_cluster_data_end.groupby('cluster')
cluster_grp_end.size()
'''
0    2723
1    1033
2     748
3    1435
4     946
'''

cluster_grp_end.describe()
'''
          count      mean       std  ...       50%       75%       max
cluster                              ...                              
0        2723.0 -0.131510  0.007528  ... -0.134919 -0.125456 -0.092944
1        1033.0 -0.142296  0.002655  ... -0.144512 -0.142623 -0.130141
2         748.0 -0.144145  0.001162  ... -0.143410 -0.141589 -0.132142
3        1435.0 -0.128308  0.005210  ... -0.138394 -0.134988 -0.129600
4         946.0 -0.146243  0.001102  ... -0.147239 -0.146454 -0.145178
'''