import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_similarity
from sklearn.base import BaseEstimator

# conn,cursor = db_conn("oracle")
# a = pd.read_sql("select * from fs_simple", conn)
# def xx(x):
#     return x.read()

# a["회사명"] = a[["회사명", "결산기준일"]].agg(xx)
# a.info()
# a["cluster"][0]
# a["결산기준일"] = a["결산기준일"].agg(xx)

def recommand_systemd(df, com):

    lst_idx = []
    num = df[df["회사명"] == com]["cluster"].unique()[0]
    df = df[df["cluster"] == num]
    df.reset_index(drop = True, inplace = True)
    # 회사 고유값
    company = df["회사명"].unique()

    # 가장 최신분기에 해당하는 값 뽑기
    for name in company:
        
        day = list(df[df["회사명"] == name]["결산기준일"].sort_values())[-1]
        company_index = list(df[(df["회사명"] == name) & (df["결산기준일"] == day)].index)[0]
        lst_idx.append(company_index)
 
    recom_df = df.iloc[lst_idx, :]
    recom_df.reset_index(drop = True, inplace = True)

    # 코사인 값 구하기
    cosine = cosine_similarity(recom_df.iloc[:,6:], recom_df.iloc[:,6:]).argsort()[:,::-1]


    company_index = recom_df[recom_df["회사명"] == com].index.values
    
    sim_index = cosine[company_index, :-2].reshape(-1)
    sim_index = sim_index[sim_index != company_index]
    
    result = recom_df.iloc[sim_index, :-2].sort_values("부채총계per자산총계", ascending = False).head(10)
    return result

