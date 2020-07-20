# Bank Info
from flask import Flask, render_template, request
#from pytest import collect
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_similarity
import pandas as pd
import numpy as np
# db 연결 객체 생성 함수
def db_conn():
    import pymysql
    config = {'host': '127.0.0.1', 'user': 'scott', 'password': 'tiger',
                'database': 'work', 'port': 3306, 'charset': 'utf8', 'use_unicode': True}
    conn = pymysql.connect(**config)
    cursor = conn.cursor()
    return conn, cursor


def get_recommend_list2(dff, name):
    company = dff["회사명"].unique()
    lst_idx = []

    for name in company:
        day = list(dff[dff["회사명"] == name]["결산기준일"].sort_values())[-1]
        company_index = list(dff[(dff["회사명"] == name) & (dff["결산기준일"] == day)].index)[0]
        lst_idx.append(company_index)

    recom_df = dff.iloc[lst_idx, :]
    recom_df.reset_index(drop=True, inplace=True)

    cluster_num = recom_df[recom_df["회사명"] == name]["cluster"].values[0]
    new_df = recom_df[recom_df["cluster"] == cluster_num]
    new_df.reset_index(drop=True, inplace=True)
    cosine = cosine_similarity(new_df.iloc[:, 9:], new_df.iloc[:, 9:]).argsort()[:, ::-1]
    company_index = new_df[new_df["회사명"] == name].index.values

    sim_index = cosine[company_index, :6].reshape(-1)
    sim_index = sim_index[sim_index != company_index]

    result = new_df.iloc[sim_index, :].sort_values("자본총계", ascending=False)
    return result



app = Flask(__name__) # object -> app object
# 함수 장식자
@app.route('/') # http://127.0.0.1:5000
def index():
    return render_template("/page1.html")


@app.route('/result', methods = ['GET', 'POST'])
def result():
    if request.method == 'POST':
        cname = request.form['name']
        print("cname = ", cname) # cname =  아이톡시

        conn,cursor = db_conn()
        print("2")
        df = pd.read_sql("select * from fs_simple", conn)
        print(df)
        df2 = df.drop('index', axis=1, inplace=True)
        print(df2)

        data3 = get_recommend_list2(df, cname)
        #data4 = data3[['회사명'], ['업종명'], ['결산기준일'], ['자산총계'], ['자본총계'], ['부채총계']]
        data4 = data3.iloc[:, 0:6]
        data5 = np.array(data4)
        print('data5 =', data5)
        print('a = ', type(data5))

        size = len(data5)
        print('size =', size)
        '''
        conn, cursor = db_conn()
        sql = f"select 회사명, 업종명, 결산기준일, 자산총계, 자본총계, 부채총계 from fs_simple where 회사명 = '{cname}'"
        cursor.execute(sql)
        data4 = cursor.fetchall()
        size = len(data4)
        '''
        return render_template("/page2.html", dataset=data5, size=size, title=cname)


# 프로그램 시작점
if __name__ == "__main__" :
    app.run() # application 실행
