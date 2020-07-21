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



def get_recommend_list2(dff, cname):
    # 유사도 검사에 사용할 비율 컬럼 리스트 생성
    dff.drop('index', axis=1, inplace=True)
    cols = list(dff.columns)
    cols_use = []
    for col in cols:
        if '비율' in col:
            print(col)
            cols_use.append(col)

    company = dff["회사명"].unique()
    lst_idx = []

    for name in company:
        day = list(dff[dff["회사명"] == name]["결산기준일"].sort_values())[-1]
        company_index = list(dff[(dff["회사명"] == name) & (dff["결산기준일"] == day)].index)[0]
        lst_idx.append(company_index)

    recom_df = dff.iloc[lst_idx, :]
    recom_df.reset_index(drop=True, inplace=True)

    cluster_num = recom_df[recom_df["회사명"] == cname]["cluster"].values[0]
    new_df = recom_df[recom_df["cluster"] == cluster_num]
    new_df.reset_index(drop=True, inplace=True)

    dist = euclidean_distances(new_df.loc[:, cols_use], new_df.loc[:, cols_use]).argsort()[:, ::-1]
    company_index = new_df[new_df["회사명"] == cname].index.values

    sim_index = dist[company_index, :10].reshape(-1)
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
        conn,cursor = db_conn()

        df = pd.read_sql("select * from fs", conn)

        data3 = get_recommend_list2(df, cname)
        #data4 = data3[['회사명'], ['업종'], ['결산기준일'], ['자산총계'], ['자본총계'], ['부채총계']]
        data4 = data3.iloc[:, [0,1, 4, 6, 94, 170, 172]]
        def pre(x):
            x = x.replace("[", "")
            a = x.replace("]", "")
            return a
        data4["종목코드"] = data4["종목코드"].agg(pre)
        data5 = np.array(data4)
        size = len(data5)
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
    app.run(host = '172.20.10.8', port=16) # application 실행

