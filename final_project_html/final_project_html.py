# Bank Info
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_similarity
import pandas as pd
# db 연결 객체 생성 함수
def db_conn():
    import pymysql
    config = {'host': '127.0.0.1', 'user': 'scott', 'password': 'tiger',
                'database': 'work', 'port': 3306, 'charset': 'utf8', 'use_unicode': True}
    conn = pymysql.connect(**config)
    cursor = conn.cursor()
    return conn, cursor

conn,cursor = db_conn()
sql = f"select *from fs_simple"
cursor.execute(sql)
row = cursor.fetchall()
row[0]
row2 = pd.DataFrame(row, header=True)
row2.columns

def get_recommend_list2(dff, name, top=10):
    lst_idx = []
    company = dff["회사명"].unique()

    for name in company:
        day = list(dff[dff.iloc[2] == name]["결산기준일"].sort_values())[-1]
        company_index = list(dff[(dff["회사명"] == name) & (dff["결산기준일"] == day)].index)[0]
        lst_idx.append(company_index)

    cluster_num = dff[dff["회사명"] == name]["cluster_23_last"].values[0]
    new_df = dff[dff["cluster_23_last"] == cluster_num]
    new_df.reset_index(drop=True, inplace=True)
    cosine = cosine_similarity(new_df.iloc[:, 9:], new_df.iloc[:, 9:]).argsort()[:, ::-1]
    company_index = new_df[new_df["회사명"] == name].index.values

    sim_index = cosine[company_index, :30].reshape(-1)
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
        name = request.form['name']
        conn,cursor = db_conn()
        sql = f"select *from fs_simple"
        cursor.execute(sql)
        row = cursor.fetchall()
        row2 = pd.DataFrame(row)
        data3 = get_recommend_list2(row2, name, top=10)
        data4 = data3.iloc[:, 0:5]


        size = len(data4)
        return render_template("/page2.html", dataset = data4, size=size)


# 프로그램 시작점
if __name__ == "__main__" :
    app.run(host = '172.20.10.8', port=16) # application 실행

