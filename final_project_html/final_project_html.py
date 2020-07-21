# final_project_html

from flask import Flask, render_template, request
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
import seaborn as sns
from io import BytesIO



from flask import Markup
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


def financial_statement_barplot(df, cname, recomm_data):
    # barplot 그릴 회사 데이터
    selected_company_df = df[df['회사명'] == cname]
    selected_date = selected_company_df['결산기준일'].max()
    selected_company_df = selected_company_df[selected_company_df['결산기준일'] == selected_date]
    selected_company_df = selected_company_df[
        ['회사명', '유동자산비율', '현금비율', '금융자산비율', '비유동자산비율', '유형자산비율', '무형자산비율', '유동부채비율', '금융부채비율', '비유동부채비율', '부채비율',
         '이익잉여금비율', '자본비율', 'cluster']]
    recomm_cols = list(selected_company_df.columns)
    top_3 = recomm_data[recomm_cols]
    top_3 = top_3.iloc[:3]
    cluster_df = df[recomm_cols]
    # cluster별 평균
    cluster_num = df['cluster'][(df['회사명'] == cname) & (df['결산기준일'] == selected_date)].values[0]
    cluster_mean = cluster_df[cluster_df['cluster'] == cluster_num].agg('mean')
    # barplot
    img = BytesIO()
    plt.figure(figsize=(30, 10))
    sns.barplot(x=cluster_mean.index, y=cluster_mean.values)
    sns.violinplot(x=selected_company_df.columns[1:13], y=selected_company_df.values[0][1:13], linewidth=5, color='black', label=cname)
    sns.lineplot(data=top_3.iloc[0, 1:13].astype("float").values, label=top_3.iloc[0, 0], color='orange', marker='o')
    sns.lineplot(data=top_3.iloc[1, 1:13].astype("float").values, label=top_3.iloc[1, 0], color='blue', marker='o')
    sns.lineplot(data=top_3.iloc[2, 1:13].astype("float").values, label=top_3.iloc[2, 0], color='green', marker='o')
    plt.ylim(bottom=0.0, top=1.5)
    plt.xticks(fontsize=25, rotation=20)
    plt.yticks(fontsize=20)
    plt.title(f"{cname}와/과 재무상태가 유사한 회사 TOP3\n(막대 : 클러스터 평균, 선 : 선택한 회사 데이터)", fontsize=30)
    plt.legend(loc='upper right', fontsize=20)
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url



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

        sql = f"select cluster from fs where 회사명 = '{cname}'"
        cursor.execute(sql)
        row = cursor.fetchall()
        cluster = row[0]

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

        sql2 = f"select 결산기준일, 자산총계, 유동자산비율, 유형자산비율, 무형자산비율, 금융자산비율, 자본비율, 부채비율 from fs where 회사명 = '{cname}'"
        cursor.execute(sql2)
        data4 = cursor.fetchall()


        plot_url = financial_statement_barplot(df, cname, data3)

        return render_template("/page2.html", dataset=data5, size=size, title=cname, cluster=cluster, plot_url=plot_url, dataset2=data4)


# 프로그램 시작점
if __name__ == "__main__" :
    app.run(host = '172.20.10.8', port=16) # application 실행

