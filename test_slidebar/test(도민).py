# Bank Info
from flask import Flask, render_template, request
import cx_Oracle
import os
import pandas as pd
import numpy as np
from recommand import recommand_systemd
from sqlalchemy import create_engine
from sqlalchemy.types import String, Date, DateTime
engine = create_engine('oracle+cx_oracle://scott:tiger@localhost:32772/?service_name=xe')




app = Flask(__name__) # object -> app object
# 함수 장식자
@app.route('/') # http://127.0.0.1:5000
def index():
    return render_template("/test2.html")

@app.route('/test2', methods = ['GET', 'POST'])
def depositPro():
    if request.method == 'POST':
        ycode = request.form['code']
        from database import db_conn

        conn,cursor = db_conn("oracle")
        a = pd.read_sql("select * from fs_simple", conn)
        def xx(x):
            return x.read()
        
        a["회사명"] = a["회사명"].apply(xx)
        a["결산기준일"] = a["결산기준일"].agg(xx)
        a["종목코드"] = a["종목코드"].agg(xx)
        b = a.copy()
        result = recommand_systemd(a, ycode)
        def pre(x):
            x = x.replace("[", "")
            a = x.replace("]", "")
            return a
        result["종목코드"] = result["종목코드"].agg(pre)
        lst = np.array(result.iloc[:, :7])
        lst
        return render_template("/test5.html", dataset = lst, title = ycode)
        # return render_template("/test5.html", dataset = [result.to_html(header = True)], title = ycode)

# 프로그램 시작점
if __name__ == "__main__" :
    app.run() # application 실행

