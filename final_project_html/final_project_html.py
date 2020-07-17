# Bank Info
from flask import Flask, render_template, request
# db 연결 객체 생성 함수
def db_conn():
    import pymysql
    config = {'host': '127.0.0.1', 'user': 'scott', 'password': 'tiger',
                'database': 'work', 'port': 3306, 'charset': 'utf8', 'use_unicode': True}
    conn = pymysql.connect(**config)
    cursor = conn.cursor()
    return conn, cursor

app = Flask(__name__) # object -> app object
# 함수 장식자
@app.route('/') # http://127.0.0.1:5000
def index():
    return render_template("/page1.html")

@app.route('/result', methods = ['GET', 'POST'])
def depositPro():
    if request.method == 'POST':
        ycode = int(request.form['code'])
        conn,cursor = db_conn()
        sql = f"select *from deposit where ycode = {ycode}"
        cursor.execute(sql)
        row = cursor.fetchall()
        if row:
            sql = f"""select b.bcode, b.bname, b.bjoin, d.ycode, d.yname, d.period, d.rate 
            from bank b inner join deposit d
            on b.bcode = d.bcode and d.ycode = {ycode}"""
            cursor.execute(sql)
            data = cursor.fetchall()
            size = len(data)
            return render_template("/page2.html", dataset = data, size=size)
        else :
            return render_template("/error_name.html", info="회사명을 다시 확인 바랍니다.")

# 프로그램 시작점
if __name__ == "__main__" :
    app.run(host = '172.20.10.8', port=16) # application 실행

