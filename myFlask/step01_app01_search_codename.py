'''
1. templates 파일 작성
 - 사용자 요청과 서버의 응답을 작성하는 html file
2. static 파일 작성
 - 정적 파일 : image file, js, css 등
'''

from flask import Flask, render_template, request  # html 페이지 호출

# flask application
app = Flask(__name__)  # 생성자 -> object(app)

@ app.route('/')  # 기본 url : http://127.0.0.1:5000/
def index() :  # 호출 함수
    return render_template('./app01/index.html')  # 호출할 html 페이지

@app.route('/search', methods=['Get', 'POST'])  # http://127.0.0.1:5000/info
def search() :
    if request.method == 'POST' :
        name = request.form['name']
        config = {
            'host': '127.0.0.1',
            'user': 'scott',
            'password': 'tiger',
            'database': 'final',
            'port': 3306,
            'charset': 'utf8',
            'use_unicode': True}

        try:
            import pymysql
            # 1. db연동 객체
            conn = pymysql.connect(**config)
            # 2. cursor 객체 : sqp문 실행
            cursor = conn.cursor()

            ###########################################
            ### 1. 항목명으로 검색
            ###########################################
            #name = input("검색할 항목명 입력 :")

            sql = f"select * from FS_3s where code_name like '%{name}%'"
            cursor.execute(sql)
            data = cursor.fetchall()
            if data:
                for row in data :
                    print(row)

                print("조회된 주소 수 :", len(data))
                size = len(data)
            else:
                print('해당 항목 없음')
                size = 0

        except Exception as e:
            print("db 연동 오류 :", e)
            conn.rollback()

        finally:
            cursor.close();
            conn.close()

        return render_template('./app01/result.html',
                                name = name, size=size, data=data)

if __name__ == "__main__" :
    app.run()
    