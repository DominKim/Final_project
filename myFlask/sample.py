import flask
print(flask.__version__)  # 1.1.2  : 버전정보가 나오므로 패키지 정상설치됨
from flask import Flask
from flask import Flask, render_template


# flask application
app = Flask(__name__)  # 생성자 -> object(app)  : 플라스크 웹

# 함수 장식자 : 사용자 요청 url -> 함수 호출
@app.route('/')  # http://localhost/   #  http://127.0.0.1:5000/  : 기본 url
def hello() :
    return render_template('./index_briefcase.html')

# 프로그램 시작점
if __name__ == "__main__" :
    app.run()  # application 실행
