from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# 모델 가져오기
print(os.getcwd())
model = pickle.load(open('pg_rf.pkl','rb'))

# 플라스크 앱 객체 지정
app = Flask(__name__)

# 플라스크 앱의 루트 디렉터리를 초기화
@app.route('/')
def main():
    return render_template('start.html')

# flask app 기동
# 예측하기 선택했을 때, 구동되는 부분 만들기

@app.route('/home', methods=['POST']) 
def home():
    dat1 = request.form['val1']
    dat2 = request.form['val2']
    dat3 = request.form['val3']
    dat4 = request.form['val4']
    val1 = float(dat1); 
    val2 = float(dat2); 
    val3 = float(dat3); 
    val4 = float(dat4);
    
    arr = np.array([[dat1, dat2, dat3, dat4]])
    pred = model.predict(arr)
    app.logger.info(arr)
    return render_template('after.html', data=pred)


# 실행시 플라스크 앱 기동
if __name__ == "__main__":
    app.run(debug=True)