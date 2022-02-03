# app.py

from flask import Flask
from sqlalchemy.engine import create_engine
import pandas as pd
import cx_Oracle

app = Flask(__name__)

# DB Connection Setting
DIALECT = 'oracle'
SQL_DRIVER = 'cx_oracle'
USERNAME = 'onion' #enter your username
PASSWORD = 'smhrd123' #enter your password
HOST = 'project-db-stu.ddns.net' #enter the oracle db host url
PORT = 1524 # enter the oracle port number
SERVICE = 'xe' # enter the oracle db service name
ENGINE_PATH_WIN_AUTH = DIALECT + '+' + SQL_DRIVER + '://' + USERNAME + ':' + PASSWORD +'@' + HOST + ':' + str(PORT) + '/?service_name=' + SERVICE

engine = create_engine(ENGINE_PATH_WIN_AUTH)

#test query
test_df = pd.read_sql_query('SELECT * FROM tbl_member', engine)
# insert_sql_result = engine.execute('INSERT ')
#

# temp code --> 모델 불러오기
# 1. 불러온 모델로 결과값을 변수에 저장.
# gii test
#model = torch.load('model_kobert.pt')

# result = model.eval()
#
@app.route('/')
def hello_world():
    # 2. 여기가 홈페이지에서 보이는 부분.
    # 이클립스 서블릿이 안만들어져 있으니깐 여기서 표시하는건 단순한 String만 가능!
    # 1번에서 저장했던 변수에 .to_string() 을 써서 String 출력

    # return result.to_string()
    return test_df.to_string()

#host_addr = 'project-db-stu.ddns.net'
host_addr = 'localhost'
port_num = 1524

#파이썬 명령어로 실행할 수 있음
if __name__ == '__main__':
    app.run(host=host_addr, port=port_num,  debug=True)
