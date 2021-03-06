# app.py

from flask import Flask
from flask import request, redirect
from sqlalchemy.engine import create_engine
import pandas as pd
import cx_Oracle

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
# kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
# transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

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


# temp code --> 모델 불러오기
# CPU
device = torch.device("cpu")

# bert 모델, vocabulary 불러오기
bertmodel, vocab = get_pytorch_kobert_model()

# Setting parameters
max_len = 80
batch_size = 64
warmup_ratio = 0.1
num_epochs = 15
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

#토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=6,  ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

model = BERTClassifier(bertmodel, dr_rate=0.5)

# 정확도 측정을 위한 함수 정의
def calc_accuracy(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc

def softmax(vals, idx):
    valscpu = vals.cpu().detach().squeeze(0)
    a = 0
    for i in valscpu:
        a += np.exp(i)
    return ((np.exp(valscpu[idx]))/a).item() * 100

def testModel(model, seq):
    cate = ['기쁨', '불안', '슬픔', '당황', '상처', '분노']
    tmp = [seq]
    transform = nlp.data.BERTSentenceTransform(tok, max_len, pad=True, pair=False)
    tokenized = transform(tmp)

    model.eval()
    result = model(torch.tensor([tokenized[0]]).to(device), [tokenized[1]], torch.tensor(tokenized[2]).to(device))
    idx = result.argmax().cpu().item()
    # print("신뢰도는:", "{:.2f}%".format(softmax(result,idx)))
    return cate[idx]

@app.route("/emotions", methods=['GET','POST'])
def emotion():
    # 2. 여기가 홈페이지에서 보이는 부분.
    # 이클립스 서블릿이 안만들어져 있으니깐 여기서 표시하는건 단순한 String만 가능!
    # 1번에서 저장했던 변수에 .to_string() 을 써서 String 출력

    if request.method == 'POST':
        # 안드로이드에서 getParams()메소드 안에 작성한
        # parsms.put("num1", "데이터1");
        # params.put("num2", "데이터2");
        # 의 key값인 num1, num2를 통해서 데이터 접근하여 변수에 저장
        title = request.form['num1']
        content = request.form['num2']
        id = request.form['num3']

        # num2 = int(request.form['num2'])

        model.load_state_dict(torch.load('model_state_dict.pt', map_location=device))
        model.to(device)
        model.eval()

        # chatModel load


        # 딥러닝 모델을 함수로 구현한 후 결과값을 return하여 result변수에 저장
        # 딥러닝 모델은 아래 Cell에 만들 것
        result = testModel(model, content)
        # result2 = chatModel(charmodel, content)

        print(title)
        print(content)
        print(id)
        print(result)

        # DB 에 값넣는 파트
        s_vals = f"('{title}','{content}','{id}','{result}')"
        insert_sql_result = engine.execute(f"INSERT INTO TBL_DIARY (D_TITLE, D_CONTENT, M_ID, DEEP_EMOTION) VALUES {s_vals}")

        print(insert_sql_result)
        # 결과값 리턴하면 안드로이드 스튜디오 StringRequest 안에 있는 onResponse()로 값 전달
        return result





# host_addr = 'project-db-stu.ddns.net'
host_addr = '121.147.52.194'
port_num = 80

# 파이썬 명령어로 실행할 수 있음
if __name__ == '__main__':
    app.run(host=host_addr, port=port_num,  debug=True)


    # 명재 깃 on
