## 달력이나 이외의 외적인 부분 깔끔하게 하기

#################################### import ####################################
import gc

# 대시보드
import streamlit as st
import pandas as pd
from IPython.core.display import HTML, display
from IPython.core import display
from PIL import Image
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
plt.rcParams['font.family'] = 'NanumGothic'

## KoBERT
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
import gluonnlp as nlp
import numpy as np
import datetime

## BERT
import urllib.request
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

## DB
from google.cloud import firestore

##################################### var ######################################
USER = 'user0001'  ## 임시 저장
d = datetime.datetime.now()
TODAY = str(d)[8:10]
MONTH = str(d)[5:7]
#YEAR 없음

#보류
emotions = ['짜증','기쁨','슬픔','중립','당황','불안']
MONTH_list = {'01': 'January', '02': 'February','03':'March','04':'April','05':'May','06':'June','07':'July','08':'August','09':'September','10':'October','11':'November','12':'December'}

## DB
db = firestore.Client.from_service_account_json("/content/drive/MyDrive/finalpjt-14eb3-firebase-adminsdk-2rcyn-aaaadce745.json")
docs = db.collection('user0001').get() # 일기 쓴 날짜 리스트 가져오기
doc_ref = db.collection("calendar").document(MONTH) # 달력 가져오기
doc = doc_ref.get()
read_cal = pd.DataFrame([doc.to_dict()['w0'][1:-1].split(' '),doc.to_dict()['w1'][1:-1].split(' '),doc.to_dict()['w2'][1:-1].split(' '),doc.to_dict()['w3'][1:-1].split(' '),doc.to_dict()['w4'][1:-1].split(' '),doc.to_dict()['w5'][1:-1].split(' ')])
read_cal.columns = ['MON','TUE','WED','THU','FRI','SAT','SUN']

## 경고 해제
pd.set_option('mode.chained_assignment',  None)

##################################### def ######################################
def add_bg_from_url():
  st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
      background-image: url("https://raw.githubusercontent.com/kimaenzu/finalPJT_st/main/image/smile.jpg");
      background-attachment: fixed;
      background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
  )
def sider_bg_from_url():
  st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"] {{
      background-image: url("https://raw.githubusercontent.com/kimaenzu/finalPJT_st/main/image/smile.jpg");
      background-attachment: fixed;
      background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
  )

sider_bg_from_url()
## 일기를 작성받는 함수
def input_emotion():
  st.text(f"안녕하세요 <{USER}>님! 오늘의 일기를 작성해주세요")
  
  message = st.text_area("일기 작성 칸") #### 작성칸 길이 조절

  if st.button("기록", key='message'):
    result = message.title()
    st.text(f"{MONTH}월 {TODAY}일 기록이 완료됐습니다.")
      ## 감정 이모티콘 출력
    emo = update_emo(predict(message))
      
      ## 위로 문장 출력
    answer = return_similar_answer(result)
    st.write(answer)
      ## DB에 일기 업데이트
    db.collection(u'user0001').document(str(d)[:10]).set('')  ## DB에 오늘 날짜 추가
    doc_ref = db.collection(USER).document(str(d)[:10])
    doc_ref.update({"emotion": emo, "daylog":message, "answer":answer}) 

    
## 일기쓴 날 스탬프 추가
def update_emo(emo):
  global USER, d
  global read_cal
  read_cal = read_cal.replace(TODAY[-2:], emo)

  ## DB에 update
  for i in range(6):
    doc_ref.update({
      f"w{i}": str(read_cal.loc[i].values).replace('\'', '')}) 
  return emo
  
## 달력에 이미지를 넣는 함수
def to_img_tag(path):
  return f'<img src="https://raw.githubusercontent.com/kimaenzu/finalPJT_st/main/image/{path}.png" width="50" >'

## 달력 가져와서 출력하기 : 아이콘을 눌러서 일기를 볼 수 있을까?
def calendar_emo(cal_mon):
  ## DB에서 달력 읽기 (1월 달력 calendar1)
  doc_ref = db.collection("calendar").document(cal_mon)
  doc = doc_ref.get()
  read_cal = pd.DataFrame([doc.to_dict()['w0'][1:-1].split(' '),doc.to_dict()['w1'][1:-1].split(' '),doc.to_dict()['w2'][1:-1].split(' '),doc.to_dict()['w3'][1:-1].split(' '),doc.to_dict()['w4'][1:-1].split(' '),doc.to_dict()['w5'][1:-1].split(' ')])
  read_cal.columns = ['MON','TUE','WED','THU','FRI','SAT','SUN']

  ## return으로 넘기고 필요할 때만 출력
  st.write(HTML(read_cal.to_html(escape=False,formatters={'MON':to_img_tag,'TUE':to_img_tag,'WED':to_img_tag,'THU':to_img_tag,'FRI':to_img_tag,'SAT':to_img_tag,'SUN':to_img_tag})))

## kobert 감정 분류 일기 작성 후 이미지 출력
def predict_img(emo):
    ## gif 해서 되면 gif로 (st.markdowm 사용)
  st.write(HTML(f'<img src="https://raw.githubusercontent.com/kimaenzu/finalPJT_st/main/image/{emo}.png" width="250">'))

def plot_emo(cal_mon):
  doc_ref = db.collection("calendar").document(cal_mon)
  doc = doc_ref.get()
  read_cal = pd.DataFrame([doc.to_dict()['w0'][1:-1].split(' '),doc.to_dict()['w1'][1:-1].split(' '),doc.to_dict()['w2'][1:-1].split(' '),doc.to_dict()['w3'][1:-1].split(' '),doc.to_dict()['w4'][1:-1].split(' '),doc.to_dict()['w5'][1:-1].split(' ')])
  read_cal.columns = ['MON','TUE','WED','THU','FRI','SAT','SUN']
  for emo in emotions :
    globals()[emo] = 0
    for col in read_cal.columns :
      try:
        globals()[emo] += read_cal[col].value_counts()[emo]
      except:
        a=0
  plot1 = pd.DataFrame({'emotion' : emotions, 'count': [짜증,기쁨,슬픔,중립,당황,불안]})
  st.write(plot1.T)

#    fig, ax = plt.subplots()
#    barplot = sns.barplot(x='emotion', y='count', data=plot1, ax=ax, palette='Set2')
#    fig = barplot.get_figure()
#    st.pyplot(fig)

#  col1, col2 = st.columns((1,1))
#  with col1:

  fig = px.pie(plot1, names='emotion', values='count', color_discrete_sequence=px.colors.qualitative.Pastel)
  st.plotly_chart(fig)

#  with col2:
#    st.write(plot1.max())
#    if plot1.max()[emo] == '짜증' :
#      st.write("짜증")
#    if plot1.max() == '기쁨' :
#      st.write("기쁨")
#  st.write(plot1)
  max_emotion = str(plot1[plot1['count'] == plot1['count'].max()]['emotion'].values)[2:-2]
  max_count = plot1[plot1['count'] == plot1['count'].max()]['count'].values

#  st.write(max_count)
#  st.write(plot1[plot1['count'] == plot1.max()])
#  st.write(plot1[plot1['count'].max()].columns)
  plot_bot(max_emotion, max_count)

def plot_bot(emo, count):
  if emo == '슬픔' :
    st.write(HTML(f'<img src="https://raw.githubusercontent.com/kimaenzu/finalPJT_st/main/image/통계_{emo}.png">'))
    if count > 28 :
      st.write("위험해요! 누군가에게 도움을 요청하세요!")
    elif count > 21:
      st.write("3주 넘게 우울")
    elif count > 14:
      st.write("유독 슬픈 달이네요 무슨일이 있으신가요?")
    else:
      st.write("우울하네요.. 좋은 일을 만들어 봐요")
  elif emo == '짜증' :
    st.write(HTML(f'<img src="https://raw.githubusercontent.com/kimaenzu/finalPJT_st/main/image/통계_{emo}.png">'))
    st.write("무언가 괴롭게 하는게 있나요 ?")
  elif emo == '기쁨' :
    st.write(HTML(f'<img src="https://raw.githubusercontent.com/kimaenzu/finalPJT_st/main/image/통계_{emo}.png">'))
    st.write("매일 매일 즐거운 하루를 보내시는 군요")
  elif emo == '중립' :
    st.write(HTML(f'<img src="https://raw.githubusercontent.com/kimaenzu/finalPJT_st/main/image/통계_{emo}.png">'))
    st.write("꾸준한 당신, 멋저요!")
  elif emo == '당황' :
    st.write(HTML(f'<img src="https://raw.githubusercontent.com/kimaenzu/finalPJT_st/main/image/통계_{emo}.png">'))
    st.write("무슨일이 일어나고 있는거죠?")
  elif emo == '불안' :
    st.write(HTML(f'<img src="https://raw.githubusercontent.com/kimaenzu/finalPJT_st/main/image/통계_{emo}.png">'))
    if count > 28 :
      st.write("상담이 필요합니다!")
    elif count > 21:
      st.write("불안이 길어집니다! 도와줄 사람을 찾아보세요")
    elif count > 14:
      st.write("걱정되는 일이 많으신가요?")
    else:
      st.write("무슨 일이 있으신가요?")



  
#################################### KoBERT ####################################
## GPU 설정
device = torch.device("cuda:0")
# device = torch.device('cpu')

class BERTClassifier(nn.Module):
  def __init__(self,
              bert,
              hidden_size = 768,
              num_classes=7,
              dr_rate=None,
              params=None):
    super(BERTClassifier, self).__init__()
    self.bert = bert
    self.dr_rate = dr_rate
                 
    self.classifier = nn.Linear(hidden_size , num_classes)
    if dr_rate:
      self.dropout = nn.Dropout(p=dr_rate)
    
  def gen_attention_mask(self, token_ids, valid_length):
    attention_mask = torch.zeros_like(token_ids)
    for i, v in enumerate(valid_length):
      attention_mask[i][:v] = 1
      return attention_mask.float()

  def forward(self, token_ids, valid_length, segment_ids):
    attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
    _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
    if self.dr_rate:
      out = self.dropout(pooler)
      return self.classifier(out)


## 4. 데이터 전처리(토큰화, 정수 인코딩, 패딩)
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

# Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

## 8. 결과물 테스트

## 감정 예측
def predict(predict_sentence):

    ## 모델 불러오기
    model = torch.load('/content/drive/MyDrive/Colab Notebooks/감정분석기/models/6emotions_model.pt')

    data = [predict_sentence, '0']
    dataset_another = [data]
    ## bertmodel의 vocabulary
    bertmodel, vocab = get_pytorch_kobert_model()

    ## 토큰화
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)
    
    model.eval()
 
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        test_eval=[]
        per_emo = []
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()
            per_emo.append(logits)
            if np.argmax(logits) == 0:
                test_eval.append("짜증")
            elif np.argmax(logits) == 1:
                test_eval.append("기쁨")
            elif np.argmax(logits) == 2:
                test_eval.append("불안")
            elif np.argmax(logits) == 3:
                test_eval.append("당황")
            elif np.argmax(logits) == 4:
                test_eval.append("슬픔")
            elif np.argmax(logits) == 5:
                test_eval.append("중립")

#        st.write("logits", per_emo[0])
#        emo_list = per_emo[0]
#        for _ in emo_list :
#            st.write(min(emo_list), _-min(emo_list))
 #           st.write((_-min(logits)/sum(emo_list)-min(logits)*7)*100)

        predict_img(test_eval[0])
        st.write(f"일기에서 {test_eval[0]}이 느껴집니다.")
 
        return (test_eval[0])

##################################### BERT #####################################

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

def return_similar_answer(input):
## 데이터 불러오기
## 모델 불러오기
    BERT_model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')  # BERT 모델

    QA = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/감정분석기/data/QA_175745.csv')
    npy = np.load('/content/drive/MyDrive/Colab Notebooks/감정분석기/data/Embedding_175745.npy', allow_pickle=True)
    QA['embedding'] = npy
    new_data = QA
    embedding = BERT_model.encode(input)
    new_data['score'] = new_data.apply(lambda x: cos_sim(x['embedding'], embedding), axis=1)
    answer = new_data.loc[new_data['score'].idxmax()]['A']
    del QA, npy, new_data
    gc.collect()
    return answer
#################################################################################
## Title
st.title("감정저장소")


## 메뉴 선택
add_selectbox = st.sidebar.selectbox("무엇이 궁금하세요?",("감정기록", "과거의 감정", "감정그래프"))
#new_selectbox = st.sidebar.selectbox("<<TIP! 이렇게 써보세요!>>",("<<TIP! 이렇게 써보세요!>>", "① 글쓰기 전 명상은 어떠신가요?", "② 일단 시작해 보세요! 쓰다보면 몰입될 거에요", "③ 일기에서도 나의 프라이버시를 보호하세요"), label_visibility='collapsed')
with st.sidebar:
    st.success("<<TIP! 이렇게 써보세요!>>")
    st.write("① 글쓰기 전 명상은 어떠신가요?")
    st.write("② 일단 시작해 보세요! 쓰다보면 몰입될 거에요")
    st.write("③ 일기에서도 나의 프라이버시를 보호하세요")
## 감정기록
if add_selectbox == "감정기록":
   add_bg_from_url()
   input_emotion()

## 과거의 감정
if add_selectbox == "과거의 감정":
  add_bg_from_url()
  st.subheader("과거의 감정")

	## 감정 스탬프
  MONTH_arr = ['', '01월','02월', '03월', '04월', '05월', '06월', '07월', '08월', '09월', '10월', '11월','12월']    ## 달 이름 깔끔하게 출력하기
  MONTH = st.selectbox("", MONTH_arr)
  if MONTH == '' :
    MONTH = MONTH_arr[2]
  st.text(f"감정 달력({MONTH})")
  calendar_emo(MONTH[:-1])

	## 지난 일기 보기  
  log_data = []
  log_data.append('')
  for doc in docs:
    log_data.append(doc.id)

  ch = st.selectbox("지난 일기를 보려면 날짜를 선택하세요", log_data)
  if ch != '':
    st.write(ch, " 일기를 불러왔습니다.")
    doc_ref = db.collection("user0001").document(ch)
    doc = doc_ref.get()
    st.write("===============")
    predict_img(doc.to_dict()['emotion'])
    st.write(doc.to_dict()['answer']) ## 공감문장
    st.write("===============")
    st.write(doc.to_dict()['daylog']) ## 일기 내용
    
  else :
    st.write('')

## 감정그래프
if add_selectbox == "감정그래프":
  st.subheader("감정그래프")
  MONTH_arr = ['', '01월','02월', '03월', '04월', '05월', '06월', '07월', '08월', '09월', '10월', '11월','12월']   
  MONTH = st.selectbox("", MONTH_arr)
  if MONTH == '' :
    MONTH = MONTH_arr[2]
  st.subheader(f"───── ❝ {MONTH_list.get(MONTH[:-1])} ❞ ─────")

  plot_emo(MONTH[:-1])
