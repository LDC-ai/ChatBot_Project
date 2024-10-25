# 국회 회의록 기반 의정활동 지원 및 대국민 알권리 보장 챗봇 서비스
---
## 1. 배경 및 목적
- 언어모델 기술을 활용하여 국회 회의록을 기반으로 지식을 검색하고 질의응답할 수 있는 인공지능 모델 및 서비스를 개발
- 국민의 알권리를 보장하고 국회의 입법 활동 투명성을 높임
- AI 기술을 활용한 실시간 검색 대화형 챗봇의 상용화 기술이 발전하여 국회의 회의록 자료를 보다 쉽고 편리하게 접근할 수 있는 방식을 개발하게 되었음
- 이를 통해 정보의 접근성을 높이고 기술을 활용한 공익의 증대를 추구함

## 2. 목표
- 현안 히스토리 QA : 특정 정책/법률 현안에 관련된 논의 내용을 검색하고 시간순 및 이슈별로 정리/요약하여 답변
- 평가방법 : 정보의 정확성(환각 포함 여부)과 누락된 정보 없는지에 대한 재현성
- 국회 회의록에 대한 배경지식이 없어도 단순한 단어나 기초정보에 관한 질문만으로 사용자가 쉽게 원하는 정보에 접근할 수 있도록 설계

## 3. 대회 기간
- 참가신청 : 2024.08.23~2024.10.25

## 4. 내용
![테스트이미지](https://github.com/LDC-ai/ChatBot_Project/blob/main/project/chatbot.jpg)
- 챗봇에 질문을 하면 질문 내용에 맞는 회의록 정보를 제공합니다.
- JSON 파일에 주어진 데이터를 활용하여 각 feature에 맞는 정보들을 제공합니다.
- 회의명, 위원회명, 질문자, 응답자, 질문 내용 및 응답 내용 외의 다른 정보 또한 확인할 수 있습니다.

## 5. 프로젝트 담당 역할
- 챗봇 학습용 데이터 전처리
  <https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71795>
- 자연어처리 모델 및 챗봇 모델 개발
- 스트림릿을 통한 웹앱 구현

## 6. 프로세스-국회 회의록 처리 및 GPT-4 기반 응답 시스템 구현 과정

### 6.1. 데이터 처리 개요

여러 개로 나눠져 있는 국회 회의록 **JSON** 파일을 하나로 합쳐, **LangSmith JSON 로더**를 이용해 필요한 키들만 추출했습니다.  
이후 **GPU**를 이용해 **FAISS 파일**로 임베딩한 후 저장하였고, 이를 통해 **웹 서비스**에서 바로 사용할 수 있도록 가공했습니다.

### 6.2. API와 시스템 활용

#### Retriever와 GPT-4 API 연동
- **Retriever**와 **GPT-4 API**를 사용하여 **JSON 파일**에서 필요한 국회 내용을 추출했습니다.
- 사용자의 질문에 따라 관련된 정보를 제공할 수 있도록 시스템을 구성했습니다.

---

### 6.3. 코드 버전별 작업 흐름

#### 6.3.1 agent_hs.ipynb (ver.1)

- 기존에 사용하던 **PDF 로더 코드**를 **JSON 용**으로 약간 변경.
- 성능이 좋지 않았기 때문에 해당 파일은 남겨두고, **agent2.ipynb(ver.2)**로 작업을 이어감.

#### 6.3.2 agent2.ipynb (ver.2)

- **JSON 로더**로 로드하는 키들을 **필요한 항목만** 남기고 최적화.
- **Retriever 도구**와 **시스템 프롬프트 최적화**를 통해 성능이 향상됨.

##### 제공된 정보 예시

만약, "특정 인물이 언급되거나 참여한 회의"를 찾는다면 다음과 같은 비교적 자세한 정보를 제공:

- **회의 번호**: 1000
- **회의 이름**: XX 위원회
- **세대 번호**: XX대
- **법률**: XXX부 관련 법률 개정
- **질문자**: XXX 위원
- **답변자**: XXX 차관

그러나, **대화 내용**은 충분히 자세하지 않았고, "마지막 회의는 언제였나?"라는 질문에 **2023년**까지의 자료가 있음에도 **2019년** 회의를 알려주는 등 성능이 충분하지 않음.

#### 6.3.3 agent3.ipynb (ver.3)

- **도구 함수**를 수정하여 자세한 **JSON 파일 키 값**을 모두 입력.
- 하지만, 아무 정보도 찾지 못하여 **성능**이 완전히 무용지물이 됨.

#### 6.3.4 agent4.ipynb (ver.4)

- **JSON 파일**을 **Hadoop에서 많이 사용하는 Parquet 파일**로 변환.
- **Pandas**를 사용해 읽어온 후 **GPU로 임베딩** 작업 진행.
- 이후의 과정은 **agent2 파일**과 유사하게 진행됨.
- **대화 내용**은 agent2보다 잘 받아왔으나, **요약 정보** 제공 성능은 미흡함.

#### 6.3.5 agent6.ipynb (ver.5)

- **JSON 파일**을 **JsonLoader가 아닌, 자체적인 코드를 이용해 회의록 1장당 1개의 Chunk로 분할하는 방식 사용** 전처리 진행.
- **LangGraph**모듈을 사용하여 **Workflow**를 형성, **GroundedCheck**기능을 이용하여, model의 답변과 데이터의 상관성을 평가함.

#### 6.3.6 agent5.ipynb (ver.6)

- **Ensemble Retriever**를 활용하여 **Bm25**와 **FAISS + MultiQuery**리트리버를 가중값(0.7,0.3)으로 결합한 Retriever를 사용하여 검색을 수행
-  **GroundedCheck**기능은 토큰값을 많이 소모하기 때문에 삭제하였음.
- 대신, 재귀과정**Workflow**를 설계하였으나, 성능이 낮음.
![테스트이미지](https://github.com/LDC-ai/ChatBot_Project/blob/main/project/workflow.jpg)

#### 6.3.7 PhoenixChatBot.py (ver.7)
- **streamlit**을 이용하여 웹에서 RAG기능을 기반으로한 챗봇을 설계
- **agent prompt**와 **agent tool의 description**에 한국어를 영문으로 번역하여 검색하도록 주문하여 검색 성능을 높임

#### 6.3.8 TruePhoenix.py (ver.8)
- **streamlit**라이브러리의 **st.sidebar**기능을 이용하여, 대화내용을 저장, 불러오기할 수 있도록 개선함
- **RAG retreiver**의 검색값(k)을 조정하여, 소모하는 토큰수를 줄임
![테스트이미지](https://github.com/LDC-ai/ChatBot_Project/blob/main/project/acfd218040b1e444.png)


이와 같은 작업 흐름으로 국회 회의록 처리 및 응답 시스템을 구축했습니다. 각 버전별 개선점과 한계점을 통해 현재 시스템의 성능을 개선하고자 했습니다.


## 7. 연구결과물

### 7.1. 데이터 및 모델
- 데이터 : 대회 주체측에서 제공한 국회 회의록 기반 지식 검색 데이터(https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71795)를 사용
- Embedding모델 : HuggingFaceEmbeddings의 ,'jhgan/ko-sroberta-multitask'모델 사용(https://huggingface.co/jhgan/ko-sroberta-multitask)
- LLM모델 : OPENAI사의 gpt-40-mini, gpt-4o
 
### 7.2. 학습/추론 수행방법
- 주어진 데이터의 json파일들을 하나의 파일로 병합
- 읽어온 병합본을 문자열로 치환하여 불필요하거나 중복되는 정보들을 삭제한 뒤, 회의록 문서 1개를 Chunk 1개로 크기로 분할
- 분할한 문서들을 FAISS 벡터로 임베딩하여, VectorStore에 저장
- **RAG(Retrieval-Augmented Generation,검색 증강 생성)방법**을 사용함
- Bm25 Retreiver를 이용해 문자열을 검색하고, FAISS Retreiver와 MultiQuerry Retreiver를 이용하여, 각각 검색을 진행한 뒤 각각 가중값(0.7,0.3)을 두어 Ensemble Retriever을 사용하여 검색을 진행함
- LLM 모델(Chat-gpt)을 이용하여 검색한 내용들을 취합하여 이용자의 요구에 맞는 내용으로 답변하도록 설계함

### 7.2. 실행환경
- 리눅스 클라우드 가상환경을 이용하여 전 과정을 수행함
- NVIDIA-SMI 535.183.06  Driver Version: 535.183.06  CUDA Version: 12.2


