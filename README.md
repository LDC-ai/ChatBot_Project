# 국회 회의록 기반 의정활동 지원 및 대국민 알권리 보장 챗봇 서비스
---
## 1. 배경 및 목적
- 언어모델 기술을 활용하여 국회 회의록을 기반으로 지식을 검색하고 질의응답할 수 있는 인공지능 모델 및 서비스를 개발
- 국민의 알권리를 보장하고 국회의 입법 활동 투명성을 높임

## 2. 목표
- 현안 히스토리 QA : 특정 정책/법률 현안에 관련된 논의 내용을 검색하고 시간순 및 이슈별로 정리/요약하여 답변
- 평가방법 : 정보의 정확성(환각 포함 여부)과 누락된 정보 없는지에 대한 재현성

## 3. 대회 기간
- 참가신청 : 2024.08.23~2024.10.25

## 4. 내용

## 5. 프로젝트 담당 역할
- 챗봇 학습용 데이터 전처리
  <https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71795>
- 자연어처리 모델 및 챗봇 모델 개발
- 스트림릿을 통한 웹앱 구현

## 6. 프로세스

# 국회 회의록 처리 및 GPT-4 기반 응답 시스템 구현 과정

## 1. 데이터 처리 개요

여러 개로 나눠져 있는 국회 회의록 **JSON** 파일을 하나로 합쳐, **LangSmith JSON 로더**를 이용해 필요한 키들만 추출했습니다.  
이후 **GPU**를 이용해 **FAISS 파일**로 임베딩한 후 저장하였고, 이를 통해 **웹 서비스**에서 바로 사용할 수 있도록 가공했습니다.

## 2. API와 시스템 활용

### Retriever와 GPT-4 API 연동
- **Retriever**와 **GPT-4 API**를 사용하여 **JSON 파일**에서 필요한 국회 내용을 추출했습니다.
- 사용자의 질문에 따라 관련된 정보를 제공할 수 있도록 시스템을 구성했습니다.

---

## 3. 코드 버전별 작업 흐름

### 3.1 agent_hs.ipynb (ver.1)

- 기존에 사용하던 **PDF 로더 코드**를 **JSON 용**으로 약간 변경.
- 성능이 좋지 않았기 때문에 해당 파일은 남겨두고, **agent2.ipynb(ver.2)**로 작업을 이어감.

### 3.2 agent2.ipynb (ver.2)

- **JSON 로더**로 로드하는 키들을 **필요한 항목만** 남기고 최적화.
- **Retriever 도구**와 **시스템 프롬프트 최적화**를 통해 성능이 향상됨.

#### 제공된 정보 예시

만약, "특정 인물이 언급되거나 참여한 회의"를 찾는다면 다음과 같은 비교적 자세한 정보를 제공:

- **회의 번호**: 1000
- **회의 이름**: XX 위원회
- **세대 번호**: XX대
- **법률**: XXX부 관련 법률 개정
- **질문자**: XXX 위원
- **답변자**: XXX 차관

그러나, **대화 내용**은 충분히 자세하지 않았고, "마지막 회의는 언제였나?"라는 질문에 **2023년**까지의 자료가 있음에도 **2019년** 회의를 알려주는 등 성능이 충분하지 않음.

### 3.3 agent3.ipynb (ver.3)

- **도구 함수**를 수정하여 자세한 **JSON 파일 키 값**을 모두 입력.
- 하지만, 아무 정보도 찾지 못하여 **성능**이 완전히 무용지물이 됨.

### 3.4 agent4.ipynb (ver.4)

- **JSON 파일**을 **Hadoop에서 많이 사용하는 Parquet 파일**로 변환.
- **Pandas**를 사용해 읽어온 후 **GPU로 임베딩** 작업 진행.
- 이후의 과정은 **agent2 파일**과 유사하게 진행됨.
  
#### 성능 평가

- **대화 내용**은 agent2보다 잘 받아왔으나, **요약 정보** 제공 성능은 미흡함.

---

이와 같은 작업 흐름으로 국회 회의록 처리 및 응답 시스템을 구축했습니다. 각 버전별 개선점과 한계점을 통해 현재 시스템의 성능을 개선하고자 했습니다.


## 7. 발표자료
