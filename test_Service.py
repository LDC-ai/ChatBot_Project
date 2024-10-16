# 필요한 모듈 import
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_teddynote.messages import AgentStreamParser
from langchain_core.documents import Document
import json
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# TAVILY_API_KEY 불러오기
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Streamlit 앱 설정
st.title("JSON 파일 기반 챗봇")
st.write("JSON 파일을 읽어와서 정보를 검색하는 AI 챗봇입니다.")

########## 1. 도구를 정의 ##########
### 1-1. Search 도구 ###
search = TavilySearchResults(k=6, tavily_api_key=tavily_api_key)

### 1-2. JSON 파일 문서 검색 도구 (Retriever) ###
# JSON 파일 로드
with open("label_data.json", "r", encoding="utf-8") as f:
    label_data_json = json.load(f)

# JSON 데이터를 Document 객체로 변환
documents = [Document(page_content=doc['context'], metadata={"filename": doc["filename"]}) for doc in label_data_json]

# 텍스트 분할기를 사용하여 문서를 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(documents)

# VectorStore를 생성
vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())

# Retriever를 생성
retriever = vector.as_retriever()

# JSON 파일 검색용 도구 생성
retriever_tool = create_retriever_tool(
    retriever,
    name="json_search",  
    description="use this tool to search information from the JSON document",
)

### 1-3. tools 리스트에 도구 목록을 추가 ###
tools = [search, retriever_tool]

########## 2. LLM 을 정의 ##########
llm = ChatOpenAI(model="gpt-4o", temperature=0)

########## 3. Prompt 를 정의 ##########
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. "
            "Make sure to use the `json_search` tool for searching information from the JSON document. "
            "If you can't find the information from the JSON document, use the `search` tool for searching information from the web.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

########## 4. Agent 를 정의 ##########
agent = create_tool_calling_agent(llm, tools, prompt)

########## 5. AgentExecutor 를 정의 ##########
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

########## 6. 채팅 기록을 수행하는 메모리를 추가 ##########
store = {}

# session_id 를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in store:
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]

# 채팅 메시지 기록이 추가된 에이전트를 생성
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

########## 7. Streamlit 인터페이스 ##########
# 채팅 입력 필드
user_input = st.text_input("당신의 질문을 입력하세요:")

# 세션 관리
session_id = "user_session"

if user_input:
    # 사용자 입력 처리
    response = agent_with_chat_history.invoke({"input": user_input, "chat_history": get_session_history(session_id)})
    
    # 출력 결과를 화면에 표시
    st.write(f"AI 응답: {response['output']}")
    
# 채팅 기록 표시 (선택사항)
if "chat_history" in store[session_id].to_dict():
    st.write("채팅 기록:")
    for message in store[session_id].to_dict()["chat_history"]:
        st.write(f"{message['role']}: {message['content']}")
