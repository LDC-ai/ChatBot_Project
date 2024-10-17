<<<<<<< HEAD
import os
import torch
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import JSONLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# OpenAI API 키를 환경 변수에서 로드
openai_api_key = os.getenv("OPENAI_API_KEY")

# JSON 문서 로드 및 처리
file_path = 'merged_data.json'
=======
import requests
import random
from bs4 import BeautifulSoup
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.document_loaders import PyMuPDFLoader
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_teddynote.messages import AgentStreamParser
from langchain.document_loaders import JSONLoader


# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
# API 키 정보 로드
load_dotenv()

# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install -qU langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력
logging.langsmith("ChatBot_Project")

import os
os.environ['LANGCHAIN_PROJECT'] = 'ChatBot_Project'

########## 1. 도구를 정의 ##########

### 1-1. Search 도구 ###
search = TavilySearchResults(k=6)

### 1-2. JSON 문서 검색 도구 (Retriever) ###
# JSON 파일 경로
file_path = 'merged_data.json'
# JSONLoader의 jq_schema 설정
>>>>>>> 9cc0f52 ([fix] streamlit 챗봇 파일업데이트)
jq_schema = """
.[] | {
    "filename": .filename,
    "original": .original,
    "id": .id,
    "date": .date,
    "conference_number": .conference_number,
    "question_number": .question_number,
    "meeting_name": .meeting_name,
    "generation_number": .generation_number,
    "committee_name": .committee_name,
    "meeting_number": .meeting_number,
    "session_number": .session_number,
    "agenda": .agenda,
    "law": .law,
    "qna_type": .qna_type,
    "context": .context,
    "context_learn": .context_learn,
    "context_summary": {
        "summary_q": .context_summary.summary_q,
        "summary_a": .context_summary.summary_a
    },
<<<<<<< HEAD
    "questioner": {
        "name": .questioner_name,
        "ID": .questioner_ID,
        "ISNI": .questioner_ISNI,
        "affiliation": .questioner_affiliation,
        "position": .questioner_position
    },
=======
    "questioner_name": .questioner_name,
    "questioner_ID": .questioner_ID,
    "questioner_ISNI": .questioner_ISNI,
    "questioner_affiliation": .questioner_affiliation,
    "questioner_position": .questioner_position
    ,
>>>>>>> 9cc0f52 ([fix] streamlit 챗봇 파일업데이트)
    "question": {
        "tag": .question.tag,
        "comment": .question.comment,
        "keyword": .question.keyword
    },
<<<<<<< HEAD
    "answerer": {
        "name": .answerer_name,
        "ID": .answerer_ID,
        "ISNI": .answerer_ISNI,
        "affiliation": .answerer_affiliation,
        "position": .answerer_position
    },
=======
    "answerer_name": .answerer_name,
    "answerer_ID": .answerer_ID,
    "answerer_ISNI": .answerer_ISNI,
    "answerer_affiliation": .answerer_affiliation,
    "answerer_position": .answerer_position,
>>>>>>> 9cc0f52 ([fix] streamlit 챗봇 파일업데이트)
    "answer": {
        "tag": .answer.tag,
        "comment": .answer.comment,
        "keyword": .answer.keyword
    }
}
"""

<<<<<<< HEAD
# JSON 파일 로드
loader = JSONLoader(file_path, jq_schema=jq_schema, text_content=False)
documents = loader.load()

# 텍스트 분할기 설정
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(documents)

# CUDA 사용 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 임베딩 모델 설정 (GPU 사용)
embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={'device': device}
)

# FAISS 인덱스 설정
index_path = 'faiss_index'

# VectorStore 생성 또는 로드
if os.path.exists(index_path):
    print("저장된 FAISS 인덱스를 로드합니다...")
    vectorstore = FAISS.load_local(
        index_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
else:
    print("FAISS 인덱스를 생성합니다...")
    vectorstore = FAISS.from_documents(split_docs, embedding_model)
    vectorstore.save_local(index_path)
# Retriever 생성
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# 도구 함수 정의
def json_search_tool(input_text):
    docs = retriever.get_relevant_documents(input_text)
    if docs:
        summaries = [doc.page_content for doc in docs]
        return '\n\n'.join(summaries)
    else:
        return "해당하는 정보를 찾을 수 없습니다."

# 도구 정의
json_search_tool = Tool(
    name="JSONSearch",
    func=json_search_tool,
    description=(
        "Use this tool to search for information about a specific person or issue from the JSON document. "
        "Provides a summary of fields such as conference number, meeting name, generation number, committee name, agenda, law, Q&A type, context, learning context, "
        "context summary, questioner, and question."
    )
)
# 도구 목록 생성
tools = [json_search_tool]

# LLM 정의
llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=openai_api_key)

# 메모리 설정
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 에이전트 초기화
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)

# 시스템 메시지 설정
agent.agent.llm_chain.prompt.messages[0].prompt.template = ("Be sure to answer in Korean"
    "You are an AI assistant that helps users find information about individuals or issues from the National Assembly JSON document. "
    "When a user asks about a specific person or issue, you should use the JSONSearch tool to find meeting information where that person is mentioned. "
    "Provide a summarized response including fields such as conference number, meeting name, generation number, committee name, agenda, law, Q&A type, context, learning context, context summary, questioner, and question. "
    "If you cannot find the information in the JSON document, politely inform the user that the information is not available. "
    "Do not mention the use of tools; just provide the necessary information."
)

=======
# JSONLoader를 사용하여 파일 로드
loader = JSONLoader(file_path, jq_schema=jq_schema, text_content=False)  # text_content=False 설정
documents = loader.load()

# 텍스트 분할기를 사용하여 문서를 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# 문서를 로드하고 분할
split_docs = loader.load_and_split(text_splitter)

# VectorStore를 생성
vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())

# Retriever를 생성
retriever = vector.as_retriever()

# retriever_tool 정의
retriever_tool = create_retriever_tool(
    retriever,
    name="json_search",  # 도구의 이름을 입력
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
            "You are a QA chatbot based on National Assembly meeting records, tasked with sorting data according to the given JSON file database schema. You will answer questions based on the attributes of each item and sort the data accordingly." 
            "Sorting data by a specific date"
            "Sorting data by a specific committee"
            "Sorting data by the name of the questioner or answerer"
            "Sorting data by agenda or legislation"
            "You are to sort the data based on the requested attribute and return the sorted results. All answers should be clear and consistent."
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

def get_session_history(session_ids):
    if session_ids not in store:
        store[session_ids] = ChatMessageHistory()
    return store[session_ids]

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

########## 7. Agent 파서를 정의 ##########
agent_stream_parser = AgentStreamParser()


>>>>>>> 9cc0f52 ([fix] streamlit 챗봇 파일업데이트)
# 시작 타이틀
st.title('안녕하세요! 국회 회의록 검색 서비스 챗봇입니다')

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = [] 

<<<<<<< HEAD
if "session_history" not in st.session_state:
    st.session_state["session_history"] = {}  # 세션 기록을 위한 딕셔너리

def chat_with_agent(user_input):
    # 에이전트와 대화하여 응답을 얻는 함수
    response = agent.run(input=user_input)
    return response

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in st.session_state.session_history:
        st.session_state.session_history[session_ids] = ChatMessageHistory()
    return st.session_state.session_history[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환

# 대화 내용을 출력하는 함수
def print_messages():
    """대화 내용을 출력하는 함수"""
    for msg in st.session_state["messages"]:
        st.chat_message(msg['role']).write(msg['content'])

# 사용자 입력 처리
user_input = st.chat_input('질문이 무엇인가요?')

if user_input:
    # 세션 ID를 설정
    session_id = "default_session"

    # 세션 기록 가져오기
    session_history = get_session_history(session_id)

    # 이전 대화 기록을 에이전트에 전달
    if session_history.messages:
        previous_messages = [{"role": msg['role'], "content": msg['content']} for msg in session_history.messages]
        response = chat_with_agent(user_input + "\n\nPrevious Messages: " + str(previous_messages))
    else:
        response = chat_with_agent(user_input)

    # 메시지를 세션에 추가
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.session_state["messages"].append({"role": "assistant", "content": response})

    # 세션 기록에 메시지를 추가
    session_history.add_message({"role": "user", "content": user_input})
    session_history.add_message({"role": "assistant", "content": response})

# 대화 내용 출력
print_messages()


=======
if "answer_list" not in st.session_state:
    st.session_state["answer_list"] = ['안녕'] 

def print_messages():
    """대화 내용을 출력하는 함수"""
    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg['content'])

# 사용자 입력 처리
start = st.chat_input('질문이 무엇인가요?')

response = agent_with_chat_history.stream(
    {"input":  f'{start}'},
    # 세션 ID를 설정
    # 여기서는 간단한 메모리 내 ChatMessageHistory를 사용하기 때문에 실제로 사용되지 않음
    config={"configurable": {"session_id": "abc123"}},
)

for step in response:
    agent_stream_parser.process_agent_steps(step)

if agent_stream_parser.final_response:
    st.chat_message("assistant").write(agent_stream_parser.final_response)
>>>>>>> 9cc0f52 ([fix] streamlit 챗봇 파일업데이트)
