import requests
import random
from bs4 import BeautifulSoup
import os
import torch
import streamlit as st
import time
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
from langchain_core.callbacks.base import BaseCallbackHandler

from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# OpenAI API 키를 환경 변수에서 로드
openai_api_key = os.getenv("OPENAI_API_KEY")

# JSON 문서 로드 및 처리
file_path = 'merged_data.json'
# JSONLoader의 jq_schema 설정
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
    "questioner": {
        "name": .questioner_name,
        "ID": .questioner_ID,
        "ISNI": .questioner_ISNI,
        "affiliation": .questioner_affiliation,
        "position": .questioner_position
    },
    "questioner_name": .questioner_name,
    "questioner_ID": .questioner_ID,
    "questioner_ISNI": .questioner_ISNI,
    "questioner_affiliation": .questioner_affiliation,
    "questioner_position": .questioner_position
    ,
    "question": {
        "tag": .question.tag,
        "comment": .question.comment,
        "keyword": .question.keyword
    },
    "answerer": {
        "name": .answerer_name,
        "ID": .answerer_ID,
        "ISNI": .answerer_ISNI,
        "affiliation": .answerer_affiliation,
        "position": .answerer_position
    },
    "answerer_name": .answerer_name,
    "answerer_ID": .answerer_ID,
    "answerer_ISNI": .answerer_ISNI,
    "answerer_affiliation": .answerer_affiliation,
    "answerer_position": .answerer_position,
    "answer": {
        "tag": .answer.tag,
        "comment": .answer.comment,
        "keyword": .answer.keyword
    }
}
"""
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
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)

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

# 시작 타이틀
st.title('안녕하세요! 국회 회의록 검색 서비스 챗봇입니다')

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = [] 

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

# 대화창 메세지 스트리밍 class
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text  
    
    def on_llm_new_token(self,token:str,**kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# 대화 내용을 출력하는 함수
def print_messages():
    """대화 내용을 출력하는 함수"""
    for msg in st.session_state["messages"]:
        with st.chat_message(msg['role']):
            steram_handler = StreamHandler(st.empty())
            st.write(msg['content'])

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
