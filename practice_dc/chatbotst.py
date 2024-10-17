# 필요한 모듈 import
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
from langchain.document_loaders import JSONLoader

# 나머지 코드는 동일하게 유지


########## 1. 도구를 정의 ##########

### JSON 문서 검색 도구 (Retriever) ###
# JSON 파일 경로
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
    "answer": {
        "tag": .answer.tag,
        "comment": .answer.comment,
        "keyword": .answer.keyword
    }
}
"""

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
agent = create_tool_calling_agent(llm, retriever_tool, prompt)

########## 5. AgentExecutor 를 정의 ##########
agent_executor = AgentExecutor(agent=agent, tools=retriever_tool, verbose=False)

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


import streamlit as st

# Streamlit 앱 설정
st.set_page_config(page_title="Chatbot", page_icon="🤖")

# 타이틀
st.title("챗봇")

# 사용자 입력을 위한 텍스트 입력 박스
user_input = st.text_input("질문을 입력하세요:")

# 세션 상태를 사용하여 채팅 기록 관리
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 챗봇 응답 함수
def get_bot_response(user_input):
    if user_input:
        # 에이전트 실행 및 응답 가져오기
        response = agent_with_chat_history.invoke({"input": user_input})
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": response["output"]})
        return response["output"]
    return ""

# 챗봇 응답 표시
if st.button("전송"):
    bot_response = get_bot_response(user_input)
    
    # 사용자 질문 및 챗봇 응답 표시
    st.write("### 사용자 질문:")
    st.write(user_input)
    
    st.write("### 챗봇 응답:")
    st.write(bot_response)

# 채팅 기록 표시
if st.session_state.chat_history:
    st.write("### 대화 기록:")
    for message in st.session_state.chat_history:
        role = "사용자" if message["role"] == "user" else "챗봇"
        st.write(f"**{role}:** {message['content']}")
