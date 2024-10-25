import os
import re
import json
import torch
import streamlit as st
from typing import TypedDict
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langchain.retrievers import BM25Retriever, EnsembleRetriever, MultiQueryRetriever
from langchain_openai import OpenAIEmbeddings

# 페이지 설정
st.set_page_config(page_title="국회 회의록 AI검색 서비스", layout="wide", page_icon="🤖")

with st.sidebar:
    st.session_state["OPENAI_API"] = st.text_input(label="OPENAI API 키", placeholder="Enter Your API Key", value="", type="password")
    st.session_state["LANGCHAIN_API"] = st.text_input(label="LANGCHAIN API 키", placeholder="Enter Your API Key", value="", type="password")

    st.markdown('---')

# OpenAI API 키가 입력되었는지 확인
if st.session_state["OPENAI_API"] and st.session_state["LANGCHAIN_API"]:
    os.environ['OPENAI_API_KEY'] = st.session_state["OPENAI_API"]
    os.environ['LANGCHAIN_API_KEY'] = st.session_state["LANGCHAIN_API"]

    # JSON 파일을 열고 로드
    with open('merged_data.json', 'r', encoding='utf-8') as file:
        data = json.load(file)


    # JSON 객체를 문자열로 변환
    json_string = str(data)

    # 특정 기준 문자열을 사용하여 분리
    split_docs = json_string.split('}}')

    # 데이터를 저장할 리스트
    organized_data_list = []

    for data_str in split_docs:
        # 각 필드 추출
        organized_data = {}
        
        # 각 필드에 대해 정규 표현식을 적용하고 결과 확인
        filename_match = re.search(r"'filename': '([^']*)'", data_str)
        organized_data['filename'] = filename_match.group(1) if filename_match else None

        date_match = re.search(r"'date': '([^']*)'", data_str)
        organized_data['date'] = date_match.group(1) if date_match else None

        conference_number_match = re.search(r"'conference_number': '([^']*)'", data_str)
        organized_data['conference_number'] = conference_number_match.group(1) if conference_number_match else None

        question_number_match = re.search(r"'question_number': '([^']*)'", data_str)
        organized_data['question_number'] = question_number_match.group(1) if question_number_match else None

        meeting_name_match = re.search(r"'meeting_name': '([^']*)'", data_str)
        organized_data['meeting_name'] = meeting_name_match.group(1) if meeting_name_match else None

        generation_number_match = re.search(r"'generation_number': '([^']*)'", data_str)
        organized_data['generation_number'] = generation_number_match.group(1) if generation_number_match else None

        committee_name_match = re.search(r"'committee_name': '([^']*)'", data_str)
        organized_data['committee_name'] = committee_name_match.group(1) if committee_name_match else None

        meeting_number_match = re.search(r"'meeting_number': '([^']*)'", data_str)
        organized_data['meeting_number'] = meeting_number_match.group(1) if meeting_number_match else None

        session_number_match = re.search(r"'session_number': '([^']*)'", data_str)
        organized_data['session_number'] = session_number_match.group(1) if session_number_match else None

        agenda_match = re.search(r"'agenda': '([^']*)'", data_str)
        organized_data['agenda'] = agenda_match.group(1) if agenda_match else None

        law_match = re.search(r"'law': '([^']*)'", data_str)
        organized_data['law'] = law_match.group(1) if law_match else None

        context_match = re.search(r"'context': '([^']*)'", data_str)
        organized_data['context'] = context_match.group(1) if context_match else None

        context_learn_match = re.search(r"'context_learn': '([^']*)'", data_str)
        organized_data['context_learn'] = context_learn_match.group(1) if context_learn_match else None

        context_summary_match = re.search(r"'context_summary': {'summary_q': '([^']*)', 'summary_a': '([^']*)'}", data_str)
        organized_data['context_summary'] = context_summary_match.groups() if context_summary_match else (None, None)

        questioner_name_match = re.search(r"'questioner_name': '([^']*)'", data_str)
        organized_data['questioner_name'] = questioner_name_match.group(1) if questioner_name_match else None

        questioner_position_match = re.search(r"'questioner_position': '([^']*)'", data_str)
        organized_data['questioner_position'] = questioner_position_match.group(1) if questioner_position_match else None

        question_match = re.search(r"'question': {'tag': '[^']*', 'comment': '([^']*)', 'keyword': '([^']*)'}", data_str)
        organized_data['question'] = question_match.groups() if question_match else (None, None)

        answerer_name_match = re.search(r"'answerer_name': '([^']*)'", data_str)
        organized_data['answerer_name'] = answerer_name_match.group(1) if answerer_name_match else None

        # 새 리스트에 추가
        organized_data_list.append(str(organized_data))

    # Langchain Document 객체로 변환
    documents = [Document(page_content=doc.strip() + '}}') for doc in organized_data_list if doc.strip()]

    # CUDA 사용 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 임베딩 모델 설정 (GPU 사용)
    embedding_model_name = 'jhgan/ko-sroberta-multitask'
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': device}
    )

    # FAISS 인덱스 설정
    index_path = 'json_faiss_index'

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
        vectorstore = FAISS.from_documents(documents, embedding_model)
        vectorstore.save_local(index_path)

    # LLM 정의
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

    # bm25 retriever와 faiss retriever를 초기화합니다.
    bm25_retriever = BM25Retriever.from_texts(
        split_docs,
    )
    bm25_retriever.k =1

    embedding = OpenAIEmbeddings()  # OpenAI 임베딩을 사용합니다.
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    # MultiQueryRetriever 설정
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=faiss_retriever,
        llm=llm,
    )

    # 앙상블 retriever를 초기화합니다.
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, multi_query_retriever],
        weights=[0.7, 0.3],
    )
    # Retriever 생성
    retriever = ensemble_retriever

    # GraphState 클래스 정의
    class GraphState(TypedDict):
        question: str       # 질문
        context: str        # 문서의 검색 결과
        answer: str         # 답변
        relevance: str      # 답변의 문서에 대한 관련성
        recursion_count: int  # 재귀 횟수

    # 도구 함수 정의
    def json_search_tool(input_text):
        docs = retriever.get_relevant_documents(input_text)
        if docs:
            summaries = [doc.page_content for doc in docs]
            return '\n\n'.join(summaries)
        else:
            return "해당하는 정보를 찾을 수 없습니다."

    #도구 설명 
    json_search_tool = Tool(
        name="JSONSearch",
        func=json_search_tool,
        description=(
            "Use this tool to search for information about a specific person or issue from the JSON document. "
            "Provides a summary of fields such as conference number, meeting name, generation number, committee name, agenda, law, Q&A type, context, learning context, context summary, questioner, and question. "
            "If a specific person or issue is searched, first prioritize gathering information for exact matches of the search term, and then proceed to work on approximate matches."
        )
    )

    # 도구 목록 생성
    tools = [json_search_tool]

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

    #시스템 프롬프트

    agent.agent.llm_chain.prompt.messages[0].prompt.template = (
        "You are an AI assistant that helps users find information about individuals or issues from the National Assembly JSON document. "
        "When a user asks about a specific person or issue, you should use the JSONSearch tool to find meeting information. "
        "If the user inputs a question in Korean, translate it into English before proceeding with the search. "
        "For queries related to specific policies or legislative issues, summarize the discussions in chronological order and by issue. "
        "Provide a summarized response including fields such as conference number, meeting name, generation number, committee name, agenda, law, Q&A type, context, learning context, context summary, questioner, and question. "
        "Do not mention the use of tools; just provide the necessary information."
        "Be sure to answer in Korean. " 
    )
    # 문서 검색을 위한 함수 정의
    def retrieve_document(state: GraphState) -> GraphState:
        # 재귀 횟수 증가 또는 초기화
        recursion_count = state.get('recursion_count', 0) + 1
        # 사용자의 질문에 따라 문서를 검색합니다.
        retrieved_docs = retriever.get_relevant_documents(state['question'])
        context = '\n\n'.join([doc.page_content for doc in retrieved_docs])
        # 기존의 'context'를 업데이트합니다.
        state.update({
            'context': context,
            'recursion_count': recursion_count
        })
        return state

    # 에이전트의 답변을 생성하는 함수 정의
    def llm_answer(state: GraphState) -> GraphState:
        # 에이전트를 사용하여 답변을 생성합니다.
        answer = agent.run(input=state['question'])
        
        # 출력 글자 수 제한
        max_length = 20000
        if len(answer) > max_length:
            answer = answer[:max_length] + '...'  # 자르고 '...' 추가
        
        state.update({'answer': answer})
        return state

    # 그래프 워크플로우 정의
    workflow = StateGraph(GraphState)

    # 노드 추가
    workflow.add_node("retrieve", retrieve_document)
    workflow.add_node("llm_answer", llm_answer)

    # 엣지 연결
    workflow.add_edge("retrieve", "llm_answer")

    # 시작점 설정
    workflow.set_entry_point("retrieve")

    # 메모리 저장소 설정
    memory_saver = MemorySaver()

    # 워크플로우 컴파일
    app = workflow.compile(checkpointer=memory_saver)

    # 에이전트와 대화하는 함수 수정
    def chat_with_agent(user_input):
        # 초기 상태 설정
        state = GraphState(
            question=user_input,
            context='',
            answer='',
            recursion_count=0
        )
        config = RunnableConfig(recursion_limit=20, configurable={"thread_id": "SELF-RAG"})
        output = app.invoke(state, config=config)
        recursion_count = output.get('recursion_count', 0)
        
        # 최종 답변에 재귀 횟수만 포함
        final_answer = output.get('answer', '')
        # final_answer += f"\n\n총 재귀 횟수: {recursion_count}"
        
        return final_answer

    # 대화 내용을 출력하는 함수
    def print_messages():
        """대화 내용을 출력하는 함수"""
        for msg in st.session_state["messages"]:
            st.chat_message(msg['role']).write(msg['content'])


    # Streamlit 메인 코드
    st.title("안녕하세요! 국회 회의록 검색AI입니다!")  # 시작 타이틀


    # 세션 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = [] 

    if "session_history" not in st.session_state:
        st.session_state["session_history"] = {}  # 세션 기록을 위한 딕셔너리
        
    # 세션 ID 리스트 초기화
    if "session_ids" not in st.session_state:
        st.session_state["session_ids"] = []  # 세션 ID 리스트 초기화

    with st.sidebar:
    # 세션 ID 입력
        session_id = st.text_input("기록할 ID 입력", placeholder="ID를 입력하세요")

    # 세션 저장 버튼
        if st.button("대화기록 저장"):
            # 세션 ID가 입력되었는지 확인
            if session_id:
                if session_id not in st.session_state.session_ids:  # 중복 세션 ID 확인
                    st.session_state.session_history[session_id] = st.session_state.messages
                    st.session_state["messages"] = []
                    st.session_state.session_ids.append(session_id)  # 세션 ID 추가
                    st.success(f"세션 '{session_id}'이 저장되었습니다.")
                else:
                    st.warning("이미 존재하는 ID입니다.")
            else:
                st.warning("ID를 입력하세요.")

        # 불러올 세션 선택
        if st.session_state.session_ids:  # 세션 ID가 있는 경우에만 드롭다운 표시
            selected_session_id = st.selectbox("불러올 ID 선택", options=st.session_state.session_ids)
            # 세션 불러오기 버튼
            if st.button("기록 불러오기"):
                # if selected_session_id in st.session_state.session_history.keys():
                st.session_state.messages = st.session_state.session_history[selected_session_id]
                st.success(f"대화기록을 불러왔습니다.")  
                    
        else:
            st.info("저장된 대화기록이 없습니다.")

        # 세션 초기화 및 임시저장
        if st.button("새로운 대화 생성"):
            st.session_state.session_ids.append("이전 대화기록")
            st.session_state.session_history["이전 대화기록"] = st.session_state.messages
            st.session_state["messages"] = []

        
        if session_id :
                session_id = selected_session_id
        else :
            session_id = "default_session"


    # 이전 대화 기록을 에이전트에 전달
    if st.session_state.messages:
        previous_messages = [{"role": msg['role'], "content": msg['content']} for msg in st.session_state.messages]


    # 사용자 입력 처리
    user_input = st.chat_input('질문이 무엇인가요?')

    if user_input:
        # 세션 ID를 설정
        if st.session_state.messages:
            response = chat_with_agent(user_input + "\n\nPrevious Messages: " + str(previous_messages))
        else:
            response = chat_with_agent(user_input)

        # 메시지를 세션에 추가
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.session_state["messages"].append({"role": "assistant", "content": response})

    # 대화 내용 출력
    print_messages()

else:
    st.warning("API 키를 입력하세요")
