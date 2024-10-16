import requests
import random
from bs4 import BeautifulSoup
import streamlit as st

# 시작 타이틀
st.title('안녕하세요! 국회 회의록 검색 서비스 챗봇입니다')

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = [] 

if "answer_list" not in st.session_state:
    st.session_state["answer_list"] = ['안녕'] 

def print_messages():
    """대화 내용을 출력하는 함수"""
    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg['content'])

# 사용자 입력 처리
start = st.chat_input('질문이 무엇인가요?')