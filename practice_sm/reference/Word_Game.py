import requests
import random
from bs4 import BeautifulSoup
import streamlit as st

# 게임 시작 타이틀
st.title('끝말잇기 게임을 시작합니다! :sunglasses:')

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = [] 

if "answer_list" not in st.session_state:
    st.session_state["answer_list"] = ['안녕'] 

def print_messages():
    """대화 내용을 출력하는 함수"""
    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg['content'])

def fetch_word_from_api(start_word):
    """우리말샘 API를 사용해 단어 리스트를 가져오는 함수"""
    URL = f'https://opendict.korean.go.kr/api/search?certkey_no=6833&key=D223927B24E62F3A378084A6DFEEABED&target_type=search&req_type=xml&part=word&q={start_word}&sort=dict&start=1&num=100&advanced=y&target=1&method=start&pos=1&letter_s=2&letter_e=10'
    res = requests.get(URL)
    soup = BeautifulSoup(res.content, 'xml', from_encoding='utf-8')
    return [i.get_text().replace('-', '') for i in soup.select('word')]


# 사용자 입력 처리
start = st.chat_input('단어를 입력해주세요.')

# 게임 종료 명령 처리
if start == '/그만':
    st.error('게임이 종료되었습니다. 다시 시작하려면 단어를 입력해주세요.', icon="🚨")
    st.session_state["answer_list"] = []  # 게임 종료 후 리스트 초기화
elif start == '/다시':
    st.session_state["answer_list"] = []  # 게임 초기화
    st.success('게임이 초기화되었습니다. 다시 단어를 입력해주세요.')
else:
    if start:
        # 중복 단어 체크
        if start in st.session_state["answer_list"]:
            st.chat_message('assistant').write(f'패배, {start}(은)는 이미 사용한 단어입니다.')
            st.session_state["answer_list"] = []  # 게임 리셋
        else:
            # 입력 단어가 사전에 있는지 확인
            word_list = fetch_word_from_api(start)
            if not word_list:
                st.chat_message('assistant').write(f'패배, {start}(은)는 사전에 없는 단어입니다.')
            else:
                # 다음 단어 찾기
                next_word_list = fetch_word_from_api(start[-1])
                if not next_word_list:
                    st.chat_message('assistant').write(f'플레이어의 승리, {start[-1]}(으)로 시작하는 단어가 없습니다.')
                else:
                    answer = random.choice(next_word_list)
                    while answer in st.session_state["answer_list"]:
                        # 중복 단어일 경우 다른 단어를 선택
                        answer = random.choice(next_word_list)

                    # 대화 내용 업데이트
                    st.session_state["messages"].append({"role": "user", "content": start})
                    st.session_state["messages"].append({"role": "assistant", "content": answer})
                    st.session_state["answer_list"].extend([start, answer])
                    print_messages()
    