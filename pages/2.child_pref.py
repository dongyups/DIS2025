import streamlit as st
# 페이지 설정은 반드시 다른 Streamlit 명령어보다 먼저 실행되어야 함
st.set_page_config(layout="wide")
import sounddevice as sd
import os, re, openai, json
from dotenv import load_dotenv
import utils

# Streamlit 앱 제목
st.title("안녕 친구 🎈")

# 컨테이너를 사용하여 채팅 영역과 입력 영역을 분리
chat_container = st.container()
input_container = st.container()

# OpenAI API 키 설정
load_dotenv(verbose=True)
openai.api_key = os.getenv("OPENAI_API_KEY")

# 사용 가능한 채널 수 확인
device_info = sd.query_devices(kind='input')
channels = device_info['max_input_channels']  # 사용 가능한 최대 입력 채널 수

# 글로벌 변수
if "child_prefer" not in st.session_state:
    st.session_state.child_prefer = None
if "show_text" not in st.session_state:
    st.session_state.show_text = False
if "child_input" not in st.session_state:
    st.session_state.child_input = None

### 다음 요소를 기억해: {st.session_state.parent_prefer}
# 세션 상태 변수 초기화
if 'child_messages' not in st.session_state:
    st.session_state.child_messages = []
    st.session_state.child_messages.append({"role" : "system", "content" : 
        '''
        너는 아동과 대화하며 친구가 되어주기 위한 챗봇이야. 

        1. 아동의 선호 요소(예: 좋아하는 음식, 동물, 캐릭터 등등)를 파악해봐. 
        2. 이때 아동의 답변에서 핵심 키워드를 찾아줘.
        3. 총 다섯 가지 질문을 하되, 여러 질문을 한 번에 하지 마. 
        4. 아동이 답변하면 그 다음에 질문을 해. 마지막 답변을 듣고 나서 ‘너가 좋아하는 것들로 동화를 만들어 줄게’ 라는 식으로 말하되, 질문형으로 끝내지 마. 
        답변이 모두 끝나면 다음과 같은 형식으로 예시처럼 정리해줘.

        형식: {답변1의 키워드, 답변2의 키워드, 답변3의 키워드, 답변4의 키워드, 답변5의 키워드}
        예시: {포도, 토끼, 하늘색, 숨바꼭질, 루피}
        '''
    })
    st.session_state.child_messages.append({
        "role":"assistant","content":"안녕! 난 너의 다정한 친구야 나랑 같이 동화를 만들어보자!"
    })

# 채팅 영역에 메시지 표시
with chat_container:
    # 스크롤 가능한 영역 생성
    with st.container():
        for message in st.session_state.child_messages:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.write(message["content"])

# 입력 영역을 화면 하단에 고정
with input_container:
    # CSS로 입력창을 하단에 고정
    st.markdown(
        """
        <style>
        .stTextInput {
            position: fixed;
            bottom: 3rem;
            width: calc(100% - 15rem);
        }
        .stSpinner {
            position: fixed;
            bottom: 7rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    button_child = st.button("녹음 시작!")

    # 음성 녹음
    if button_child:
        audio_file = utils.record_audio(duration=5, fs=44100, filename=st.session_state.pv_outputs+"tmp_child_voice.wav")
        if audio_file:
            user_audio = utils.recognize_speech(audio_file)
            if user_audio is not None:
                st.session_state.child_input = user_audio
                print(f"음성 입력 받음: {st.session_state.child_input}")
            else:
                st.session_state.show_text = True
        else:
            st.write("음성을 녹음하지 못했어요...")
            st.session_state.show_text = True

    # 텍스트 입력 처리
    if st.session_state.show_text:
        text_input = st.text_input("음성 인식에 실패했습니다. 텍스트로 입력해주세요:", key="text_input")
        if text_input:  # 텍스트가 입력되었을 때만
            st.session_state.child_input = text_input
            print(f"텍스트 입력 받음: {st.session_state.child_input}")
            st.session_state.show_text = False
            st.rerun()  # 화면 갱신

    if st.session_state.child_input is not None:
        # 메시지 저장 및 응답 생성
        st.session_state.child_messages.append({"role": "user", "content": st.session_state.child_input})
        with chat_container:
            with st.chat_message("user"):
                st.write(st.session_state.child_input)
        st.session_state.child_input=None

        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("생각 중..."):
                    llm = openai.chat.completions.create(
                        model="gpt-4",
                        messages=st.session_state.child_messages
                    )
                    gpt_response = "\n".join(llm.choices[0].message.content.strip().split('\n'))
                    # 응답 저장
                    st.session_state.child_messages.append({"role": "assistant", "content": gpt_response})
                    st.write(gpt_response)

        # 질문 끝나면 선호도에 정리 부분 저장
        response = gpt_response.strip()
        if '{' and '}' in response:
            match = re.search(r'\{(.*?)\}', response)
            st.session_state.child_prefer = match.group(1)
            print("아동 선호도 결과:", st.session_state.child_prefer)


# 모든 질문이 끝난 경우 결과 출력
if st.session_state.child_prefer is not None:
    # 아동 선호도 json 저장
    filename_c = st.session_state.session_id + "_child_prefer.json"
    with open(st.session_state.pv_outputs + filename_c, 'w', encoding='utf-8') as f:
        json.dump(st.session_state.child_prefer, f, ensure_ascii=False, indent=4)
    # 다음 페이지로
    st.write("모든 질문이 완료되었습니다. 결과를 정리합니다. 다음 버튼을 눌러주세요.")
    st.page_link("pages/3.generate_fairytale.py", label="완성!", icon="3️⃣")
