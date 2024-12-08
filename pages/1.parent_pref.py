import streamlit as st
# 페이지 설정은 반드시 다른 Streamlit 명령어보다 먼저 실행되어야 함
st.set_page_config(layout="wide")
import warnings, os, re, uuid, openai, json
from dotenv import load_dotenv
warnings.simplefilter(action='ignore')


# api키 설정
load_dotenv(verbose=True)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit 앱 제목
st.title("다문화가정 아동 부모 상담 챗봇")

# 컨테이너를 사용하여 채팅 영역과 입력 영역을 분리
chat_container = st.container()
input_container = st.container()

# 글로벌 변수
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "parent_prefer" not in st.session_state:
    st.session_state.parent_prefer = None

# 세션 상태 변수 초기화 # 부모의 출신국가 언어
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "system", "content": 
        '''
        너는 다문화가정 아동들의 부모와 대화하는 챗봇이야. 순서대로 5가지 질문을 하나씩 해줘. 부모가 답변하면 그 다음에 질문을 해.

        1. 부모의 출신국가 
        2. 아동 나이 
        3. 자녀가 한국어와 다른 언어 중 어느 것을 더 어려워하는지 
        4. 부모가 자녀가 배웠으면 하는 언어 표현이 있는지 
        5. 부모가 자녀에게 알려주고 싶은 문화, 풍습, 단어, 설화 등의 문화적 요소가 있는지
         
        2번 질문부터는 부모의 출신국가 언어로도 번역해줘. 출신국가가 캐나다인 경우에는 영어로.
        아동의 나이가 0~3세이면 영아로 정리해줘. 
        아동의 나이가 4세~7세 이상이면 유아로 정리해줘. 
        5번의 답변을 듣고 답변에 대해 너가 이해한대로 설명해줘. 만일 사용자가 아니라고 하면, 다시 이해하고 맞는지 질문해.
        5번의 답변이 끝나면 다음과 같은 형식으로 예시처럼 정리해줘.

        형식: {(1)의 답변, (2)의 답변, (3)에서 답변하지 않은 언어, (3)의 답변, (4)의 답변, (5)의 답변} 
        예시: {캐나다, 유아, 한국어, 영어, 날씨에 대한 표현, 아이스 하키}
        '''
    })
    st.session_state.messages.append({
        "role":"assistant","content": "출신국가가 어디십니까?"
    })

# 채팅 영역에 메시지 표시
with chat_container:
    # 스크롤 가능한 영역 생성
    with st.container():
        for message in st.session_state.messages:
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
            width: calc(100% -15rem);
        }
        .stSpinner {
            position: fixed;
            bottom: 7rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 입력창
    user_input = st.text_input("메시지를 입력하세요:", key="user_input")

    if user_input:
        # 사용자 메시지 표시
        with chat_container:
            with st.chat_message("user"):
                st.write(user_input)

        # 메시지 저장 및 응답 생성
        st.session_state.messages.append({"role": "user", "content": user_input})

        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("생각 중..."):
                    llm = openai.chat.completions.create(
                        model="gpt-4",
                        messages=st.session_state.messages
                    )
                    gpt_response = "\n".join(llm.choices[0].message.content.strip().split('\n'))
                    # 응답 저장
                    st.session_state.messages.append({"role": "assistant", "content": gpt_response})
                    st.write(gpt_response)

        # 질문 끝나면 선호도에 정리 부분 저장
        response = gpt_response.strip()
        if '{' and '}' in response:
            match = re.search(r'\{(.*?)\}', response)
            st.session_state.parent_prefer = match.group(1)
            print("부모 선호도 결과:", st.session_state.parent_prefer)


# 모든 질문이 끝난 경우 결과 출력
if st.session_state.parent_prefer is not None:
    # 부모 선호도 json 저장
    filename_p = st.session_state.session_id + "_parent_prefer.json"
    with open(st.session_state.pv_outputs + filename_p, 'w', encoding='utf-8') as f:
        json.dump(st.session_state.parent_prefer, f, ensure_ascii=False, indent=4)
    # 다음 페이지로
    st.write("모든 질문이 완료되었습니다. 결과를 정리합니다. 다음 버튼을 눌러주세요.")
    st.page_link("pages/2.child_pref.py", label="아동 선호도", icon="2️⃣")