# reference: https://huggingface.co/spaces/coqui/xtts/blob/main/app.py#L32
import streamlit as st
import warnings, os, random
from pydub import AudioSegment
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch, torchaudio
import numpy as np
import utils
warnings.simplefilter(action='ignore')

# 하이퍼파라미터 세팅
use_gpu = False # True: 0번 single-GPU 사용 & False: CPU 사용
os.environ["CUDA_VISIBLE_DEVICES"] = "0" if use_gpu else "-1"
seed_num = 777 # 수정가능, 결과값 고정을 위함
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed_num)
random.seed(seed_num)


# TTS 모델 캐싱
@st.cache_resource(show_spinner=True)
def Caching_XTTS_Model():
    # 디폴트 Windows parent folder -> C:/Users/USERNAME/AppData/Local/tts
    config = XttsConfig()
    config.load_json(
        "./xtts/config.json"
    )
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config, 
        checkpoint_dir="./xtts/", 
    )
    if use_gpu:
        model.cuda()
    return model
st.session_state.ttsmodel = Caching_XTTS_Model()


# 제목
st.title("TTS를 위한 음성녹음")


# Part1: 개별 분류 코드 작성
supported_languages = [
    "  : -- Select Your Language -- :  ",
    "Arabic(아랍어) : ar", 
    "Brazilian Portuguese(포르투갈어) : pt", 
    "Mandarin Chinese(중국어) : zh-cn", 
    "Czech(체코어) : cs", 
    "Dutch(네덜란드어) : nl", 
    "English(영어) : en", 
    "French(프랑스어) : fr", 
    "German(독일어) : de", 
    "Italian(이탈리아어) : it", 
    "Polish(폴란드어) : pl", 
    "Russian(러시아어) : ru", 
    "Spanish(스페인어) : es", 
    "Turkish(터키어) : tr", 
    "Japanese(일본어) : ja", 
    "Korean(한국어) : ko", 
    "Hungarian(헝가리어) : hu", 
    "Hindi(힌디어) : hi"
]
chosen_lang = st.selectbox("아이의 제2언어(Second Language) 를 선택해주세요.", supported_languages)
st.session_state.select_language = chosen_lang.split(" : ")[-1] # ex) ko
st.session_state.select_lang_name = chosen_lang.split("(")[-1].split(")")[0] # ex) 한국어

name = st.text_input("자녀분의 성함을 영어로 작성해주세요. EX) 홍길동 -> gildong hong", value="")
button0 = st.button("Confirm")
if button0:
    if st.session_state.select_language == ' ':
        st.warning("언어가 선택되지 않았습니다.", icon="⚠️")
    if name == '':
        st.warning("성함이 입력되지 않았습니다.", icon="⚠️")
    else:
        st.success(f"사용될 언어 코드와 성함: {st.session_state.select_language} & {name}")
name = "_".join(name.lower().split())

st.markdown(f"선택한 '제2언어_성함' 으로 경로를 생성합니다. EX) ko_gildong_hong")
button1 = st.button("Submit")
st.session_state.pv_inputs = os.getcwd().replace("\\","/")+f"/candidates/{st.session_state.select_language}_{name}/inputs/"
st.session_state.pv_outputs = os.getcwd().replace("\\","/")+f"/candidates/{st.session_state.select_language}_{name}/outputs/"
if button1:
    # 경로설정
    os.makedirs(st.session_state.pv_inputs, exist_ok=True)  # 개별 음성파일 저장 경로
    os.makedirs(st.session_state.pv_outputs, exist_ok=True) # TTS 결과파일 저장 경로
    st.success('경로 생성됨')


# Part2: 개별 목소리 업로드/변환 및 로컬 저장
with st.form("upload-then-clear-form", clear_on_submit=True):
        file_list  = st.file_uploader(
            '음성파일을 업로드 하세요. 여러 파일을 한번에 업로드 하셔도 됩니다.', 
            type=['m4a','wav'], accept_multiple_files=True
        )
        button2 = st.form_submit_button("Convert")
        if button2:

            # 업로드 된 파일 로컬에 저장
            for file in file_list:
                with open(st.session_state.pv_inputs + file.name.lower(), 'wb') as f:
                    f.write(file.getbuffer())

            # 확장자 변환 및 trim
            for file in os.listdir(st.session_state.pv_inputs):
                # m4a 파일의 경우
                if len(file.split(".m4a")[0]) != len(file):
                    tobesaved = st.session_state.pv_inputs + file.split(".m4a")[0]+".wav"
                    audio = AudioSegment.from_file(st.session_state.pv_inputs + file, format="m4a")
                    audio.export(tobesaved, format="wav")
                    os.remove(st.session_state.pv_inputs + file) # m4a 파일 제거
                    audio = AudioSegment.from_wav(tobesaved)
                    audio = audio[:-200] # 윈도우 녹음기 사용시 마지막 노이즈 제거
                    audio.export(tobesaved, format="wav") # 덮어쓰기

                # wav 파일의 경우
                else:
                    tobesaved = st.session_state.pv_inputs + file
                    audio = AudioSegment.from_wav(tobesaved)
                    audio = audio[:-200] # 윈도우 녹음기 사용시 마지막 노이즈 제거
                    audio.export(tobesaved, format="wav") # 덮어쓰기

            del file_list
            st.success('변환 완료')


# Part3: 모델 인퍼런스
output_name = "tmp_parent_voice"
tts_input = st.text_area("TTS로 변환할 샘플 텍스트를 입력하세요.", height=1)
button3 = st.button("Run")

if button3:
    # st.write(prompt)
    with st.spinner("변환 중..."):
        # 확인
        st.write("레퍼런스 파일: " + ", ".join(os.listdir(st.session_state.pv_inputs)))
        out = utils.xttsmodel_inference(tts_input)
        # HTML Display
        st.audio(np.expand_dims(np.array(out["wav"]), 0), sample_rate=24000)
        # 자동 저장
        torchaudio.save(st.session_state.pv_outputs+f"{output_name}.wav", 
                        torch.tensor(out["wav"]).unsqueeze(0), 24000)

        st.success('TTS 생성 및 저장 완료')

        st.markdown("아래의 버튼을 눌러 계속 진행해 주세요.")
        st.page_link("pages/1.parent_pref.py", label="부모 선호도 조사", icon="1️⃣")
