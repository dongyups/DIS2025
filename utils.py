import streamlit as st
import torch, torchaudio
import uuid
import sounddevice as sd
import wave
import speech_recognition as sr
import urllib.request
import os
import re
import numpy as np


# xtts 인퍼런스
def xttsmodel_inference(tts_input):
    prompt= re.sub("([^\x00-\x7F]|\w)(\.|\。|\?)",r"\1 \2\2", tts_input)
    gpt_cond_latent, speaker_embedding = st.session_state.ttsmodel.get_conditioning_latents(
        gpt_cond_len=30, gpt_cond_chunk_len=4, max_ref_length=60,
        audio_path=[
            st.session_state.pv_inputs + x for x in os.listdir(st.session_state.pv_inputs)
        ]
    )
    out = st.session_state.ttsmodel.inference(
        prompt,
        st.session_state.select_language,
        gpt_cond_latent,
        speaker_embedding,
        repetition_penalty=5.0,
        temperature=0.75,
    )
    return out

# xtts 결과 display 및 저장
def generate_audio(text_chunk):
    # 몇페이지의 음성인지 파악가능
    whatpage, tts_input = text_chunk.split(":")
    aud_dest = st.session_state.pv_outputs + f"voices/{st.session_state.session_id}/"
    os.makedirs(aud_dest, exist_ok=True) # 디렉토리가 없으면 생성
    file_path_name = os.path.join(aud_dest, f"{whatpage}.wav") #최종 파일 경로
    # 인퍼런스
    out = xttsmodel_inference(tts_input)
    # HTML Display
    st.audio(np.expand_dims(np.array(out["wav"]), 0), sample_rate=24000)
    # 자동 저장
    torchaudio.save(file_path_name, torch.tensor(out["wav"]).unsqueeze(0), 24000)


# 음성 녹음 함수 정의
def record_audio(duration, fs, filename):
    # 사용 가능한 채널 수 확인
    device_info = sd.query_devices(kind='input')
    channels = device_info['max_input_channels']  # 사용 가능한 최대 입력 채널 수
    with st.spinner("녹음중입니다..."):
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='int16')
        sd.wait()  # 녹음이 끝날 때까지 대기
    
    # WAV 파일로 저장 (wave 모듈 사용)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16비트 오디오
        wf.setframerate(fs)
        wf.writeframes(recording.tobytes())
    
    st.success("녹음이 완료 되었습니다!")
    return filename


# google voice recognition 사용을 위한 언어 코드 재설정
# https://cloud.google.com/speech-to-text/docs/speech-to-text-supported-languages?hl=ko
def convert_lang_code_for_google_vr(lang_code):
    if lang_code == "ar":
        return "ar-SA"
    elif lang_code == "pt":
        return "pt-PT"
    elif lang_code == "zh-cn":
        return "cmn-Hans-CN"
    elif lang_code == "cs":
        return "cs-CZ"
    elif lang_code == "nl":
        return "nl-NL"
    elif lang_code == "en":
        return "en-US"
    elif lang_code == "fr":
        return "fr-FR"
    elif lang_code == "de":
        return "de-DE"
    elif lang_code == "it":
        return "it-IT"
    elif lang_code == "pl":
        return "pl-PL"
    elif lang_code == "ru":
        return "ru-RU"
    elif lang_code == "es":
        return "es-ES"
    elif lang_code == "tr":
        return "tr-TR"
    elif lang_code == "ja":
        return "ja-JP"
    elif lang_code == "ko":
        return "ko-KR"
    elif lang_code == "hu":
        return "hu-HU"
    elif lang_code == "hi":
        return "hi-IN"
    else:
        raise ValueError("Not supported language detected.")


# 음성 인식 함수 정의
def recognize_speech(audio_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)  # 전체 오디오 파일 읽기
    try:
        text = r.recognize_google(audio, language=convert_lang_code_for_google_vr(st.session_state.select_language))
        return text
    except sr.UnknownValueError:
        st.write("음성을 인식하지 못했어요...")
        return None
    except sr.RequestError as e:
        st.write(f"Could not request results from Google Speech Recognition service; {e}")
        return None


# 이미지 생성 함수
def generate_image(word,client,setting):
    #이미지 url 저장
    image_urls = []

    #프롬프트 설정
    for pt in word:
        prom_word = pt + "in style of" + setting

        #이미지 생성 요청
        response = client.images.generate(
            model = "dall-e-3",
            prompt = prom_word,
            n=1,
            size="1024x1024"
        )

        #생성된 이미지의 url 출력
        image_url = response.data[0].url
        image_urls.append(image_url)

        #세션 ID별 디렉토리 생성
        img_dest = st.session_state.pv_outputs + f"images/{st.session_state.session_id}/" #세션 ID 기반 저장 경로
        os.makedirs(img_dest, exist_ok=True) # 디렉토리가 없으면 생성

        #고유한 파일 이름 생성
        unique_id = str(uuid.uuid4()) #UUID로 고유 이름 생성
        file_path = os.path.join(img_dest, f"{unique_id}.jpg") #최종 파일 경로

        #이미지 다운로드 및 저장
        urllib.request.urlretrieve(image_url, file_path)

    return image_urls


# 이미지 프롬프트 정화
def sanitize_prompt(prompt):
    prohibited_words = ["흥분", "촉수"]  # 금지된 단어 목록

    # 만약 prompt가 리스트일 경우 각 요소에 대해 replace 적용
    if isinstance(prompt, list):
        sanitized_prompt = [sanitize_prompt(item) for item in prompt]  # 각 요소에 대해 재귀적으로 sanitize_prompt 호출
        return sanitized_prompt

    # 만약 prompt가 문자열일 경우 직접 replace 적용
    for word in prohibited_words:
        prompt = prompt.replace(word, "")

    return prompt


# gpt 응답 저장
def save_gpt_response(gpt_response, text_storage):
    for page in gpt_response:
        # 빈 줄 건너뛰기
        if not page.strip():
            continue

        # 페이지 번호와 내용 분리
        if page.startswith("페이지"):
            parts = page.split(":", 1)
            if len(parts) > 1:
                page_num = parts[0]
                content = parts[1].strip()
                text_storage.append({"role": "assistant", "content": f"{page_num}: {content}"})
            else:
                print(f"Warning: Unexpected format in line: {page}")
        else:
            print(f"Warning: Line does not start with '페이지': {page}")


# ChatMessage 객체를 딕셔너리로 변환하는 함수
def chat_message_to_dict(chat_message):
    return {
        "role": chat_message['role'],
        "content": chat_message['content']
    }
