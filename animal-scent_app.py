pip install streamlit opencv-python mediapipe numpy
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# --------------------
# 기본 설정
# --------------------
st.set_page_config(page_title="동물상 관상 향 추천", layout="centered")
st.title("🐾 얼굴 관상 기반 동물상 & 향 추천")
st.caption("※ 본 서비스는 재미를 위한 AI 분석입니다.")

# --------------------
# MediaPipe 얼굴 메쉬
# --------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True)

# --------------------
# 향 추천 테이블
# --------------------
scent_table = {
    "고양이상": {
        "scent": "플로럴 & 머스크",
        "desc": "세련되고 도도한 분위기를 강조해줘요."
    },
    "강아지상": {
        "scent": "시트러스 & 프루티",
        "desc": "밝고 친근한 인상과 잘 어울려요."
    },
    "여우상": {
        "scent": "우디 & 스파이시",
        "desc": "성숙하고 신비로운 이미지를 살려줘요."
    },
    "곰상": {
        "scent": "파우더리 & 바닐라",
        "desc": "포근하고 안정적인 느낌을 줘요."
    }
}

# --------------------
# 얼굴 분석 함수
# --------------------
def analyze_face(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark

    # 주요 포인트 (눈, 턱)
    left_eye = np.array([landmarks[33].x, landmarks[33].y])
    right_eye = np.array([landmarks[263].x, landmarks[263].y])
    chin = np.array([landmarks[152].x, landmarks[152].y])

    eye_distance = np.linalg.norm(left_eye - right_eye)
    face_height = np.linalg.norm(chin - (left_eye + right_eye) / 2)

    ratio = face_height / eye_distance

    # --------------------
    # 규칙 기반 동물상 판별
    # --------------------
    if ratio > 2.1:
        return "여우상"
    elif ratio > 1.8:
        return "고양이상"
    elif ratio > 1.6:
        return "강아지상"
    else:
        return "곰상"

# --------------------
# UI
# --------------------
uploaded = st.file_uploader("📸 얼굴 사진을 업로드하세요", type=["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    image_np = np.array(image)

    st.image(image, caption="업로드한 이미지", width=300)

    with st.spinner("관상을 분석 중입니다..."):
        animal = analyze_face(image_np)

    if animal is None:
        st.error("얼굴을 인식하지 못했어요 😢")
    else:
        st.success(f"✨ 분석 결과: **{animal}**")

        scent = scent_table[animal]
        st.markdown(f"""
        ### 🌸 어울리는 향
        **{scent['scent']}**

        {scent['desc']}
        """)

        st.info("AI 분석 결과는 참고용이며 실제 성격·운명과는 무관합니다.")
streamlit run animal_scent_app.py      
