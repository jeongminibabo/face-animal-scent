import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --------------------
# 페이지 설정
# --------------------
st.set_page_config(page_title="동물상 관상 향 추천", layout="centered")
st.title("🐾 얼굴 관상 기반 동물상 & 향 추천")
st.caption("※ 본 서비스는 재미를 위한 AI 분석입니다.")

# --------------------
# OpenCV 얼굴 검출기 로드
# --------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# --------------------
# 향 추천 테이블
# --------------------
scent_table = {
    "고양이상": {
        "scent": "로지나잇",
        "desc": "세련되고 도도한 향기로운 장미 향 😽"
    },
    "강아지상": {
        "scent": "생폴드방스",
        "desc": "에너지 있고 다채로우며 조화를 이루는 과일 향 🐶"
    },
    "여우상": {
        "scent": "메디나",
        "desc": "성숙하고 이국적이며 감각적인 무드를 가진 향 🦊"
    },
    "토끼상": {
        "scent": "판테온",
        "desc": "맑고 청량하며 쾌활한 느낌의 향 🐰"
    },
    "곰상": {
        "scent": "앰버 528",
        "desc": "딥하게 무게감 있으며 포근한 우디향 🐻 "
    }
}

# --------------------
# 동물상 판별 함수 (얼굴 비율 기반)
# --------------------
def analyze_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    ratio = h / w  # 얼굴 세로/가로 비율

    if ratio > 1.35:
        return "여우상"
    elif ratio > 1.25:
        return "고양이상"
    elif ratio > 1.15:
        return "강아지상"
    elif ratio > 1.05:
        return "토끼상"
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
        st.error("얼굴을 인식하지 못했어요 😢 정면 사진을 사용해 주세요.")
    else:
        st.success(f"✨ 분석 결과: **{animal}**")

        scent = scent_table[animal]
        st.markdown(f"""
        ### 🌸 어울리는 향
        **{scent['scent']}**

        {scent['desc']}
        """)

        st.info("AI 분석 결과는 참고용이며 실제 관상·성격과는 무관합니다.")
