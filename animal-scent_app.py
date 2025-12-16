import streamlit as st
import cv2
import numpy as np
from PIL import Image


# --------------------
# 페이지 설정
# --------------------
st.set_page_config(page_title="동물상 관상 향 추천", layout="centered")

st.markdown("""
<h1>🐾 ANIMAL SCENT FINDER</h1>
<h3>얼굴 인상으로 알아보는 나만의 향</h3>
""", unsafe_allow_html=True)

st.caption("※ 본 서비스는 재미를 위한 단순 특징 기반 분석입니다.")

# --------------------
# OpenCV 분류기
# --------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# --------------------
# 향 추천 테이블
# --------------------
scent_table = {
    "고양이상": ("로지나잇", "도도하고 세련된 장미 머스크 🐱"),
    "여우상": ("메디나", "날카롭고 관능적인 이국적 향 🦊"),
    "강아지상": ("생폴드방스", "밝고 친근한 시트러스 플로럴 🐶"),
    "토끼상": ("판테온", "맑고 사랑스러운 파우더리 향 🐰"),
    "곰상": ("앰버 528", "포근하고 묵직한 우디 앰버 🐻")
}

# --------------------
# 얼굴 분석 함수
# --------------------
def analyze_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face_ratio = h / w

    face_roi = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 5)

    scores = {
        "고양이상": 0,
        "여우상": 0,
        "강아지상": 0,
        "토끼상": 0,
        "곰상": 0
    }

    # 얼굴 비율
    if face_ratio > 1.35:
        scores["여우상"] += 1
    elif face_ratio > 1.25:
        scores["고양이상"] += 1
    elif face_ratio > 1.15:
        scores["강아지상"] += 1
    else:
        scores["곰상"] += 1

    # 눈 분석
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])[:2]
        (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes

        eye_size = (h1 + h2) / 2
        eye_gap = abs(x2 - x1)

        # 눈 크기
        if eye_size > h * 0.25:
            scores["토끼상"] += 2
            scores["강아지상"] += 1
        elif eye_size < h * 0.18:
            scores["여우상"] += 1
            scores["고양이상"] += 2
        else:
            scores["고양이상"] += 1

        # 눈 사이 거리
        if eye_gap > w * 0.45:
            scores["강아지상"] += 2
        elif eye_gap < w * 0.30:
            scores["여우상"] += 2
        else:
            scores["고양이상"] += 1

    else:
        scores["곰상"] += 1

    return max(scores, key=scores.get)

# --------------------
# UI
# --------------------
uploaded = st.file_uploader("📸 얼굴 사진 업로드", type=["jpg", "png", "jpeg"])
camera = st.camera_input("또는 사진 찍기")

image = None
if uploaded:
    image = Image.open(uploaded)
elif camera:
    image = Image.open(camera)

if image:
    img_np = np.array(image)
    st.image(image, caption="분석 이미지", width=300)

    with st.spinner("얼굴 특징 분석 중..."):
        animal = analyze_face(img_np)

    if animal:
        scent, desc = scent_table[animal]
        st.success(f"✨ 분석 결과: {animal}")
        st.markdown(f"### 🌸 추천 향\n**{scent}**\n\n{desc}")
        st.info("본 결과는 단순 특징 기반 추정으로 실제 인상과 다를 수 있습니다.")
        st.markdown("""
<style>
    body {
        background-color: #FFF6F0;
    }
    .stApp {
        background-color: #FFF6F0;
    }
    h1, h2, h3 {
        font-family: 'Pretendard', sans-serif;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    body {
        background-color: #faf7f2;
    }
    .main {
        padding-top: 20px;
    }
    h1 {
        font-family: 'Pretendard', sans-serif;
        text-align: center;
    }
    h3 {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

    elif:
        st.error("얼굴을 인식하지 못했습니다. 정면 사진을 사용해 주세요.")
st.markdown(f"""
<div style="
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    margin-top: 20px;
    text-align: center;
">
    <h2>✨ 당신의 동물상은</h2>
    <h1>{animal}</h1>
    <hr style="margin:15px 0;">
    <h3>🌸 추천 향</h3>
    <h2>{scent}</h2>
    <p style="font-size:16px;">{desc}</p>
</div>
""", unsafe_allow_html=True)
