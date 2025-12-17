import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --------------------
# 페이지 설정
# --------------------
st.set_page_config(page_title="동물상 관상 향 추천", layout="centered")

# --------------------
# CSS (한 번만!)
# --------------------
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

# --------------------
# 타이틀
# --------------------
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
animal_colors = {
    "고양이상": "#F4A7B9",  # 파스텔 핑크
    "여우상": "#F6B26B",    # 파스텔 오렌지
    "강아지상": "#A4C2F4",  # 파스텔 블루
    "토끼상": "#B6D7A8",    # 파스텔 그린
    "곰상": "#C9B6A4"       # 파스텔 브라운
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

    scores = {k: 0 for k in scent_table.keys()}

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
        (_, _, _, h1), (_, _, _, h2) = eyes
        eye_size = (h1 + h2) / 2

        if eye_size > h * 0.25:
            scores["토끼상"] += 2
            scores["강아지상"] += 1
        elif eye_size < h * 0.18:
            scores["여우상"] += 1
            scores["고양이상"] += 2
        else:
            scores["고양이상"] += 1
    else:
        scores["곰상"] += 1

    return max(scores, key=scores.get)

# --------------------
# UI 입력
# --------------------
uploaded = st.file_uploader("📸 얼굴 사진 업로드", type=["jpg", "png", "jpeg"])
camera = st.camera_input("또는 사진 찍기")

image = None
if uploaded:
    image = Image.open(uploaded)
elif camera:
    image = Image.open(camera)

# --------------------
# 결과 출력
# --------------------
if image:
    img_np = np.array(image)
    st.image(image, caption="분석 이미지", width=300)

    with st.spinner("얼굴 특징 분석 중..."):
        animal = analyze_face(img_np)

    if animal:
        scent, desc = scent_table[animal]

        border_color = animal_colors[animal]
        st.markdown(f"""
        <div style="background-color:white;
            padding:20px;
            border-radius:18px;
            border: 4px solid {border_color};
            box-shadow:0 4px 12px rgba(0,0,0,0.08);
            margin-top:20px;
            text-align:center;
        ">
    <h2>✨ 당신의 동물상은</h2>
    <h1>{animal}</h1>
    <hr>
    <h3>🌸 추천 향</h3>
    <h2>{scent}</h2>
    <p>{desc}</p>
</div>
""", unsafe_allow_html=True)


            <h2>✨ 당신의 동물상은</h2>
            <h1>{animal}</h1>
            <hr>
            <h3>🌸 추천 향</h3>
            <h2>{scent}</h2>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("얼굴을 인식하지 못했습니다. 정면 사진을 사용해 주세요.")
