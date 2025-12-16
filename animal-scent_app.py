import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import math

# --------------------
# 페이지 설정
# --------------------
st.set_page_config(page_title="동물상 관상 향 추천", layout="centered")
st.title("🐾 얼굴 관상 기반 동물상 & 향 추천")
st.caption("※ 본 서비스는 재미를 위한 실험적 AI 분석입니다.")

# --------------------
# MediaPipe FaceMesh
# --------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True)

# --------------------
# 향 테이블
# --------------------
scent_table = {
    "고양이상": ("로지나잇", "도도하고 세련된 장미 머스크 🐱"),
    "여우상": ("메디나", "날카롭고 관능적인 이국적 향 🦊"),
    "강아지상": ("생폴드방스", "밝고 친근한 시트러스 플로럴 🐶"),
    "토끼상": ("판테온", "맑고 사랑스러운 파우더리 향 🐰"),
    "곰상": ("앰버 528", "포근하고 묵직한 우디 앰버 🐻")
}

# --------------------
# 각도 계산
# --------------------
def angle(p1, p2):
    return math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))

# --------------------
# 얼굴 분석
# --------------------
def analyze_face(img):
    h, w, _ = img.shape
    result = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if not result.multi_face_landmarks:
        return None

    lm = result.multi_face_landmarks[0].landmark

    def P(i):
        return np.array([lm[i].x * w, lm[i].y * h])

    # 얼굴 비율
    face_ratio = np.linalg.norm(P(10) - P(152)) / np.linalg.norm(P(234) - P(454))

    # 눈 지표
    left_eye_width = np.linalg.norm(P(33) - P(133))
    left_eye_height = np.linalg.norm(P(159) - P(145))
    eye_ratio = left_eye_height / left_eye_width

    # 눈꼬리 각도
    eye_angle = angle(P(33), P(133))

    # 점수 초기화
    scores = {
        "고양이상": 0,
        "여우상": 0,
        "강아지상": 0,
        "토끼상": 0,
        "곰상": 0
    }

    # 얼굴 비율
    if face_ratio > 1.35:
        scores["여우상"] += 2
    elif face_ratio > 1.25:
        scores["고양이상"] += 2
    elif face_ratio > 1.15:
        scores["강아지상"] += 2
    else:
        scores["곰상"] += 1

    # 눈 모양
    if eye_ratio > 0.33:
        scores["토끼상"] += 2
        scores["강아지상"] += 1
    elif eye_ratio < 0.22:
        scores["여우상"] += 2
        scores["고양이상"] += 1
    else:
        scores["고양이상"] += 1

    # 눈꼬리 각도 (🔥 핵심)
    if eye_angle > 8:
        scores["고양이상"] += 3
    elif eye_angle > 3:
        scores["여우상"] += 2
    else:
        scores["강아지상"] += 1

    return max(scores, key=scores.get)

# --------------------
# UI
# --------------------
img_file = st.file_uploader("📸 얼굴 사진 업로드", type=["jpg", "png", "jpeg"])
cam = st.camera_input("또는 사진 찍기")

image = None
if img_file:
    image = Image.open(img_file)
elif cam:
    image = Image.open(cam)

if image:
    img_np = np.array(image)
    st.image(image, caption="분석 이미지", width=300)

    with st.spinner("AI가 얼굴 특징을 분석 중입니다..."):
        animal = analyze_face(img_np)

    if animal:
        scent, desc = scent_table[animal]
        st.success(f"✨ 결과: {animal}")
        st.markdown(f"### 🌸 추천 향: **{scent}**\n{desc}")
    else:
        st.error("얼굴을 인식하지 못했습니다 😢")
