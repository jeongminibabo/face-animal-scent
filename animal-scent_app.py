import streamlit as st
import cv2
import numpy as np
from PIL import Image, ExifTags

def fix_image_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = image._getexif()

        if exif is not None:
            o = exif.get(orientation)

            if o == 3:
                image = image.rotate(180, expand=True)
            elif o == 6:
                image = image.rotate(270, expand=True)
            elif o == 8:
                image = image.rotate(90, expand=True)

    except:
        pass

    return image

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

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

    # 가장 큰 얼굴 선택
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_roi = gray[y:y+h, x:x+w]

    ratio = h / w  # 얼굴 길이 비율
    face_area = w * h
    img_h, img_w = gray.shape

    # --------------------
    # 눈 검출
    # --------------------
    eyes = eye_cascade.detectMultiScale(face_roi, 1.2, 5)

    eye_score = 0
    cat_eye_score = 0

    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])[:2]
        (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = eyes

        # 눈 중심 좌표
        cx1, cy1 = ex1 + ew1/2, ey1 + eh1/2
        cx2, cy2 = ex2 + ew2/2, ey2 + eh2/2

        # 눈 높이 비율
        eye_height_ratio = ((cy1 + cy2) / 2) / h

        # 눈 간 거리
        eye_distance_ratio = abs(cx2 - cx1) / w

        # 눈 면적 비율
        eye_area_ratio = (ew1*eh1 + ew2*eh2) / face_area

        # --------------------
        # 👁️ 눈 각도 (고양이상 핵심)
        # --------------------
        dx = cx2 - cx1
        dy = cy2 - cy1
        angle = np.degrees(np.arctan2(dy, dx))  # 각도 (도)

        # ---- 일반 눈 점수 ----
        if eye_height_ratio < 0.35:
            eye_score += 1
        if eye_distance_ratio > 0.35:
            eye_score += 1
        if eye_area_ratio > 0.05:
            eye_score += 1

        # ---- 고양이 눈 점수 ----
        if angle < -5:   # 오른쪽 눈이 더 위 (눈꼬리 상승)
            cat_eye_score += 2
        elif angle < -2:
            cat_eye_score += 1
        # ---- 눈 기반 점수 ----
        if eye_height_ratio < 0.35:
            eye_score += 1  # 눈이 위 → 고양이/여우
        if eye_distance_ratio > 0.35:
            eye_score += 1  # 눈 간 거리 큼 → 강아지/토끼
        if eye_area_ratio > 0.05:
            eye_score += 1  # 눈 큼 → 토끼/강아지

    # --------------------
    # 점수 테이블
    # --------------------
    scores = {
        "여우상": 0,
        "고양이상": 0,
        "강아지상": 0,
        "토끼상": 0,
        "곰상": 0
    }

    # 얼굴 비율 점수
    if ratio > 1.4:
        scores["여우상"] += 2
    elif ratio > 1.3:
        scores["고양이상"] += 2
    elif ratio > 1.2:
        scores["강아지상"] += 2
    elif ratio > 1.1:
        scores["토끼상"] += 1
        scores["고양이상"] += 1
    
    else:
        scores["곰상"] += 2

    # 눈 점수 반영
    if eye_score >= 3 and ratio > 1.15:
        scores["토끼상"] += 2
    elif eye_score >= 3:
        scores["고양이상"] += 2
    elif eye_score == 2:
        scores["고양이상"] += 1
        scores["강아지상"] += 1
    elif eye_score == 1:
        scores["여우상"] += 1
    else:
        scores["곰상"] += 1

        # 고양이 눈 각도 반영
    if cat_eye_score >= 2:
        scores["고양이상"] += 2
    elif cat_eye_score == 1:
        scores["고양이상"] += 1

    return max(scores, key=scores.get)

# --------------------
# UI
# --------------------

st.subheader("📸 얼굴 사진 입력")

col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader(
        "사진 업로드",
        type=["jpg", "png", "jpeg"]
    )

with col2:
    camera = st.camera_input("사진 찍기")

image = None

if uploaded:
    image = fix_image_orientation(Image.open(uploaded))
elif camera:
    image = Image.open(camera)  # ❌ EXIF 보정 금지

if image:
    image_np = np.array(image)

    st.image(image, caption="분석할 이미지", width=300)

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
