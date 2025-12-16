import streamlit as st
import cv2
import numpy as np
from PIL import Image, ExifTags

# --------------------
# 페이지 설정
# --------------------
st.set_page_config(page_title="동물상 관상 향 추천", layout="centered")
st.title("🐾 얼굴 관상 기반 동물상 & 향 추천")
st.caption("※ 본 서비스는 재미를 위한 AI 분석입니다.")

# --------------------
# 얼굴 / 눈 검출기
# --------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# --------------------
# EXIF 회전 보정 (업로드용)
# --------------------
def fix_image_orientation(image):
    try:
        for k, v in ExifTags.TAGS.items():
            if v == "Orientation":
                orientation_key = k
                break

        exif = image._getexif()
        if exif is not None:
            o = exif.get(orientation_key)
            if o == 3:
                image = image.rotate(180, expand=True)
            elif o == 6:
                image = image.rotate(270, expand=True)
            elif o == 8:
                image = image.rotate(90, expand=True)
    except:
        pass
    return image

# --------------------
# 동물상 판별 함수
# --------------------
def analyze_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    # 가장 큰 얼굴
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_roi = gray[y:y+h, x:x+w]

    face_ratio = h / w
    face_area = w * h

    # --------------------
    # 눈 검출
    # --------------------
    eyes = eye_cascade.detectMultiScale(face_roi, 1.2, 5)
    if len(eyes) < 2:
        return "곰상"

    eyes = sorted(eyes, key=lambda e: e[0])[:2]
    (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = eyes

    # 눈 중심
    cx1, cy1 = ex1 + ew1 / 2, ey1 + eh1 / 2
    cx2, cy2 = ex2 + ew2 / 2, ey2 + eh2 / 2
    eye_center_y = (cy1 + cy2) / 2

    # 특징 수치
    eye_height_ratio = eye_center_y / h
    eye_distance_ratio = abs(cx2 - cx1) / w
    eye_area_ratio = (ew1 * eh1 + ew2 * eh2) / face_area

    # 눈 각도
    dx = cx2 - cx1
    dy = cy2 - cy1
    angle = abs(np.degrees(np.arctan2(dy, dx)))

    # --------------------
    # 최종 분기 (Rule-based)
    # --------------------

    # 🦊 여우상
    if (
        face_ratio >= 1.4 and
        eye_height_ratio < 0.33 and
        angle < 8
    ):
        return "여우상"

    # 🐱 고양이상
    if (
        angle >= 6 and
        eye_height_ratio < 0.38
    ):
        return "고양이상"

    # 🐶 강아지상
    if (
        eye_distance_ratio >= 0.38 or
        eye_height_ratio > 0.38
    ):
        return "강아지상"

    # 🐰 토끼상
    if (
        eye_area_ratio > 0.05 and
        face_ratio > 1.15
    ):
        return "토끼상"

    # 🐻 곰상
    return "곰상"

# --------------------
# 향 추천 테이블
# --------------------
scent_table = {
    "고양이상": ("로지나잇", "세련되고 도도한 장미 향 😽"),
    "강아지상": ("생폴드방스", "밝고 친근한 과일 향 🐶"),
    "여우상": ("메디나", "성숙하고 이국적인 무드 🦊"),
    "토끼상": ("판테온", "맑고 청량한 플로럴 🐰"),
    "곰상": ("앰버 528", "포근하고 묵직한 우디 향 🐻")
}

# --------------------
# UI
# --------------------
st.subheader("📸 얼굴 사진 입력")

col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader("사진 업로드", type=["jpg", "jpeg", "png"])

with col2:
    camera = st.camera_input("사진 찍기")

image = None

if uploaded:
    image = fix_image_orientation(Image.open(uploaded))
elif camera:
    image = Image.open(camera)  # camera_input은 보정 ❌

if image:
    st.image(image, caption="분석할 이미지", width=300)
    image_np = np.array(image)

    with st.spinner("관상을 분석 중입니다..."):
        animal = analyze_face(image_np)

    if animal is None:
        st.error("얼굴을 인식하지 못했어요 😢")
    else:
        scent, desc = scent_table[animal]
        st.success(f"✨ 분석 결과: **{animal}**")
        st.markdown(f"""
        ### 🌸 어울리는 향
        **{scent}**

        {desc}
        """)
        st.info("AI 분석 결과는 재미를 위한 참고용입니다.")
