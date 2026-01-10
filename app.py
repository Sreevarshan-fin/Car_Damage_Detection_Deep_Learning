import streamlit as st
import cv2
import os
import tempfile
from model_helper import predict

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Vehicle Damage Detection",
    page_icon="🚗",
    layout="centered"
)

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>🚗 Vehicle Damage Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Upload a vehicle image or video</p>", unsafe_allow_html=True)
st.divider()

# ---------------- LOAD CAR VALIDATOR ----------------
CASCADE_PATH = "haarcascade_car.xml"
CAR_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)

if CAR_CASCADE.empty():
    st.error("❌ Car validation model not found.")
    st.stop()

def is_car_image(image_path):
    """Returns True if a car is detected, else False"""
    img = cv2.imread(image_path)
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cars = CAR_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(120, 120)
    )

    return len(cars) > 0

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    - **Model:** ResNet-50 (Image Classification)
    - **Validation:** Car presence check
    - **Task:** Damage classification only
    """)
    st.success("🟢 System Ready")

# ---------------- INPUT TYPE ----------------
upload_type = st.radio("Select input type", ["📷 Image", "🎥 Video"])

# =====================================================
# IMAGE
# =====================================================
if upload_type == "📷 Image":
    uploaded_image = st.file_uploader("Upload vehicle image", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        image_path = "temp_image.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        # ✅ VALIDATION ONLY
        if not is_car_image(image_path):
            st.error("❌ Invalid image. Please upload a vehicle image.")
            os.remove(image_path)
        else:
            with st.spinner("🔍 Analyzing damage..."):
                prediction = predict(image_path)

            st.success(f"✅ **Predicted Damage Class:** {prediction}")
            os.remove(image_path)

# =====================================================
# VIDEO
# =====================================================
if upload_type == "🎥 Video":
    uploaded_video = st.file_uploader("Upload vehicle video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        st.video(uploaded_video)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.getbuffer())
            video_path = tmp.name

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2))
        ret, frame = cap.read()
        cap.release()

        if ret:
            frame_path = "temp_frame.jpg"
            cv2.imwrite(frame_path, frame)

            if not is_car_image(frame_path):
                st.error("❌ No vehicle detected in video.")
            else:
                with st.spinner("🔍 Analyzing damage..."):
                    prediction = predict(frame_path)

                st.success(f"✅ **Predicted Damage Class:** {prediction}")

            os.remove(frame_path)

        os.remove(video_path)

# ---------------- FOOTER ----------------
st.divider()
st.markdown(
    "<p style='text-align:center;color:gray;'>Deep Learning • Computer Vision • Streamlit</p>",
    unsafe_allow_html=True
)









