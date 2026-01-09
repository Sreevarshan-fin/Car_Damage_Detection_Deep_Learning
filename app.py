import streamlit as st
from model_helper import predict

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Vehicle Damage Detection",
    page_icon="🚗",
    layout="centered"
)

# --------------------------------------------------
# Header Section
# --------------------------------------------------
st.markdown(
    "<h1 style='text-align: center;'>🚗 Vehicle Damage Detection</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; color: gray;'>Upload a vehicle image to detect damage type</p>",
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# Sidebar (Demo / Info)
# --------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        """
        This application uses a **deep learning model (ResNet-50)**  
        to classify **vehicle damage types** from images.
        """
    )
    st.markdown("**Classes:**")
    st.markdown(
        """
        - Front Breakage  
        - Front Crushed  
        - Front Normal  
        - Rear Breakage  
        - Rear Crushed  
        - Rear Normal  
        """
    )

# --------------------------------------------------
# File Upload Section
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a vehicle image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image_path = "temp_file.jpg"

    # Save uploaded image
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show image in a container
    st.markdown("### 🖼️ Uploaded Image")
    st.image(
        uploaded_file,
        use_container_width=True
    )

    # Prediction
    with st.spinner("🔍 Analyzing image..."):
        prediction = predict(image_path)

    # Result display
    st.success(f"✅ **Predicted Damage Class:** {prediction}")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Demo Project | Deep Learning + Streamlit</p>",
    unsafe_allow_html=True
)



