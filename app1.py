import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Rotten Fruit Detection",
    page_icon="🍎",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f9fafb;
}
.title {
    font-size: 40px;
    font-weight: bold;
    color: #2c3e50;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: #7f8c8d;
    font-size: 18px;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<div class='title'>🍎 Rotten Fruit Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>CNN-based Image Classification</div><br>", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/fruit_classifier_cnn.h5")

model = load_model()

CLASS_NAMES = [
    "Fresh Apple", "Fresh Banana", "Fresh Orange",
    "Rotten Apple", "Rotten Banana", "Rotten Orange"
]

# ---------------- UPLOAD IMAGE ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "📤 Upload a fruit image",
    type=["jpg", "png", "jpeg"]
)
st.markdown("</div><br>", unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((100, 100))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if st.button("🔍 Predict"):
            predictions = model.predict(img_array)
            confidence = np.max(predictions) * 100
            predicted_index = np.argmax(predictions)
            label = CLASS_NAMES[predicted_index]

            # Result
            st.success(f"✅ **Prediction:** {label}")
            st.info(f"📊 **Confidence:** {confidence:.2f}%")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- PROBABILITY PLOT ----------------
    if "predictions" in locals():
        st.markdown("### 📈 Prediction Probabilities")

        probs = predictions[0]

        fig, ax = plt.subplots()
        ax.barh(CLASS_NAMES, probs)
        ax.set_xlabel("Probability")
        ax.set_xlim(0, 1)

        for i, v in enumerate(probs):
            ax.text(v + 0.01, i, f"{v*100:.1f}%", va='center')

        st.pyplot(fig)

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<center>
<b>Developed by Vittal Rahul</b> | CNN Image Classification Project  
</center>
""", unsafe_allow_html=True)
