import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page setup
st.set_page_config(page_title="Rotten Fruit Detection", layout="centered")
st.title("🍎 Rotten Fruit Detection Using CNN")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/fruit_classifier_cnn.h5")

model = load_model()
CLASS_NAMES = ["Fresh Apple", "Fresh Banana", "Fresh Orange",
               "Rotten Apple", "Rotten Banana", "Rotten Orange"]

# Upload image
uploaded_file = st.file_uploader("Upload Fruit Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((100,100))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    if st.button("Predict"):
        predictions = model.predict(img_array)
        confidence = np.max(predictions)*100
        label = CLASS_NAMES[np.argmax(predictions)]
        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {confidence:.2f}%")
