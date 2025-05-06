import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("xception_deepfake_detector.h5")
    return model

model = load_model()

# Preprocess image
def preprocess_image(image):
    image = image.resize((299, 299))
    image_array = np.array(image)
    if image_array.shape[-1] == 4:  # Remove alpha channel
        image_array = image_array[:, :, :3]
    image_array = image_array / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Streamlit UI
st.title("🧠 Deepfake Image Detector (XceptionNet)")

st.write("Upload a face image to check if it's **real** or **AI-generated (fake)**.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)[0][0]

        st.subheader("Result:")
        if prediction > 0.5:
            st.error(f"🔴 Fake Image Detected! Confidence: {prediction:.2f}")
        else:
            st.success(f"🟢 Real Image Detected! Confidence: {1 - prediction:.2f}")
