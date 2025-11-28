import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load the Teachable Machine model
model = load_model('keras_model.h5')

# Load the labels
with open('labels.txt', 'r') as f:
    class_names = [line.strip() for line in f]


#implement image upload

st.title("Teachable Machine Image Classifier")

uploaded_file = st.file_uploader("Upload an image for classification", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify Image"):
        # Preprocess the image for the model
        image = image.resize((224, 224))  # Resize to the input size of your TM model
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Make a prediction
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index] * 100

        st.write(f"Prediction: {class_name}")
        st.write(f"Confidence: {confidence_score:.2f}%")