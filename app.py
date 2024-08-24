import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

from PIL import Image
# Load the saved model
model = load_model('disease_classification_model.h5')
####################general
st.write("🦷 Dental Disease Classifier: AI-Powered Diagnosis 💻")
# Load the image using PIL
img = Image.open('./background.jpeg')

# Get the original width and height
original_width, original_height = img.size

# Display the image with double the original width
st.image(img, caption='Image with Face Detections' )

Illness=['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']


def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(128, 128))  
    img = np.array(img) / 255.0  
    img = np.expand_dims(img, axis=0) 
    return img
# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the image
    img = preprocess_image(uploaded_file)
    
    # Predict the class
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_probability = prediction[0][predicted_class]
    
    st.write(f"Predicted Class: {Illness[predicted_class]}")
    st.write(f"Probability: {predicted_probability:.4f}")
