import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Alzheimer's Disease Classification",
    page_icon="🧠",
    layout="centered"
)

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to match your model's input size
    img = image.resize((128, 128))
    # Convert to RGB if not already
    img = img.convert('RGB')
    # Convert to array and normalize
    img_array = np.array(img) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Load the model
@st.cache_resource
def load_classification_model():
    model = load_model('tm41a.h5')
    # Add compilation to avoid the metrics warning
    model.compile(optimizer='adam', 
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model

# Main UI
def main():
    st.title("Alzheimer's Disease Classification")
    st.write("Upload a brain MRI scan to classify the stage of Alzheimer's Disease")

    # Load model
    try:
        model = load_classification_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_container_width=False)

            # Add a prediction button
            if st.button('Predict'):
                # Preprocess the image
                processed_image = preprocess_image(image)

                # Make prediction
                prediction = model.predict(processed_image)
                class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']  # Replace with your class names
                
                # Get the predicted class
                predicted_class = class_names[np.argmax(prediction)]
                confidence = np.max(prediction) * 100

                # Display results
                st.success(f"Predicted Class: {predicted_class}")
                st.info(f"Confidence: {confidence:.2f}%")

                # Display all class probabilities
                st.write("Class Probabilities:")
                for class_name, prob in zip(class_names, prediction[0]):
                    st.write(f"{class_name}: {prob*100:.2f}%")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    # Add information about the model
    with st.expander("About this Model"):
        st.write("""
        This model classifies brain MRI scans into four categories:
        - Non Demented
        - Very Mild Demented
        - Mild Demented
        - Moderate Demented
        
        The model expects images of size 128x128 pixels.
        """)

if __name__ == "__main__":
    main()