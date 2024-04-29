import streamlit as st
import numpy as np
from deepface import DeepFace
import cv2
from PIL import Image
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions # type: ignore
from tensorflow.keras.applications import InceptionV3 
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import io

def page1():
    # Load the pre-trained InceptionV3 model
    model = InceptionV3(weights='imagenet')

    # Define allowed image file extensions
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

    def allowed_file(filename):
        """Check if the filename has a valid image extension"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def classify_image(image_bytes):
        """Classify the given image"""
        # Read the image file and preprocess it
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((299, 299))  # Resize image to match model input size for InceptionV3
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # Make predictions
        predictions = model.predict(image)

        # Decode predictions
        decoded_predictions = decode_predictions(predictions)

        # Store all predictions in a dictionary
        all_predictions = {}
        for (_, label, probability) in decoded_predictions[0]:
           all_predictions[label] = f"{probability * 100:.2f}%"

        return all_predictions

    def main():
        """Streamlit app entry point"""
        st.title("Image Classifier ")

        # File uploader
        st.write("**Upload an image:**")
        uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg', 'gif'])

        if uploaded_file is not None:
           if uploaded_file is not None:
              # Display uploaded image as preview
              st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

              if st.button("Classify"):
                image_bytes = uploaded_file.read()
                predictions = classify_image(image_bytes)

                # Display all predictions
                st.write("**Predictions:**")
                st.write("<style>table {border-collapse: collapse; width: 400px;} th, td {border: 1.5px solid Brown;padding: 8px;}</style>", unsafe_allow_html=True)
                st.write("<table><tr><th style='width: 200px; background-color: #3498db;'>Object Name</th><th style='width: 200px; background-color: #3498db;'>Accuracy</th></tr>", unsafe_allow_html=True)
                for label, accuracy in predictions.items():
                    st.write(f"<tr><td style='width: 200px;'>{label}</td><td style='width: 200px;'>{accuracy}</td></tr>", unsafe_allow_html=True)
                    st.write("</table>", unsafe_allow_html=True)

    if __name__ == '__main__':
      main()


    
def page2():
    def main():
       st.title("Face Emotion Recongnization")

       # File uploader for image
       uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

       if uploaded_file is not None:
          # Convert file to opencv format
          file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
          img = cv2.imdecode(file_bytes, 1)

          # Resize the image
          max_dimension = 250
          height, width = img.shape[:2]
          if max(height, width) > max_dimension:
             if height > width:
                img = cv2.resize(img, (int(width * max_dimension / height), max_dimension))
             else:
                img = cv2.resize(img, (max_dimension, int(height * max_dimension / width)))

          # Display the image
          st.image(img, channels="BGR", caption='Uploaded Image', use_column_width=True)

          # Analyze the image
          predictions = DeepFace.analyze(img)

          # Extract prediction values
          dominant_emotion_value = predictions[0]['dominant_emotion']
          dominant_gender_value = predictions[0]['dominant_gender']
          dominant_race_value = predictions[0]['dominant_race']

          # Display prediction values
          st.write("Emotion:", dominant_emotion_value)
          st.write("Gender:", dominant_gender_value)
          st.write("Race:", dominant_race_value)

    if __name__ == "__main__":
      main()

    
def page3():
    st.title("Page 3")
    st.write("This is Page 3.")

def main():
    st.sidebar.title("Object Detection")
    page = st.sidebar.radio(" ", ["Image Classifier", "face Emotion identifier", "Object Detection using video"])

    if page == "Image Classifier":
        page1()
    elif page == "face Emotion identifier":
        page2()
    elif page == "Page 3":
        page3()

if __name__ == "__main__":
    main()
