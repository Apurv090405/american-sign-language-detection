import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import datetime

# Load your saved models
image_model_path = "model_efficientnet_finetuned.h5"
video_model_path = "model_efficientnet_finetuned.h5"
image_model = load_model(image_model_path)
video_model = load_model(video_model_path)

alphabet_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
                   'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
                   'V', 'W', 'X', 'Y']

def preprocess_image(img):
    img = cv2.resize(img, (32, 32))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_video_frame(frame):
    frame = cv2.resize(frame, (32, 32))
    frame = frame.astype('float32') / 255.0
    return frame

def predict_from_frame(frame):
    preprocessed_frame = preprocess_image(frame)
    predictions = image_model.predict(preprocessed_frame)
    predicted_class = alphabet_labels[np.argmax(predictions)]
    return predicted_class

def predict_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        preprocessed_frame = preprocess_video_frame(frame)
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
        prediction = video_model.predict(preprocessed_frame)
        predicted_class = alphabet_labels[np.argmax(prediction)]
        predictions.append(predicted_class)
    
    cap.release()
    return predictions

def is_within_time_range(start_time, end_time):
    now = datetime.datetime.now().time()
    start = datetime.time(*start_time)
    end = datetime.time(*end_time)
    return start <= now <= end

st.title("Hand Gesture Recognition")

# Option to choose Upload Image, Upload Video, or Use Webcam
st.sidebar.title("Choose Mode")
app_mode = st.sidebar.radio(
    "Select mode:",
    ('Upload Image', 'Upload Video', 'Use Webcam')
)

if is_within_time_range((14, 0, 0), (21, 30, 0)):  # Between 6 PM and 9 PM
    if app_mode == 'Upload Image':
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            st.image(img, caption='Uploaded Image.', use_column_width=True)
            predicted_class = predict_from_frame(img)
            st.write(f"Prediction: {predicted_class}")

    elif app_mode == 'Upload Video':
        st.subheader("Upload Video")
        uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

        if uploaded_video is not None:
            video_bytes = uploaded_video.read()
            video_path = 'uploaded_video.mp4'
            with open(video_path, 'wb') as f:
                f.write(video_bytes)

            predictions = predict_from_video(video_path)
            st.write(f"Predictions: {predictions}")

    elif app_mode == 'Use Webcam':
        st.subheader("Use Webcam")
        run_webcam = st.checkbox("Run Webcam")
        FRAME_WINDOW = st.image([])
        video_capture = cv2.VideoCapture(0)

        if run_webcam:
            prediction_text = st.empty()  # Placeholder for prediction text

            while run_webcam:  # Loop while webcam checkbox is checked
                ret, frame = video_capture.read()
                if not ret:
                    st.warning('Error: Check if your webcam is working.')
                    break

                predicted_class = predict_from_frame(frame)

                # Display the frame with prediction
                cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                st.image(frame, channels="BGR", caption='Webcam', use_column_width=True)

                # Update prediction text
                prediction_text.text(f"Prediction: {predicted_class}")

                # Check if user wants to stop webcam
                if not st.sidebar.checkbox("Stop Webcam"):
                    run_webcam = False  # Set run_webcam to False to break the loop

            video_capture.release()
            FRAME_WINDOW.image(frame)
            video_capture.release()
            cv2.destroyAllWindows()

else:
    st.warning("The application is available only between 6 PM and 9 PM.")

st.markdown("""
    ### Instructions:
    - **Upload Image:** Choose an image file (jpg, jpeg, png) for prediction.
    - **Upload Video:** Choose a video file (mp4, avi, mov) for prediction.
    - **Use Webcam:** Check the box to start using your webcam. Predictions will be shown in real-time.
""")
