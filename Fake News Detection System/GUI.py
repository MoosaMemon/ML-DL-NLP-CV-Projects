import streamlit as st
import speech_recognition as sr
from moviepy.editor import VideoFileClip
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import nltk
import tempfile
import os
# import pyaudio
# import wave

# Download necessary nltk resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the speech recognizer
r = sr.Recognizer()

def audio_to_text(audio_file_path):
    with sr.AudioFile(audio_file_path) as source:
        audio_data = r.record(source)
    try:
        return r.recognize_google(audio_data)
    except (sr.UnknownValueError, sr.RequestError) as e:
        return f"Error processing audio: {str(e)}"

def video_to_text(video_file_buffer):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as video_file:
        video_file.write(video_file_buffer.read())
        video_path = video_file.name
    audio_path = tempfile.mktemp(suffix=".wav")
    try:
        with VideoFileClip(video_path) as video:
            video.audio.write_audiofile(audio_path)
        return audio_to_text(audio_path)
    finally:
        os.unlink(video_path)
        os.unlink(audio_path)

# Define a function for text pre-processing
def text_preprocessing(text):
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens]).lower()

# Streamlit UI
st.title("Fake News Detection System")
model_options = ["LogisticRegression", "NaiveBayes", "SupportVectorMachine", "RandomForestClassifier"]
selected_model = st.selectbox("Select a model", model_options)

input_type = st.radio("Choose your input type", ["Upload Video", "Upload Audio", "Manual Text Input"])

if input_type == "Upload Video":
    uploaded_file = st.file_uploader("Upload your video file", type=['mp4', 'avi'])
    if uploaded_file is not None:
        title_input = uploaded_file.name
        text_input = video_to_text(uploaded_file)
elif input_type == "Upload Audio":
    uploaded_file = st.file_uploader("Upload your audio file", type=['mp3', 'wav'])
    if uploaded_file is not None:
        title_input = uploaded_file.name
        text_input = audio_to_text(uploaded_file)
elif input_type == "Manual Text Input":
    title_input = st.text_input("Enter title")
    text_input = st.text_area("Enter text")

if st.button("Predict"):
    if title_input and text_input:
        input_data = title_input + " " + text_input
        preprocessed_data = text_preprocessing(input_data)
        model = joblib.load(f"SavedModels/{selected_model}.joblib")
        vectorizer = joblib.load("SavedModels/Vectorizer.joblib")
        vectorized_data = vectorizer.transform([preprocessed_data])
        prediction = model.predict(vectorized_data)
        label = "real!!" if prediction[0] == 0 else "fake!!"
        st.write("Prediction:", label)
    else:
        st.write("Please provide appropriate input based on selected type")