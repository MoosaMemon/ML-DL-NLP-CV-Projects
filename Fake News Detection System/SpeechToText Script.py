import speech_recognition as sr
import speech_recognition as sr


def recognize_speech_from_mic(recognizer, microphone, timeout=1):
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=timeout)
    try:
        response = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        response = None
    except sr.RequestError:
        response = None
    return response


recognizer = sr.Recognizer()
microphone = sr.Microphone()
response = recognize_speech_from_mic(recognizer, microphone, timeout=6)
if response is not None:
        print(f"You said: {response}")
else:
     print("Sorry, I didn't catch that.")
