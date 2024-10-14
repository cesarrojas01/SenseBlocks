import pyttsx3

def speak(message):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    
    for voice in voices:
        if "Sabina" in voice.name:
            engine.setProperty('voice', voice.id)
            break
    
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)

    engine.say(message)
    engine.runAndWait()
