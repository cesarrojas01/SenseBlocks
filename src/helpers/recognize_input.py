import speech_recognition as sr

from .speak import speak

NUMBER_WORDS_TO_INT = {
    "cero": 0, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, 
    "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, 
    "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, 
    "quince": 15, "dieciséis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, 
    "veinte": 20, "treinta": 30, "cuarenta": 40, "cincuenta": 50,
    # Puedes agregar más números si es necesario
}

def recognize_speech() -> str:
    """
    Captura y convierte la entrada de voz a texto usando el servicio de Google.

    Returns:
        str: Texto reconocido o None si ocurre una excepción.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        speak("Por favor, habla ahora...")
        audio = recognizer.listen(source)
        return convert_audio_to_text(audio, recognizer)

def convert_audio_to_text(audio: sr.AudioData, recognizer: sr.Recognizer) -> str:
    """
    Convierte el audio capturado en texto utilizando Google Speech Recognition.

    Args:
        audio (sr.AudioData): El audio capturado.
        recognizer (sr.Recognizer): Instancia de reconocimiento de voz.

    Returns:
        str: Texto reconocido o None si no se reconoce.
    """
    try:
        return recognizer.recognize_google(audio, language="es-ES")
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        print(f"Error de solicitud al servicio de reconocimiento: {e}")
        return None

def convert_text_to_number(text: str):
    """
    Convierte texto que representa un número a su valor numérico, si es posible.

    Args:
        text (str): Texto posiblemente numérico.

    Returns:
        int, str: Entero si es un número, cadena si no se puede convertir.
    """
    text = text.lower().strip()

    # Si el texto es un número en palabras, lo convertimos
    if text in NUMBER_WORDS_TO_INT:
        return NUMBER_WORDS_TO_INT[text]
    
    # Si el texto es un número en formato numérico
    try:
        return int(text)  # Convertir directamente a int si es un número
    except ValueError:
        return text  # Devolver como cadena si no es un número válido

def recognize_and_process_input():
    """
    Función principal para reconocer y procesar la entrada de voz.
    Sigue solicitando la entrada si no se reconoce correctamente.

    Returns:
        int, str: Valor numérico o cadena procesada.
    """
    recognized_text = None

    while not recognized_text:
        recognized_text = recognize_speech()

        if not recognized_text:
            speak("No se pudo reconocer la entrada. Intenta nuevamente.")

    # Una vez que se obtiene el texto reconocido, procesarlo.
    return convert_text_to_number(recognized_text)
