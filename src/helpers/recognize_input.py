import speech_recognition as sr

class SpeechRecognitionService:
    """
    Servicio para el reconocimiento de voz y procesamiento de entrada.
    Proporciona funciones para capturar la entrada por voz y limpiarla.
    """
    
    NUMBER_WORDS_TO_INT = {
        "cero": 0, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, 
        "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, 
        "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, 
        "quince": 15, "dieciséis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19, 
        "veinte": 20, "treinta": 30, "cuarenta": 40, "cincuenta": 50,
        # Puedes agregar más números si es necesario
    }

    def __init__(self):
        self.recognizer = sr.Recognizer()

    def recognize_speech(self) -> str:
        """
        Captura y convierte la entrada de voz a texto usando el servicio de Google.

        Returns:
            str: Texto reconocido o un mensaje de error si ocurre una excepción.
        """
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("Por favor, habla ahora...")
            audio = self.recognizer.listen(source)
            return self._convert_audio_to_text(audio)
    
    def _convert_audio_to_text(self, audio: sr.AudioData) -> str:
        """
        Convierte el audio capturado en texto utilizando Google Speech Recognition.

        Args:
            audio (sr.AudioData): El audio capturado.

        Returns:
            str: Texto reconocido o mensaje de error.
        """
        try:
            return self.recognizer.recognize_google(audio, language="es-ES")
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            return f"Error de solicitud al servicio de reconocimiento: {e}"

    def _convert_text_to_number(self, text: str):
        """
        Convierte texto que representa un número a su valor numérico, si es posible.

        Args:
            text (str): Texto posiblemente numérico.

        Returns:
            int, str: Entero si es un número, cadena si no se puede convertir.
        """
        text = text.lower().strip()
        
        # Si el texto es un número en palabras, lo convertimos
        if text in self.NUMBER_WORDS_TO_INT:
            return self.NUMBER_WORDS_TO_INT[text]
        
        # Si el texto es un número en formato numérico
        try:
            return int(text)  # Convertir directamente a int si es un número
        except ValueError:
            return text  # Devolver como cadena si no es un número válido


def recognize_and_process_input():
    """
    Función principal para reconocer y procesar la entrada de voz.
    Captura la entrada y devuelve el valor procesado.

    Returns:
        int, str: Valor numérico o cadena procesada o mensaje de error.
    """
    recognizer_service = SpeechRecognitionService()
    
    recognized_text = recognizer_service.recognize_speech()
    
    if not recognized_text:
        return "No se pudo reconocer la entrada. Intenta nuevamente."
    
    return recognizer_service._convert_text_to_number(recognized_text)
