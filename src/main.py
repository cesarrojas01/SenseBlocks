import cv2
import speech_recognition as sr
from interpreter.lexer import Lexer
from interpreter.parser import Parser
from interpreter.evaluator import Evaluator
from interpreter.ast_node import ast_to_yaml, ast_to_json
from helpers.speak import speak

import cv2

def capture_image(camera_index=1, save_path='captured_image.png'):
    # Inicia la captura de video desde la cámara
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: No se puede acceder a la cámara.")
        return None

    # Establece la resolución a 1920x1080 (Full HD)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("Presiona 'c' para capturar la imagen o 'q' para salir.")

    while True:
        # Lee el frame actual de la cámara
        ret, frame = cap.read()
        if not ret:
            print("Error: No se puede capturar el frame.")
            break

        # Muestra el video en una ventana
        cv2.imshow('Vista previa de la cámara', frame)

        # Espera a que el usuario presione 'c' para capturar o 'q' para salir
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Guarda la imagen capturada
            cv2.imwrite(save_path, frame)
            print(f"Imagen capturada y guardada en {save_path}")
            break
        elif key == ord('q'):
            print("Salida sin capturar imagen.")
            break

    # Libera los recursos
    cap.release()
    cv2.destroyAllWindows()
    return save_path

def main(image_path: str):
    try:
        # Fase de Lexing
        lexer = Lexer(image_path)
        tokens = lexer.tokenize()
        print(f"Generated tokens: {tokens}")

        # Fase de Parsing
        parser = Parser(tokens)
        ast = parser.parse()
        print("Abstract Syntax Tree (AST):")
        print(ast_to_yaml(ast))

        # Fase de Evaluación
        evaluator = Evaluator()
        evaluator.evaluate(ast)

    except Exception as e:
        speak(f"Error procesando imagen: {e}")
        print(f"Error processing image: {e}")

def recognize_voice():
    # Inicializa el reconocimiento de voz
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Ajustando micrófono para el ruido ambiental...")
        recognizer.adjust_for_ambient_noise(source)
        print("Micrófono listo, di 'Correr' para comenzar...")
        
        while True:
            print("Escuchando...")
            audio = recognizer.listen(source)

            try:
                # Reconoce el audio
                voice_input = recognizer.recognize_google(audio, language="es-ES")
                print(f"Escuchado: {voice_input}")

                if "correr" in voice_input.lower():
                    print("Iniciando captura de imagen y ejecución del programa.")
                    image_path = capture_image()  # Captura la imagen
                    if image_path:
                        main(image_path)  # Procesa la imagen capturada

            except sr.UnknownValueError:
                print("No se entendió lo que dijiste. Intenta nuevamente.")
            except sr.RequestError as e:
                print(f"Error de reconocimiento de voz: {e}")
            except Exception as e:
                print(f"Error inesperado: {e}")

if __name__ == "__main__":
    # image = capture_image()
    # main(image)
    main("src/images/p3.png")
