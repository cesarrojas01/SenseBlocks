import cv2
from interpreter.lexer import Lexer
from interpreter.parser import Parser
from interpreter.evaluator import Evaluator
from interpreter.ast_node import ast_to_yaml, ast_to_json
from helpers.speak import speak
from helpers.recognize_input import recognize_speech


def capture_image(camera_index=0, save_path='captured_image.png'):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: No se puede acceder a la cámara.")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("Presiona 'c' para capturar la imagen o 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se puede capturar el frame.")
            break

        cv2.imshow('Vista previa de la cámara', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            cv2.imwrite(save_path, frame)
            print(f"Imagen capturada y guardada en {save_path}")
            break
        elif key == ord('q'):
            print("Salida sin capturar imagen.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return save_path


def process_code(image_path: str):
    lexer = Lexer(image_path)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    evaluator = Evaluator()
    
    return tokens, ast, evaluator

def list_tokens(tokens):
    # Convierte cada token a su representación de cadena
    token_sequence = ', '.join([str(token.type.name) for token in tokens])
    speak(f"Los tokens detectados son: {token_sequence}")
    print(f"Tokens: {token_sequence}")



def verify_code(evaluator, ast):
    evaluator.evaluate(ast)
    speak("El código no contiene errores.")
    print("El código no contiene errores.")


def execute_code(evaluator, ast):
    evaluator.evaluate(ast)
    speak("El código se ha ejecutado correctamente.")
    print("Código ejecutado correctamente.")


def main(image_path):
    while True:
        speak("Ingresa el comando que deseas. Puedes decir 'listar', 'verificar' o 'ejecutar'.")

        voice_input = recognize_speech()

        if voice_input:
            voice_input = voice_input.lower()
            print(f"Comando recibido: {voice_input}")

            if voice_input in ["listar", "lista"]:
                speak("Listando tokens detectados.")  
                if image_path:
                    tokens, ast, evaluator = process_code(image_path)
                    list_tokens(tokens)

            elif voice_input in ["verificar", "verifica"]:
                speak("Verificando código.")  
                if image_path:
                    tokens, ast, evaluator = process_code(image_path)
                    verify_code(evaluator, ast)

            elif voice_input in ["ejecutar", "ejecuta"]:
                speak("Ejecutando el código.")  
                if image_path:
                    tokens, ast, evaluator = process_code(image_path)
                    execute_code(evaluator, ast)
        else:
            speak("No se pudo entender el comando. Intenta nuevamente.")


# def main(image_path):
#     lexer = Lexer(image_path)
#     tokens = lexer.tokenize()
#     print(tokens)

#     parser = Parser(tokens)
#     ast = parser.parse()
#     print(ast_to_yaml(ast))

#     evaluator = Evaluator()
#     evaluator.evaluate(ast)


if __name__ == "__main__":
    # image_path = capture_image()
    image_path = "src/images/bitmap1.png"
    main(image_path)
