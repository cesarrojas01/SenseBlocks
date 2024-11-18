# Tesis: Sistema Interactivo para la Enseñanza de Programación a Niños con Discapacidad Visual 

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Licencia MIT](https://img.shields.io/badge/license-MIT-green)
![Estado](https://img.shields.io/badge/status-En%20Desarrollo-yellow)

Este proyecto es un sistema interactivo educativo diseñado para niños con discapacidad visual. La herramienta permite a los niños utilizar bloques físicos para crear programas que son reconocidos mediante visión por computadora, interpretados como un lenguaje de programación accesible, y ejecutados con retroalimentación auditiva. 

El objetivo principal es fomentar el aprendizaje de habilidades computacionales y de programación en un entorno inclusivo y accesible.

## Características

- **Reconocimiento de bloques físicos:** El sistema utiliza visión por computadora para identificar bloques de colores con códigos de barras únicos. Cada bloque representa un componente de programación, como números, variables, operadores, o estructuras de control.
- **Entrada de voz:** Los usuarios pueden interactuar con el sistema a través de comandos de voz, lo que elimina la necesidad de interfaces visuales complejas.
- **Ejecución de programas:** Los programas construidos con los bloques son interpretados y ejecutados automáticamente, permitiendo a los niños escuchar los resultados.
- **Retroalimentación por audio:** Se utiliza síntesis de voz para proporcionar mensajes claros, informando sobre los tokens detectados, errores en el código, y resultados de ejecución.
- **Diseño modular:** La arquitectura del proyecto está organizada en módulos independientes que facilitan su mantenimiento y expansión.

---

## Tabla de Contenidos

1. [Instalación](#instalación)
2. [Uso](#uso)
3. [Estructura del Proyecto](#estructura-del-proyecto)
4. [Flujo de Trabajo](#flujo-de-trabajo)
5. [Dependencias](#dependencias)
6. [Licencia](#licencia)
7. [Contribuciones](#contribuciones)

---

## Instalación

1. **Clona este repositorio:**
   ```bash
   git clone https://github.com/cesarrojas01/SenseBlocks.git
   cd SenseBlocks
   ```

2. **Crea y activa un entorno virtual (opcional):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instala las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verifica que todo esté instalado correctamente:**
   ```bash
   python main.py
   ```

---

## Uso

### Capturar una Imagen

1. Conecta una cámara al dispositivo.
2. Ejecuta el script principal del sistema:
   ```bash
   python main.py
   ```
3. Sigue las instrucciones en pantalla:
   - Presiona `c` para capturar la imagen de los bloques.
   - Presiona `q` para salir sin capturar una imagen.

### Reconocer y Ejecutar Código

1. Coloca los bloques físicos en la vista de la cámara.
2. Una vez que la imagen sea capturada, el sistema identificará los bloques y construirá el programa basado en su disposición.
3. Usa comandos de voz para interactuar:
   - **`listar`**: Muestra todos los tokens detectados en la imagen.
   - **`verificar`**: Analiza el programa para comprobar que no contiene errores.
   - **`ejecutar`**: Interpreta y ejecuta el programa detectado.

---

## Estructura del Proyecto

```plaintext
src/
├── helpers/
│   ├── __init__.py
│   ├── recognize_input.py
│   ├── speak.py
├── interpreter/
│   ├── __init__.py
│   ├── ast_node.py
│   ├── evaluator.py
│   ├── lexer.py
│   ├── parser.py
│   ├── tokens.py
├── main.py
├── captured_image.png
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

---

## Flujo de Trabajo

### 1. Captura de Imagen
El sistema utiliza la cámara para capturar una imagen que contiene los bloques físicos organizados por el usuario. Los bloques deben estar bien iluminados y visibles para garantizar una detección precisa.

### 2. Tokenización
La imagen es procesada por el analizador léxico (`Lexer`), que identifica los bloques en función de su color y código de barras. Cada bloque es clasificado en una categoría como operadores, números, variables, o estructuras de control.

### 3. Construcción del AST
El analizador sintáctico (`Parser`) organiza los tokens en un Árbol de Sintaxis Abstracta (AST). Este árbol representa la estructura lógica del programa y define cómo se deben evaluar las instrucciones.

### 4. Ejecución del Código
El evaluador (`Evaluator`) recorre el AST y ejecuta cada instrucción en orden. Los resultados, errores o mensajes de estado son proporcionados al usuario mediante retroalimentación por voz.

---

## Dependencias

El proyecto utiliza las siguientes bibliotecas principales:

- **OpenCV**: Procesamiento de imágenes y detección de bloques.
- **SpeechRecognition**: Reconocimiento de comandos de voz.
- **pyttsx3**: Generación de mensajes de voz.
- **NumPy**: Operaciones matriciales necesarias para el análisis de imágenes.
- **Matplotlib**: Depuración visual de procesos de análisis.
- **PyAudio**: Captura de audio para el reconocimiento de voz.

Para instalar todas las dependencias, ejecuta:
```bash
pip install -r requirements.txt
```

---

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más información.

---

## Contribuciones

¡Contribuciones son bienvenidas! Si deseas mejorar el sistema, agregar nuevas funcionalidades, o reportar errores, por favor crea un *issue* o envía un *pull request*. Sigue estos pasos para contribuir:

1. Haz un *fork* del repositorio.
2. Crea una nueva rama para tu funcionalidad o corrección:
   ```bash
   git checkout -b contrib
   ```
3. Realiza los cambios y asegúrate de probarlos.
4. Envía tu *pull request* describiendo claramente tus cambios.
