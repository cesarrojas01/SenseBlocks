import cv2
import numpy as np
import logging
from interpreter.tokens import Token, TokenType
from typing import List, Dict

class LexerError(Exception):
    """Excepción lanzada cuando ocurre un error durante el análisis léxico."""
    pass

class Lexer:
    """
    Analizador léxico que convierte una imagen en una secuencia de tokens.
    """

    # Rango de colores en formato HSV para cada categoría de bloque
    # CATEGORY_COLOR_RANGES = {
    #     'NUMBER': ((20, 150, 150), (30, 255, 255)),                # Amarillo (ajustado)
    #     'CONTROL_STRUCTURE': ((35, 100, 100), (80, 255, 255)),     # Verde (ajustado)
    #     'OPERATOR': ((85, 150, 100), (110, 255, 255)),             # Cian (ajustado)
    #     'SET': ((5, 150, 100), (18, 255, 255)),                    # Naranja (ajustado)
    #     'VARIABLE': ((135, 100, 100), (165, 255, 255)),            # Fucsia (ajustado)
    #     'PRINT': ((100, 100, 100), (130, 255, 255)),               # Azul oscuro (ajustado)
    #     # 'START_END': ((170, 120, 70), (180, 255, 255))         # Rango adicional para rojo
    #     'START_END': ((0, 100, 100), (10, 255, 255))         # Rango adicional para rojo
    # }

    CATEGORY_COLOR_RANGES = {
        'NUMBER': ((25, 150, 150), (35, 255, 255)),              # Amarillo ajustado
        'CONTROL_STRUCTURE': ((35, 100, 100), (85, 255, 255)),   # Verde ajustado
        'OPERATOR': ((85, 150, 100), (100, 255, 255)),           # Cyan ajustado
        'SET': ((10, 150, 100), (25, 255, 255)),                 # Naranja ajustado
        'VARIABLE': ((140, 100, 100), (160, 255, 255)),          # Fucsia ajustado
        'PRINT': ((100, 100, 100), (120, 255, 255)),             # Azul ajustado
        'START_END': ((0, 100, 100), (10, 255, 255))             # Rojo ajustado
    }

    # Mapeo de códigos de barras a tipos de token por categoría
    CATEGORY_BARCODE_MAPPING = {
        'NUMBER': {
            '0000': (TokenType.NUMBER, 0),
            '0001': (TokenType.NUMBER, 1),
            '0010': (TokenType.NUMBER, 2),
            '0011': (TokenType.NUMBER, 3),
            '0100': (TokenType.NUMBER, 4),
            '0101': (TokenType.NUMBER, 5),
            '0110': (TokenType.NUMBER, 6),
            '0111': (TokenType.NUMBER, 7),
            '1000': (TokenType.NUMBER, 8),
            '1001': (TokenType.NUMBER, 9),
        },
        'VARIABLE': {
            '0000': (TokenType.IDENTIFIER, 'A'),
            '0001': (TokenType.IDENTIFIER, 'B'),
            '0010': (TokenType.IDENTIFIER, 'C'),
            '0011': (TokenType.IDENTIFIER, 'D'),
            '0100': (TokenType.IDENTIFIER, 'E'),
        },
        'CONTROL_STRUCTURE': {
            '0000': TokenType.FOR_START,
            '0001': TokenType.FOR_END,
            '0010': TokenType.WHILE_START,
            '0011': TokenType.WHILE_END,
            '0100': TokenType.IF_START,
            '0101': TokenType.IF_ELSE,
            '0110': TokenType.IF_END,
            '0111': TokenType.TO,
        },
        'OPERATOR': {
            '0000': TokenType.PLUS,
            '0001': TokenType.MINUS,
            '0010': TokenType.MULTIPLY,
            '0011': TokenType.DIVIDE,
            '0100': TokenType.GREATER,
            '0101': TokenType.LESS,
            '0110': TokenType.GREATER_EQUAL,
            '0111': TokenType.LESS_EQUAL,
            '1000': TokenType.EQUAL,
            '1001': TokenType.DIFFERENT,
            '1010': TokenType.AND,
            '1011': TokenType.OR,
            '1100': TokenType.NOT,
            '1101': TokenType.LPAREN,
            '1110': TokenType.RPAREN,
            '1111': TokenType.MODULO
        },
        'SET': {
            '0000': TokenType.SET
        },
        'PRINT': {
            '0000': TokenType.SAY,
            '1111': TokenType.INPUT,
        },
        'START_END': {
            '0000': TokenType.START,
            '1111': TokenType.END
        }
    }

    def __init__(self, image_path: str, display=False):
        """
        Inicializa una nueva instancia del Lexer.

        Args:
            image_path (str): La ruta al archivo de imagen que contiene el código.
            display (bool): Si es True, muestra imágenes intermedias para depuración.
        """
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"El archivo de imagen no se encontró en la ruta: {image_path}")

        # Convertir la imagen de BGR a HSV para trabajar con los rangos de color
        self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.display = display
        self.blocks = []

    def tokenize(self) -> List[Token]:
        """
        Procesa la imagen y genera la lista de tokens correspondiente.

        Returns:
            List[Token]: La lista de tokens generada a partir de la imagen.
        """
        bounding_boxes = self._extract_bounding_boxes()
        rows = self._group_blocks_by_row(bounding_boxes)
        self._extract_blocks_from_rows(rows)

        # Convertir los bloques detectados en tokens
        tokens = []
        for block in self.blocks:
            x, y, w, h = block['position']
            
            # Obtener el color dominante en el área del bloque
            r_med, g_med, b_med = self.get_dominant_color(x, y, w, h)
            
            # Convertir de RGB a HSV para comparar con los rangos de color
            dominant_hsv = cv2.cvtColor(np.uint8([[[r_med, g_med, b_med]]]), cv2.COLOR_BGR2HSV)[0][0]
            
            # Identificar la categoría del bloque basándose en el color dominante
            category = self._detect_category_by_color(dominant_hsv)
            block['category'] = category
            
            # Mapeo del bloque a un token
            block_token = self._map_block_to_token(block)
            if block_token:
                tokens.append(block_token)

        return tokens
    
    def get_dominant_color(self, x, y, w, h) -> tuple:
        """Obtiene el color dominante en un área."""
        roi = self.image[y:y+h, x:x+w]
        pixels = np.float32(roi.reshape(-1, 3))
        r_med, g_med, b_med = np.median(pixels, axis=0)
        return int(r_med), int(g_med), int(b_med)

    def _detect_category_by_color(self, dominant_hsv: tuple) -> str:
        """
        Detecta la categoría del bloque basada en el color dominante.

        Args:
            dominant_hsv (tuple): El color dominante en formato HSV.

        Returns:
            str: La categoría detectada del bloque.
        """
        for category, (lower_range, upper_range) in self.CATEGORY_COLOR_RANGES.items():
            if lower_range[0] <= dominant_hsv[0] <= upper_range[0] and \
            lower_range[1] <= dominant_hsv[1] <= upper_range[1] and \
            lower_range[2] <= dominant_hsv[2] <= upper_range[2]:
                return category
        return 'UNKNOWN'

    def _map_block_to_token(self, block: dict) -> Token:
        """
        Convierte un bloque en un token usando la categoría y el código de barras.

        Args:
            block (dict): Diccionario que contiene los datos del bloque detectado.

        Returns:
            Token: El token correspondiente al bloque.
        """
        category = block['category']
        barcode = block['barcode']
        
        if not barcode:
            return None

        block_mapping = self.CATEGORY_BARCODE_MAPPING.get(category, {})

        if barcode in block_mapping:
            token_data = block_mapping[barcode]
            if isinstance(token_data, tuple):
                token_type, value = token_data
                return Token(token_type, value)
            return Token(token_data)
        
        return Token(TokenType.UNKNOWN)

    def _extract_bounding_boxes(self) -> List[tuple]:
        """
        Extrae los bounding boxes de los bloques en la imagen.

        Returns:
            List[tuple]: Lista de tuplas que contienen las coordenadas, contornos y categoría de cada bloque.
        """
        bounding_boxes = []

        for category, (lower_range, upper_range) in self.CATEGORY_COLOR_RANGES.items():
            lower = np.array(lower_range, dtype=np.uint8)
            upper = np.array(upper_range, dtype=np.uint8)
            mask = cv2.inRange(self.hsv_image, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Filtrar contornos demasiado pequeños
                if w < 50 or h < 50:
                    continue
                bounding_boxes.append((x, y, w, h, contour, category))

        return bounding_boxes

    def _group_blocks_by_row(self, bounding_boxes: List[tuple], y_threshold: int = 20) -> List[List[tuple]]:
        """
        Agrupa los bloques en filas basándose en sus coordenadas Y.

        Args:
            bounding_boxes (List[tuple]): Lista de bounding boxes a agrupar.
            y_threshold (int, opcional): Umbral para determinar si un bloque pertenece a una nueva fila.

        Returns:
            List[List[tuple]]: Lista de filas, cada una con los bloques correspondientes.
        """
        if not bounding_boxes:
            return []

        bounding_boxes.sort(key=lambda box: box[1])  # Ordenar por coordenada Y

        rows = []
        current_row = [bounding_boxes[0]]
        current_y = bounding_boxes[0][1]

        for box in bounding_boxes[1:]:
            _, y, _, _, _, _ = box
            if abs(y - current_y) > y_threshold:
                rows.append(current_row)
                current_row = [box]
                current_y = y
            else:
                current_row.append(box)

        if current_row:
            rows.append(current_row)

        return rows

    def _extract_blocks_from_rows(self, rows: List[List[tuple]]):
        """
        Extrae bloques de las filas de bloques agrupados.

        Args:
            rows (List[List[tuple]]): Lista de filas con bloques.
        """
        for row in rows:
            row.sort(key=lambda box: box[0])  # Ordenar por coordenada X
            for x, y, w, h, contour, category in row:
                masked_image = self._mask_block_area(contour)
                barcode = self._decode_barcode(masked_image, (x, y, w, h))
                self.blocks.append({'category': category, 'barcode': barcode, 'position': (x, y, w, h)})

                if self.display:
                    block_image = masked_image[y:y+h, x:x+w].copy()
                    cv2.putText(block_image, f"{category}_{barcode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    # cv2.imshow(f"Bloque: {category}_{barcode}", block_image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

    def _mask_block_area(self, contour) -> np.ndarray:
        """
        Aplica una máscara a la imagen para mantener solo el área exacta del bloque y la muestra con imshow.

        Args:
            contour: El contorno del bloque.

        Returns:
            np.ndarray: La imagen enmascarada.
        """
        # Crear una máscara del tamaño de la imagen con ceros
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        
        # Dibujar el contorno en la máscara
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Aplicar la máscara a la imagen
        masked_image = cv2.bitwise_and(self.image, self.image, mask=mask)
        
        # Mostrar la imagen enmascarada
        # cv2.imshow("Masked Block Area", masked_image)
        # cv2.waitKey(0)  # Espera hasta que se presione una tecla
        # cv2.destroyAllWindows()  # Cierra la ventana

        return masked_image

    def _decode_barcode(self, masked_image: np.ndarray, bbox: tuple) -> str:
        """
        Decodifica el código de barras dentro de un bloque enmascarado.

        Args:
            masked_image: La imagen del bloque enmascarado.
            bbox: Bounding box del bloque (x, y, w, h).

        Returns:
            str: Código binario de 4 bits que identifica el bloque.
        """
        x, y, w, h = bbox
        barcode_roi = masked_image[y + int(h * 0.3):y + int(h * 0.7), x + int(w * 0.2):x + int(w * 0.8)]
        barcode_gray = cv2.cvtColor(barcode_roi, cv2.COLOR_BGR2GRAY)
        barcode_blur = cv2.GaussianBlur(barcode_gray, (5, 5), 0)
        _, barcode_thresh = cv2.threshold(barcode_blur, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Operaciones morfológicas para limpiar la imagen
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        barcode_clean = cv2.morphologyEx(barcode_thresh, cv2.MORPH_OPEN, kernel)
        # cv2.imshow('barcode_clean', barcode_clean)
        # cv2.waitKey(0)
        contours, _ = cv2.findContours(barcode_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrar contornos que no sean barras y obtener sus anchos
        bar_contours = [cnt for cnt in contours if self._is_valid_bar(cnt)]
        bar_contours.sort(key=lambda cnt: cv2.boundingRect(cnt)[0])

        if len(bar_contours) < 5:
            logging.warning("No se detectaron 5 barras en el código de barras.")
            return None

        ref_width = cv2.boundingRect(bar_contours[0])[2]
        code = ''.join(['1' if cv2.boundingRect(cnt)[2] >= ref_width * 1.5 else '0' for cnt in bar_contours[1:5]])
        
        print(f"Decoded barcode for block at position {bbox}: {code}")

        barcode_contour_img = cv2.cvtColor(barcode_clean, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(barcode_contour_img, bar_contours, -1, (0, 255, 0), 2)
        # cv2.imshow("Barcode Contours", barcode_contour_img)
        # cv2.waitKey(0)
        
        return code

    def _is_valid_bar(self, contour) -> bool:
        """
        Verifica si un contorno representa una barra válida en el código de barras.

        Args:
            contour: Contorno a verificar.

        Returns:
            bool: True si el contorno es válido como barra, False en caso contrario.
        """
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / float(w)
        return aspect_ratio > 1.5 and cv2.contourArea(contour) > 50
