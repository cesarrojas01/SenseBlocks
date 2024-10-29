from .ast_node import (
    ASTNode, ProgramNode, SetNode, PrintNode, IfNode, ForNode, WhileNode,
    VariableNode, NumberNode, InputNode, BinaryOperationNode, UnaryOperationNode
)
from helpers.speak import speak
from .tokens import Token, TokenType
from typing import List, Dict, Optional


class ParseError(Exception):
    """Excepción personalizada lanzada por errores durante el análisis sintáctico."""
    def __init__(self, message, position=None):
        friendly_message = f"Algo no está bien: {message}. Vamos a revisar esta parte con calma."
        super().__init__(friendly_message)
        self.position = position
        speak(friendly_message)


# Tabla de precedencia de operadores
PRECEDENCE: Dict[TokenType, int] = {
    TokenType.O: 1,
    TokenType.Y: 2,
    TokenType.NO: 3,
    TokenType.IGUAL: 4,
    TokenType.DIFERENTE: 4,
    TokenType.MENOR_QUE: 5,
    TokenType.MENOR_O_IGUAL_QUE: 5,
    TokenType.MAYOR_QUE: 5,
    TokenType.MAYOR_O_IGUAL_QUE: 5,
    TokenType.MAS: 6,
    TokenType.MENOS: 6,
    TokenType.MULTIPLICAR: 7,
    TokenType.DIVIDIR: 7,
    TokenType.MODULO: 7
}

class Parser:
    """
    Analizador sintáctico que convierte una secuencia de tokens en un AST.
    Utiliza el método de parsing de Pratt para manejar la precedencia de operadores.
    """

    def __init__(self, tokens: List[Token]):
        """
        Inicializa el parser con una lista de tokens.

        Args:
            tokens (List[Token]): Lista de tokens generados por el lexer.
        """
        self.tokens = tokens
        self.position = 0

    def parse(self) -> ProgramNode:
        """
        Punto de entrada para analizar todo el programa.

        Returns:
            ProgramNode: Nodo raíz del programa (AST).

        Raises:
            ParseError: Si no se encuentra el token 'START' o 'END'.
        """
        try:
            return self.parse_program()
        except IndexError:
            raise ParseError("El análisis se detuvo de forma inesperada. Tal vez falta algo en el programa.")

    def parse_program(self) -> ProgramNode:
        """
        Analiza el programa completo, que debe comenzar con 'START' y terminar con 'END'.

        Returns:
            ProgramNode: Nodo que representa el programa.

        Raises:
            ParseError: Si falta el token 'START' o 'END'.
        """
        self.expect(TokenType.INICIO, "El programa debe comenzar con el bloque 'INICIO'. ¡Recuerda siempre iniciar con este bloque!")
        statements = []

        while not self.match(TokenType.FIN):
            statements.append(self.parse_statement())

        self.expect(TokenType.FIN, "El programa debe terminar con el bloque 'FIN'. Asegúrate de que el bloque final esté bien colocado.")
        return ProgramNode(statements)

    def parse_statement(self) -> ASTNode:
        """
        Analiza una sentencia individual y retorna el nodo correspondiente.

        Returns:
            ASTNode: Nodo que representa la sentencia.

        Raises:
            ParseError: Si el token es inesperado en una sentencia.
        """
        token_type = self.current_token().type

        if token_type == TokenType.ASIGNAR:
            return self.parse_set_statement()
        elif token_type == TokenType.LEER:
            return self.parse_input_statement()
        elif token_type == TokenType.DECIR:
            return self.parse_print_statement()
        elif token_type == TokenType.SI:
            return self.parse_if_statement()
        elif token_type == TokenType.PARA:
            return self.parse_for_statement()
        elif token_type == TokenType.MIENTRAS:
            return self.parse_while_statement()

        raise ParseError(f"Encontré algo inesperado '{token_type.name}' en la posición {self.position}. Vamos a revisar la estructura.")

    def parse_set_statement(self) -> SetNode:
        """
        Analiza una sentencia 'SET' para asignar un valor a una variable.

        Returns:
            SetNode: Nodo que representa la sentencia 'SET'.

        Raises:
            ParseError: Si el formato de la sentencia 'SET' no es válido.
        """
        self.expect(TokenType.ASIGNAR, "¡Oh, parece que falta el 'SET' para asignar algo! Vamos a colocarlo.")
        variable = self.parse_variable()
        expression = self.parse_expression()  # Continuar con la evaluación normal
        return SetNode(variable, expression)
    
    def parse_input_statement(self) -> InputNode:
        """
        Analiza una sentencia INPUT + VAR.

        Returns:
            InputNode: Nodo que representa la instrucción INPUT.
        """
        self.expect(TokenType.LEER, "Falta la palabra 'INPUT' para recibir lo que diga. ¡Vamos a incluirla!")
        variable = self.parse_variable()
        return InputNode(variable)

    def parse_print_statement(self) -> PrintNode:
        """
        Analiza una sentencia 'PRINT' para imprimir una expresión.

        Returns:
            PrintNode: Nodo que representa la sentencia 'PRINT'.

        Raises:
            ParseError: Si falta una expresión después del token 'PRINT'.
        """
        self.expect(TokenType.DECIR, "¡Recuerda usar 'DECIR' para que se escuche lo que quieres imprimir!")
        expression = self.parse_expression()
        return PrintNode(expression)

    def parse_if_statement(self) -> IfNode:
        """
        Analiza una estructura condicional 'IF', incluyendo 'ELSE IF' y 'ELSE'.

        Returns:
            IfNode: Nodo que representa la sentencia 'IF'.

        Raises:
            ParseError: Si el formato de la estructura condicional es incorrecto.
        """
        self.expect(TokenType.SI, "Falta el bloque 'SI'. ¡Vamos a agregarlo para que podamos tomar decisiones!")
        condition = self.parse_expression()
        true_branch = self.parse_block(end_token=TokenType.FIN_SI, complement_token=TokenType.COMPLEMENTO)

        false_branch = []
        if self.current_token().type == TokenType.SINO:
            self.advance()
            false_branch = self.parse_else_block()

        self.expect(TokenType.FIN_SI, "¡No te olvides de cerrar el bloque 'FIN SI' al final de tu decisión!")
        return IfNode(condition, true_branch, false_branch)

    def parse_else_block(self) -> List[ASTNode]:
        """
        Analiza un bloque 'ELSE' o 'ELSE IF'.

        Returns:
            List[ASTNode]: Lista de nodos que representan el bloque 'ELSE' o 'ELSE IF'.

        Raises:
            ParseError: Si la estructura 'ELSE' o 'ELSE IF' es incorrecta.
        """
        if self.current_token().type in {TokenType.VARIABLE, TokenType.NUMERO}:
            condition = self.parse_expression()
            if_body = self.parse_block(end_token=TokenType.FIN_SI, complement_token=TokenType.COMPLEMENTO)

            else_body = []
            if self.current_token().type == TokenType.SINO:
                self.advance()
                else_body = self.parse_else_block()

            return [IfNode(condition, if_body, else_body)]

        return self.parse_block(end_token=TokenType.FIN_SI, complement_token=TokenType.COMPLEMENTO)

    def parse_for_statement(self) -> ForNode:
        """
        Analiza un bucle 'FOR'.

        Returns:
            ForNode: Nodo que representa el bucle 'FOR'.

        Raises:
            ParseError: Si el formato del bucle 'FOR' es incorrecto.
        """
        self.expect(TokenType.PARA, "Parece que falta el bloque 'PARA' para iniciar el bucle. ¡Vamos a incluirlo!")
        control_variable = self.parse_variable()
        start_value = self.parse_expression()
        self.expect(TokenType.HASTA, "Falta el 'HASTA' para completar la declaración del ciclo 'PARA'.")
        end_value = self.parse_expression()
        body = self.parse_block(end_token=TokenType.FIN_PARA, complement_token=TokenType.COMPLEMENTO)
        self.expect(TokenType.FIN_PARA, "¡Recuerda terminar el ciclo con 'FIN PARA'!")
        return ForNode(control_variable, start_value, end_value, body)

    def parse_while_statement(self) -> WhileNode:
        """
        Analiza un bucle 'WHILE'.

        Returns:
            WhileNode: Nodo que representa el bucle 'WHILE'.

        Raises:
            ParseError: Si el formato del bucle 'WHILE' es incorrecto.
        """
        self.expect(TokenType.MIENTRAS, "Para iniciar el ciclo falta el bloque 'MIENTRAS'. ¡No lo olvides!")
        condition = self.parse_expression()
        body = self.parse_block(end_token=TokenType.FIN_MIENTRAS, complement_token=TokenType.COMPLEMENTO)
        self.expect(TokenType.FIN_MIENTRAS, "¡Recuerda cerrar el ciclo con 'FIN MIENTRAS'!")
        return WhileNode(condition, body)

    def parse_expression(self, precedence: int = 0) -> ASTNode:
        """
        Analiza una expresión utilizando el método de parsing de Pratt.

        Args:
            precedence (int): Precedencia del operador actual.

        Returns:
            ASTNode: Nodo que representa la expresión.

        Raises:
            ParseError: Si la expresión es incorrecta o incompleta.
        """
        token = self.advance()

        # Verificar si hay secuencia de números consecutivos para formar un número completo
        if token.type == TokenType.NUMERO:
            left = self.combine_consecutive_numbers(token)
        else:
            left = self.nud(token)

        # Depuración temporal
        print(f"Parseando expresión: {token.type} con precedencia {precedence}")

        # Procesar operadores binarios (con precedencia)
        while precedence < self.get_precedence():
            token = self.advance()
            left = self.led(left, token)

        return left

    def combine_consecutive_numbers(self, token: Token) -> ASTNode:
        """
        Combina números consecutivos en un solo valor numérico.

        Args:
            token (Token): El token inicial (NUMBER).

        Returns:
            NumberNode: Nodo que representa el número combinado.
        """
        number_value = str(token.value)
        
        # Combinar números consecutivos
        while self.match(TokenType.NUMERO):
            number_token = self.advance()
            number_value += str(number_token.value)
        
        return NumberNode(int(number_value))

    def nud(self, token: Token) -> ASTNode:
        """
        Maneja los casos donde no hay un operando a la izquierda.

        Args:
            token (Token): Token actual que se está analizando.

        Returns:
            ASTNode: Nodo que representa el operando o expresión unaria.

        Raises:
            ParseError: Si el token es inesperado en el contexto actual.
        """
        if token.type == TokenType.NUMERO:
            return NumberNode(token.value)
        elif token.type == TokenType.VARIABLE:
            return VariableNode(token.value)
        elif token.type == TokenType.PARENTESIS_IZQUIERDO:
            expr = self.parse_expression()
            self.expect(TokenType.PARENTESIS_DERECHO, "Falta cerrar la expresión con un paréntesis de cierre ')'. ¡No olvides cerrarlo!")
            return expr
        elif token.type in {TokenType.NO, TokenType.MENOS}:
            operand = self.parse_expression(PRECEDENCE[token.type])
            return UnaryOperationNode(token.type, operand)

        raise ParseError(f"Encontré algo inesperado en '{token.type.name}'. ¡Vamos a verificar esta expresión!")

    def led(self, left: ASTNode, token: Token) -> ASTNode:
        """
        Maneja los operadores binarios que tienen un operando a la izquierda.

        Args:
            left (ASTNode): Nodo que representa el operando izquierdo.
            token (Token): Token que representa el operador.

        Returns:
            ASTNode: Nodo que representa la operación binaria.

        Raises:
            ParseError: Si el operador es desconocido o la operación es inválida.
        """
        # Revisar si el token es un operador binario válido
        valid_operators = {
            TokenType.MAS, TokenType.MENOS, TokenType.MULTIPLICAR, TokenType.DIVIDIR,
            TokenType.IGUAL, TokenType.DIFERENTE, TokenType.MENOR_QUE, TokenType.MAYOR_QUE,
            TokenType.MENOR_O_IGUAL_QUE, TokenType.MAYOR_O_IGUAL_QUE, TokenType.Y, TokenType.O, TokenType.MODULO
        }
        
        if token.type not in valid_operators:
            raise ParseError(f"Encontré algo inesperado '{token.type.name}' aquí. ¡Vamos a revisar la expresión!")

        precedence = PRECEDENCE[token.type]
        
        # Obtener el operando derecho
        right = self.parse_expression(precedence)
        
        # Validar que no se esté dividiendo por cero
        if token.type == TokenType.DIVIDIR and isinstance(right, NumberNode) and right.value == 0:
            raise ParseError("División por cero no permitida. No puedes dividir entre cero.")

        # Retornar el nodo de operación binaria
        return BinaryOperationNode(token.type, left, right)

    def get_precedence(self) -> int:
        """
        Obtiene la precedencia del token actual.

        Returns:
            int: Precedencia del operador actual.
        """
        token_type = self.current_token().type
        return PRECEDENCE.get(token_type, 0)

    def parse_variable(self) -> VariableNode:
        """
        Analiza una variable.

        Returns:
            VariableNode: Nodo que representa la variable.

        Raises:
            ParseError: Si el token actual no es una variable válida.
        """
        token = self.expect(TokenType.VARIABLE, "Se esperaba una variable. ¡Vamos a poner un nombre válido!")
        return VariableNode(token.value)

    def parse_block(self, end_token: TokenType, complement_token: Optional[TokenType] = None) -> List[ASTNode]:
        """
        Analiza un bloque de sentencias.

        Args:
            end_token (TokenType): Token que indica el final del bloque.
            complement_token (Optional[TokenType], opcional): Token adicional que puede ser ignorado en el bloque.

        Returns:
            List[ASTNode]: Lista de nodos que representan el bloque.

        Raises:
            ParseError: Si no se puede cerrar el bloque correctamente.
        """
        block = []
        while self.current_token().type != end_token:
            if complement_token and self.current_token().type == complement_token:
                self.advance()
                continue
            if self.current_token().type == TokenType.SINO and end_token == TokenType.FIN_SI:
                break
            block.append(self.parse_statement())
        return block

    def expect(self, token_type: TokenType, error_message: str) -> Token:
        """
        Verifica que el token actual sea del tipo esperado y avanza.

        Args:
            token_type (TokenType): Tipo de token esperado.
            error_message (str): Mensaje de error personalizado en caso de que el token no coincida.

        Returns:
            Token: El token actual.

        Raises:
            ParseError: Si el token actual no coincide con el tipo esperado.
        """
        token = self.current_token()
        if token.type != token_type:
            raise ParseError(f"{error_message}. Se encontró '{token.type.name}' en la posición {self.position}.")
        self.advance()
        return token

    def match(self, *token_types: TokenType) -> bool:
        """
        Verifica si el token actual coincide con alguno de los tipos dados.

        Args:
            token_types (TokenType): Tipos de tokens a verificar.

        Returns:
            bool: True si coincide con alguno de los tipos, False en caso contrario.
        """
        return self.current_token().type in token_types

    def current_token(self) -> Token:
        """
        Retorna el token actual sin avanzar.

        Returns:
            Token: El token actual.
        """
        if self.position >= len(self.tokens):
            return None
        return self.tokens[self.position]

    def advance(self) -> Token:
        """
        Avanza al siguiente token y retorna el anterior.

        Returns:
            Token: El token actual antes de avanzar.
        """
        if self.position < len(self.tokens):
            current = self.tokens[self.position]
            self.position += 1
            return current
