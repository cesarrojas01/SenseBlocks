from enum import Enum, auto
from dataclasses import dataclass
from typing import Any

class TokenType(Enum):
    """
    Enumeración de los tipos de tokens utilizados por el lexer y el parser.
    """

    # Start and EOF
    INICIO = auto()
    FIN = auto()

    # Control Structures
    SI = auto()
    SINO = auto()
    FIN_SI = auto()
    PARA = auto()
    FIN_PARA = auto()
    HASTA = auto()
    MIENTRAS = auto()
    FIN_MIENTRAS = auto()
    COMPLEMENTO = auto()

    # Operators
    MAS = auto()
    MENOS = auto()
    MULTIPLICAR = auto()
    DIVIDIR = auto()
    MODULO = auto()

    # Comparison Operators
    IGUAL = auto()
    MAYOR_QUE = auto()
    MENOR_QUE = auto()
    MAYOR_O_IGUAL_QUE = auto()
    MENOR_O_IGUAL_QUE = auto()
    DIFERENTE = auto()

    # Logical Operators
    Y = auto()
    O = auto()
    NO = auto()

    # Parentheses
    PARENTESIS_IZQUIERDO = auto()
    PARENTESIS_DERECHO = auto()

    # Literals and Identifiers
    NUMERO = auto()
    VARIABLE = auto()

    # Keywords
    DECIR = auto()
    LEER = auto()
    ASIGNAR = auto()

    # Miscellaneous
    DESCONOCIDO = auto()


@dataclass
class Token:
    """
    Representa un token generado por el lexer.
    Contiene el tipo de token y su valor asociado.
    """
    type: TokenType
    value: Any = None

    def __post_init__(self):
        """
        Establece el valor por defecto si no se proporciona uno.
        """
        if self.value is None:
            self.value = self.type.name

    def __repr__(self):
        """
        Retorna una representación en cadena del token en formato (Tipo, Valor).
        """
        return f"({self.type.name}, {self.value})"
