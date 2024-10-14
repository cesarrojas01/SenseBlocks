from enum import Enum, auto
from dataclasses import dataclass
from typing import Any

class TokenType(Enum):
    """
    Enumeración de los tipos de tokens utilizados por el lexer y el parser.
    """

    # Start and EOF
    START = auto()
    END = auto()

    # Control Structures
    IF_START = auto()
    IF_ELSE = auto()
    IF_END = auto()
    FOR_START = auto()
    FOR_END = auto()
    TO = auto()
    WHILE_START = auto()
    WHILE_END = auto()
    COMPLEMENT = auto()

    # Operators
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MODULO = auto()

    # Comparison Operators
    EQUAL = auto()
    GREATER = auto()
    LESS = auto()
    GREATER_EQUAL = auto()
    LESS_EQUAL = auto()
    DIFFERENT = auto()

    # Logical Operators
    AND = auto()
    OR = auto()
    NOT = auto()

    # Parentheses
    LPAREN = auto()
    RPAREN = auto()

    # Literals and Identifiers
    NUMBER = auto()
    IDENTIFIER = auto()

    # Keywords
    SAY = auto()
    INPUT = auto()
    SET = auto()

    # Miscellaneous
    UNKNOWN = auto()


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
