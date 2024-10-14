from dataclasses import dataclass, field
from typing import List, Optional, Any
import json
import yaml


class ASTNode:
    """
    Clase base para todos los nodos del AST (Árbol de Sintaxis Abstracta).
    """

    def to_dict(self) -> dict:
        """
        Convierte el nodo AST a un diccionario.

        Returns:
            dict: Representación del nodo en formato diccionario.
        """
        raise NotImplementedError("Debe implementarse en subclases")


@dataclass
class ProgramNode(ASTNode):
    """
    Nodo que representa un programa completo.
    Contiene una lista de sentencias.
    """
    body: List[ASTNode]

    def to_dict(self) -> dict:
        return {
            "type": "Program",
            "body": [stmt.to_dict() for stmt in self.body]
        }


@dataclass
class VariableNode(ASTNode):
    """
    Nodo que representa una variable.
    """
    name: str

    def to_dict(self) -> dict:
        return {
            "type": "Variable",
            "name": self.name
        }


@dataclass
class NumberNode(ASTNode):
    """
    Nodo que representa un número literal.
    """
    value: int

    def to_dict(self) -> dict:
        return {
            "type": "Literal",
            "value": self.value
        }


@dataclass
class InputNode(ASTNode):
    """
    Nodo que representa la entrada de voz para asignar un valor a una variable.
    """
    variable: VariableNode

    def to_dict(self) -> dict:
        return {
            "type": "InputStatement",
            "variable": self.variable.to_dict()
        }


@dataclass
class BinaryOperationNode(ASTNode):
    """
    Nodo que representa una operación binaria.
    """
    operator: Any  # Debería ser TokenType
    left: ASTNode
    right: ASTNode

    def to_dict(self) -> dict:
        return {
            "type": "BinaryExpression",
            "operator": self.operator.name,
            "left": self.left.to_dict(),
            "right": self.right.to_dict()
        }


@dataclass
class UnaryOperationNode(ASTNode):
    """
    Nodo que representa una operación unaria.
    """
    operator: Any  # Debería ser TokenType
    operand: ASTNode

    def to_dict(self) -> dict:
        return {
            "type": "UnaryExpression",
            "operator": self.operator.name,
            "operand": self.operand.to_dict()
        }


@dataclass
class SetNode(ASTNode):
    """
    Nodo que representa una asignación de valor a una variable.
    """
    variable: VariableNode
    expression: ASTNode

    def to_dict(self) -> dict:
        return {
            "type": "Assignment",
            "variable": self.variable.to_dict(),
            "expression": self.expression.to_dict()
        }


@dataclass
class PrintNode(ASTNode):
    """
    Nodo que representa una instrucción de impresión.
    """
    expression: ASTNode

    def to_dict(self) -> dict:
        return {
            "type": "PrintStatement",
            "expression": self.expression.to_dict()
        }


@dataclass
class IfNode(ASTNode):
    """
    Nodo que representa una estructura condicional 'IF'.
    """
    condition: ASTNode
    true_branch: List[ASTNode]
    false_branch: Optional[List[ASTNode]] = field(default=None)

    def to_dict(self) -> dict:
        return {
            "type": "IfStatement",
            "condition": self.condition.to_dict(),
            "true_branch": [stmt.to_dict() for stmt in self.true_branch],
            "false_branch": [stmt.to_dict() for stmt in self.false_branch] if self.false_branch else None
        }


@dataclass
class ForNode(ASTNode):
    """
    Nodo que representa un bucle 'FOR'.
    """
    control_variable: VariableNode
    start_value: ASTNode
    end_value: ASTNode
    body: List[ASTNode]

    def to_dict(self) -> dict:
        return {
            "type": "ForStatement",
            "control_variable": self.control_variable.to_dict(),
            "start_value": self.start_value.to_dict(),
            "end_value": self.end_value.to_dict(),
            "body": [stmt.to_dict() for stmt in self.body]
        }


@dataclass
class WhileNode(ASTNode):
    """
    Nodo que representa un bucle 'WHILE'.
    """
    condition: ASTNode
    body: List[ASTNode]

    def to_dict(self) -> dict:
        return {
            "type": "WhileStatement",
            "condition": self.condition.to_dict(),
            "body": [stmt.to_dict() for stmt in self.body]
        }


def ast_to_json(ast: ASTNode) -> str:
    """
    Convierte el AST a una cadena JSON.

    Args:
        ast (ASTNode): Nodo raíz del AST.

    Returns:
        str: Representación del AST en formato JSON.
    """
    return json.dumps(ast.to_dict(), indent=2)


def ast_to_yaml(ast: ASTNode) -> str:
    """
    Convierte el AST a una cadena YAML.

    Args:
        ast (ASTNode): Nodo raíz del AST.

    Returns:
        str: Representación del AST en formato YAML.
    """
    return yaml.dump(ast.to_dict(), default_flow_style=False, sort_keys=False, indent=4)
