from typing import Any, Dict, List
from .ast_node import (
    ASTNode, ProgramNode, SetNode, PrintNode, IfNode, ForNode, WhileNode,
    VariableNode, NumberNode, InputNode, BinaryOperationNode, UnaryOperationNode
)
from .tokens import TokenType
from helpers import speak, recognize_input
import operator as op

# Excepciones con retroalimentación por voz
class EvaluationError(Exception):
    """Clase base para excepciones de evaluación."""
    def __init__(self, message: str):
        super().__init__(message)
        speak.speak(message)

class UndefinedVariableError(EvaluationError):
    """Excepción lanzada cuando se intenta acceder a una variable no definida."""
    def __init__(self, variable_name):
        message = f"La variable '{variable_name}' no está definida. Asegúrate de que hayas guardado un número en esa variable antes de usarla."
        super().__init__(message)


class DivisionByZeroError(EvaluationError):
    """Excepción lanzada cuando se intenta dividir por cero."""
    def __init__(self):
        message = "No puedes dividir por cero. Prueba con otro número para que la operación funcione."
        super().__init__(message)

class InvalidOperationError(EvaluationError):
    """Excepción lanzada cuando se utiliza un operador desconocido."""
    def __init__(self, operation):
        message = f"La operación '{operation}' no es válida. Revisa si los bloques que usaste son los correctos."
        super().__init__(message)


class TypeMismatchError(EvaluationError):
    """Excepción lanzada cuando hay un desajuste de tipos en una operación."""
    def __init__(self, expected_type, received_type):
        message = f"Esperaba un {expected_type}, pero recibí un {received_type}. Usa el bloque correcto para esta operación."
        super().__init__(message)


class Evaluator:
    """Evaluador que ejecuta el AST generado por el parser."""

    def __init__(self):
        """
        Inicializa el evaluador con un entorno de variables vacío.
        """
        self.variables: Dict[str, Any] = {}

    def evaluate(self, node: ASTNode) -> Any:
        """
        Evalúa el nodo raíz del AST.

        Args:
            node (ASTNode): Nodo a evaluar.

        Returns:
            Any: Resultado de la evaluación del nodo.
        """
        method_name = f'eval_{type(node).__name__}'
        method = getattr(self, method_name, self.generic_eval)
        return method(node)

    def eval_ProgramNode(self, node: ProgramNode) -> Any:
        """
        Evalúa un nodo de programa ejecutando cada sentencia en su cuerpo.

        Args:
            node (ProgramNode): Nodo del programa a evaluar.

        Returns:
            Any: Resultado de la evaluación del programa.
        """
        result = None
        for stmt in node.body:
            result = self.evaluate(stmt)
        return result

    def eval_SetNode(self, node: SetNode) -> Any:
        """
        Evalúa una asignación de valor a una variable.

        Args:
            node (SetNode): Nodo de asignación.

        Returns:
            Any: Valor asignado a la variable.
        """
        value = self.evaluate(node.expression)
        self.variables[node.variable.name] = value
        return value

    def eval_InputNode(self, node: InputNode) -> Any:
        """
        Evalúa una instrucción de entrada de voz y asigna el valor a la variable.

        Args:
            node (InputNode): Nodo de Input.
        """
        variable_name = node.variable.name
        valid_input = False  # Bandera para controlar la validez de la entrada

        while not valid_input:
            try:
                prompt_message = f"Por favor, di el valor para la variable {variable_name}"
                speak.speak(prompt_message)
                # Aquí se espera que `recognize_input()` sea una función que capture la entrada por voz
                value = recognize_input.recognize_and_process_input()  # Función que usará el reconocimiento de voz
                
                # Validar el valor recibido
                if value is not None and value != "":
                    self.variables[variable_name] = value
                    message = f"Se asignó el valor {value} a la variable {variable_name} mediante entrada de voz."
                    print(message)
                    speak.speak(message)
                    valid_input = True  # Si la entrada es válida, salimos del bucle
                else:
                    raise ValueError("No se recibió un valor válido. Inténtalo de nuevo.")
            
            except Exception as e:
                error_message = f"Error en la entrada de voz: {str(e)}. Vuelve a intentarlo."
                print(error_message)
                speak.speak(error_message)

    def eval_PrintNode(self, node: PrintNode) -> None:
        """
        Evalúa una instrucción de impresión.

        Args:
            node (PrintNode): Nodo de impresión.
        """
        value = self.evaluate(node.expression)

        if isinstance(node.expression, VariableNode):
            variable_name = node.expression.name
            message = f"El valor de la variable {variable_name} es: {value}"
        else:
            message = f"El resultado de la expresión es: {value}"

        print(message)
        speak.speak(message)

    def eval_IfNode(self, node: IfNode) -> Any:
        """
        Evalúa una estructura condicional 'IF'.

        Args:
            node (IfNode): Nodo de condicional 'IF'.

        Returns:
            Any: Resultado de la evaluación de la rama correspondiente.
        """
        condition = self.evaluate(node.condition)
        if condition:
            return self.evaluate_block(node.true_branch)
        elif node.false_branch:
            return self.evaluate_block(node.false_branch)
        return None

    def eval_ForNode(self, node: ForNode) -> Any:
        """
        Evalúa un bucle 'FOR'.

        Args:
            node (ForNode): Nodo de bucle 'FOR'.

        Returns:
            Any: Resultado de la evaluación del bucle.
        """
        control_variable = node.control_variable.name
        start_value = self.evaluate(node.start_value)
        end_value = self.evaluate(node.end_value)
        self.variables[control_variable] = start_value
        result = None

        for i in range(start_value, end_value + 1):
            self.variables[control_variable] = i
            result = self.evaluate_block(node.body)

        return result

    def eval_WhileNode(self, node: WhileNode) -> Any:
        """
        Evalúa un bucle 'WHILE'.

        Args:
            node (WhileNode): Nodo de bucle 'WHILE'.

        Returns:
            Any: Resultado de la evaluación del bucle.
        """
        result = None
        while self.evaluate(node.condition):
            result = self.evaluate_block(node.body)
        return result

    def eval_BinaryOperationNode(self, node: BinaryOperationNode) -> Any:
        """
        Evalúa una operación binaria.

        Args:
            node (BinaryOperationNode): Nodo de operación binaria.

        Returns:
            Any: Resultado de la operación.
        """
        left_value = self.evaluate(node.left)
        right_value = self.evaluate(node.right)
        return self.apply_operator(node.operator, left_value, right_value)

    def eval_UnaryOperationNode(self, node: UnaryOperationNode) -> Any:
        """
        Evalúa una operación unaria.

        Args:
            node (UnaryOperationNode): Nodo de operación unaria.

        Returns:
            Any: Resultado de la operación unaria.
        """
        operand_value = self.evaluate(node.operand)

        if node.operator == TokenType.NO:
            return int(not operand_value)
        elif node.operator == TokenType.MENOS:
            return -operand_value
        else:
            raise InvalidOperationError(f"Operador unario desconocido: {node.operator.name}")

    def eval_NumberNode(self, node: NumberNode) -> int:
        """
        Evalúa un número literal.

        Args:
            node (NumberNode): Nodo de número literal.

        Returns:
            int: Valor numérico del nodo.
        """
        return node.value

    def eval_VariableNode(self, node: VariableNode) -> Any:
        """
        Evalúa una variable.

        Args:
            node (VariableNode): Nodo de variable.

        Returns:
            Any: Valor de la variable.

        Raises:
            UndefinedVariableError: Si la variable no ha sido definida.
        """
        if node.name not in self.variables:
            raise UndefinedVariableError(f"La variable '{node.name}' no está definida.")
        return self.variables[node.name]

    def apply_operator(self, operator: TokenType, left: Any, right: Any) -> Any:
        """
        Aplica un operador binario a los operandos.

        Args:
            operator (TokenType): Tipo de operador.
            left (Any): Operando izquierdo.
            right (Any): Operando derecho.

        Returns:
            Any: Resultado de la operación.

        Raises:
            TypeMismatchError: Si los operandos no son compatibles para la operación.
            InvalidOperationError: Si se utiliza un operador desconocido.
        """
        # Operadores numéricos
        numeric_operators = {
            TokenType.MAS: op.add,
            TokenType.MENOS: op.sub,
            TokenType.MULTIPLICAR: op.mul,
            TokenType.DIVIDIR: self.safe_divide,
            TokenType.MODULO: op.mod,
            TokenType.MENOR_QUE: lambda l, r: int(l < r),
            TokenType.MENOR_O_IGUAL_QUE: lambda l, r: int(l <= r),
            TokenType.MAYOR_QUE: lambda l, r: int(l > r),
            TokenType.MAYOR_O_IGUAL_QUE: lambda l, r: int(l >= r),
            TokenType.IGUAL: lambda l, r: int(l == r),
            TokenType.DIFERENTE: lambda l, r: int(l != r),
            TokenType.Y: lambda l, r: int(bool(l) and bool(r)),
            TokenType.O: lambda l, r: int(bool(l) or bool(r)),
        }

        string_operators = {
            TokenType.MAS: lambda l, r: l + r,
            TokenType.MENOS: lambda l, r: l.replace(r, ""),
            TokenType.MULTIPLICAR: lambda l, r: l * r if isinstance(r, int) else l,
            # Otros operadores para cadenas pueden agregarse aquí
        }

        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            func = numeric_operators.get(operator)
            if not func:
                raise InvalidOperationError(f"Operador desconocido: {operator.name}")
            return func(left, right)
        elif isinstance(left, str) or isinstance(right, str):
            func = string_operators.get(operator)
            if not func:
                raise InvalidOperationError(f"Operador no soportado para cadenas: {operator.name}")
            return func(left, right)
        else:
            raise TypeMismatchError("Los operandos deben ser del mismo tipo y compatibles para la operación.")

    def safe_divide(self, left: Any, right: Any) -> float:
        """
        Realiza una división segura, manejando la división por cero.

        Args:
            left (Any): Operando izquierdo.
            right (Any): Operando derecho.

        Returns:
            float: Resultado de la división.

        Raises:
            DivisionByZeroError: Si se intenta dividir por cero.
        """
        if right == 0:
            raise DivisionByZeroError("División por cero no permitida.")
        return left / right

    def evaluate_block(self, block: List[ASTNode]) -> Any:
        """
        Evalúa un bloque de sentencias.

        Args:
            block (List[ASTNode]): Lista de nodos que representan las sentencias.

        Returns:
            Any: Resultado de la evaluación del bloque.
        """
        result = None
        for stmt in block:
            result = self.evaluate(stmt)
        return result

    def generic_eval(self, node: ASTNode) -> Any:
        """
        Método genérico para evaluar un nodo no reconocido.

        Args:
            node (ASTNode): Nodo a evaluar.

        Raises:
            InvalidOperationError: Si el nodo no es reconocido.
        """
        raise InvalidOperationError(f"Nodo AST desconocido: {type(node).__name__}")
