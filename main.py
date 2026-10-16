"""DCOMP320 - Stacks Implementation.

Reads mathematical expressions from input.txt, evaluates each one using a
stack-based algorithm, and writes the results to output.txt.

Run from the project directory with:  python main.py
"""

import math
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
INPUT_FILE = PROJECT_DIR / "input.txt"
OUTPUT_FILE = PROJECT_DIR / "output.txt"

# Higher number = evaluated first. "neg" is the unary minus (e.g. -3).
PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "neg": 3}
BINARY_OPERATORS = "+-*/"
MIN_SEPARATOR_LENGTH = 3


class EmptyStackError(Exception):
    """Raised when pop() or peek() is used on an empty stack."""


class ExpressionError(Exception):
    """Raised when an expression is invalid or cannot be calculated."""


class Stack:
    """A Last-In-First-Out (LIFO) collection."""

    def __init__(self):
        self._items = []

    def push(self, item):
        """Put an item on top of the stack."""
        self._items.append(item)

    def pop(self):
        """Remove and return the top item."""
        if self.is_empty():
            raise EmptyStackError("Cannot pop from an empty stack")
        return self._items.pop()

    def peek(self):
        """Return the top item without removing it."""
        if self.is_empty():
            raise EmptyStackError("Cannot peek at an empty stack")
        return self._items[-1]

    def is_empty(self):
        return len(self._items) == 0

    def size(self):
        return len(self._items)


def tokenize(expression):
    """Split an expression string into numbers, operators and parentheses.

    Numbers become floats; everything else stays a one-character string.
    Whitespace is skipped. Any other character is an error.
    """
    tokens = []
    position = 0
    while position < len(expression):
        char = expression[position]
        if char.isspace():
            position += 1
        elif char in BINARY_OPERATORS or char in "()":
            tokens.append(char)
            position += 1
        elif char.isdigit() or char == ".":
            start = position
            while position < len(expression) and (
                expression[position].isdigit() or expression[position] == "."
            ):
                position += 1
            tokens.append(parse_number(expression[start:position]))
        else:
            raise ExpressionError(f"Invalid character '{char}'")
    return tokens


def parse_number(text):
    """Convert text such as '12' or '3.75' to a float, or raise an error."""
    if text.count(".") > 1 or text == ".":
        raise ExpressionError(f"Invalid number '{text}'")
    return float(text)


def apply_top_operator(operands, operators):
    """Pop one operator and its operand(s), calculate, push the result."""
    operator = operators.pop()

    if operator == "neg":
        operands.push(-operands.pop())
        return

    right = operands.pop()  # the right-hand number is on top of the stack
    left = operands.pop()
    if operator == "+":
        result = left + right
    elif operator == "-":
        result = left - right
    elif operator == "*":
        result = left * right
    else:
        if right == 0:
            raise ExpressionError("Division by zero")
        result = left / right

    if not math.isfinite(result):
        raise ExpressionError("Result is too large to calculate")
    operands.push(result)


def evaluate_expression(expression):
    """Evaluate an expression using an operand stack and an operator stack.

    Returns a float. Raises ExpressionError if the expression is invalid.
    """
    tokens = tokenize(expression)
    if not tokens:
        raise ExpressionError("Empty expression")

    operands = Stack()
    operators = Stack()
    expect_operand = True  # True when a number, '(' or unary '-' must come next

    for token in tokens:
        if isinstance(token, float):
            if not expect_operand:
                raise ExpressionError("Missing operator before number")
            operands.push(token)
            expect_operand = False

        elif token == "(":
            if not expect_operand:
                raise ExpressionError("Missing operator before '('")
            operators.push(token)

        elif token == ")":
            if expect_operand:
                raise ExpressionError("Missing operand before ')'")
            while not operators.is_empty() and operators.peek() != "(":
                apply_top_operator(operands, operators)
            if operators.is_empty():
                raise ExpressionError("Unmatched ')'")
            operators.pop()  # discard the matching '('
            expect_operand = False

        elif expect_operand:
            # An operator where a number should be: only '-' is allowed (unary).
            if token != "-":
                raise ExpressionError(f"Operator '{token}' is missing a left operand")
            operators.push("neg")

        else:
            # Binary operator: first finish any earlier operator of equal or
            # higher precedence (this gives left-to-right associativity).
            while (
                not operators.is_empty()
                and operators.peek() != "("
                and PRECEDENCE[operators.peek()] >= PRECEDENCE[token]
            ):
                apply_top_operator(operands, operators)
            operators.push(token)
            expect_operand = True

    if expect_operand:
        raise ExpressionError("Expression ends with an operator")

    while not operators.is_empty():
        if operators.peek() == "(":
            raise ExpressionError("Unmatched '('")
        apply_top_operator(operands, operators)

    return operands.pop()


def format_result(value):
    """Show whole numbers without '.0' and hide tiny floating-point noise."""
    value = round(value, 10)
    if value == int(value):
        return str(int(value))
    return str(value)


def is_separator(line):
    """A separator line contains only dashes (at least three), e.g. -----."""
    text = line.strip()
    return len(text) >= MIN_SEPARATOR_LENGTH and set(text) == {"-"}


def process_line(line):
    """Return the output line for one input line, and whether it was an error."""
    text = line.strip()
    if text == "":
        return "", False
    if is_separator(text):
        return text, False
    try:
        return format_result(evaluate_expression(text)), False
    except ExpressionError as error:
        return f"Error: {error}", True


def process_file(input_path, output_path):
    """Evaluate every expression in input_path and write output_path.

    Raises FileNotFoundError if the input file does not exist.
    Returns (expressions_evaluated, errors_found).
    """
    with open(input_path, "r", encoding="utf-8-sig") as input_file:
        input_lines = input_file.read().splitlines()

    output_lines = []
    expression_count = 0
    error_count = 0
    for line in input_lines:
        output_line, failed = process_line(line)
        output_lines.append(output_line)
        if line.strip() and not is_separator(line):
            expression_count += 1
        if failed:
            error_count += 1

    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(output_lines) + "\n")

    return expression_count, error_count


def main():
    try:
        expression_count, error_count = process_file(INPUT_FILE, OUTPUT_FILE)
    except FileNotFoundError:
        print(f"Error: input file not found: {INPUT_FILE}")
        return 1
    except (OSError, UnicodeDecodeError) as error:
        print(f"Error: could not read or write the files: {error}")
        return 1

    print(f"Processed {expression_count} expression(s) with {error_count} error(s).")
    print(f"Results written to {OUTPUT_FILE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
