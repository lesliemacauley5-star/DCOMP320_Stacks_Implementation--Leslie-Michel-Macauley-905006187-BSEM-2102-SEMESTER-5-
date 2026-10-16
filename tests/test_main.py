"""Automated tests for main.py. Run from the project directory with:

    python -m unittest discover -s tests -v
"""

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main
from main import (
    EmptyStackError,
    ExpressionError,
    Stack,
    evaluate_expression,
    format_result,
    is_separator,
    process_file,
)


class TestStack(unittest.TestCase):
    def test_push_adds_item_on_top(self):
        stack = Stack()
        stack.push(1)
        stack.push(2)
        self.assertEqual(stack.peek(), 2)
        self.assertEqual(stack.size(), 2)

    def test_pop_returns_items_in_lifo_order(self):
        stack = Stack()
        for item in (1, 2, 3):
            stack.push(item)
        self.assertEqual(stack.pop(), 3)
        self.assertEqual(stack.pop(), 2)
        self.assertEqual(stack.pop(), 1)

    def test_peek_does_not_remove_item(self):
        stack = Stack()
        stack.push("a")
        self.assertEqual(stack.peek(), "a")
        self.assertEqual(stack.size(), 1)

    def test_new_stack_is_empty(self):
        stack = Stack()
        self.assertTrue(stack.is_empty())
        self.assertEqual(stack.size(), 0)

    def test_pop_on_empty_stack_raises_error(self):
        with self.assertRaises(EmptyStackError):
            Stack().pop()

    def test_peek_on_empty_stack_raises_error(self):
        with self.assertRaises(EmptyStackError):
            Stack().peek()

    def test_stack_is_empty_again_after_popping_everything(self):
        stack = Stack()
        stack.push(5)
        stack.pop()
        self.assertTrue(stack.is_empty())


class TestEvaluation(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(evaluate_expression("2 + 3"), 5)

    def test_subtraction(self):
        self.assertEqual(evaluate_expression("10 - 4"), 6)

    def test_multiplication(self):
        self.assertEqual(evaluate_expression("6 * 7"), 42)

    def test_division(self):
        self.assertEqual(evaluate_expression("20 / 4"), 5)

    def test_precedence_multiplication_before_addition(self):
        self.assertEqual(evaluate_expression("3 + 5 * 2"), 13)

    def test_precedence_division_before_subtraction(self):
        self.assertEqual(evaluate_expression("10 - 8 / 4"), 8)

    def test_parentheses_change_order(self):
        self.assertEqual(evaluate_expression("(3 + 5) * 2"), 16)

    def test_nested_parentheses(self):
        self.assertEqual(evaluate_expression("((2 + 3) * (4 - 1)) / 5"), 3)

    def test_subtraction_is_left_to_right(self):
        self.assertEqual(evaluate_expression("10 - 3 - 2"), 5)

    def test_division_is_left_to_right(self):
        self.assertEqual(evaluate_expression("100 / 10 / 5"), 2)

    def test_multi_digit_numbers(self):
        self.assertEqual(evaluate_expression("120 + 345"), 465)

    def test_decimal_numbers(self):
        self.assertAlmostEqual(evaluate_expression("1.5 + 2.25"), 3.75)

    def test_no_spaces_needed(self):
        self.assertEqual(evaluate_expression("3+5*2"), 13)

    def test_extra_whitespace_is_ignored(self):
        self.assertEqual(evaluate_expression("   3   +\t5 * 2  "), 13)

    def test_single_number(self):
        self.assertEqual(evaluate_expression("42"), 42)

    def test_unary_minus_at_start(self):
        self.assertEqual(evaluate_expression("-3 + 5"), 2)

    def test_unary_minus_inside_parentheses(self):
        self.assertEqual(evaluate_expression("2 * (-4)"), -8)

    def test_unary_minus_after_operator(self):
        self.assertEqual(evaluate_expression("2 * -3"), -6)

    def test_assignment_sample_expressions(self):
        self.assertEqual(evaluate_expression("3 + 5 * 2"), 13)
        self.assertEqual(evaluate_expression("(8 / 4) + 7 * 2"), 16)
        self.assertEqual(evaluate_expression("10 - (2 + 3) * 4"), -10)


class TestInvalidExpressions(unittest.TestCase):
    def assert_invalid(self, expression, message_part=None):
        with self.assertRaises(ExpressionError) as context:
            evaluate_expression(expression)
        if message_part:
            self.assertIn(message_part, str(context.exception))

    def test_division_by_zero(self):
        self.assert_invalid("5 / 0", "Division by zero")

    def test_division_by_zero_from_subexpression(self):
        self.assert_invalid("8 / (4 - 4)", "Division by zero")

    def test_missing_closing_parenthesis(self):
        self.assert_invalid("(2 + 3", "Unmatched '('")

    def test_missing_opening_parenthesis(self):
        self.assert_invalid("2 + 3)", "Unmatched ')'")

    def test_empty_parentheses(self):
        self.assert_invalid("()")

    def test_expression_ending_with_operator(self):
        self.assert_invalid("2 +", "ends with an operator")

    def test_expression_starting_with_binary_operator(self):
        self.assert_invalid("* 2 + 3")

    def test_two_operators_in_a_row(self):
        self.assert_invalid("2 + * 3")

    def test_two_numbers_without_operator(self):
        self.assert_invalid("2 3", "Missing operator")

    def test_number_directly_before_parenthesis(self):
        self.assert_invalid("2 (3 + 4)")

    def test_invalid_character(self):
        self.assert_invalid("2 + a", "Invalid character")

    def test_malformed_decimal_number(self):
        self.assert_invalid("1..2 + 3", "Invalid number")

    def test_lone_decimal_point(self):
        self.assert_invalid(". + 3", "Invalid number")

    def test_empty_expression(self):
        self.assert_invalid("   ", "Empty expression")


class TestHelpers(unittest.TestCase):
    def test_separator_is_recognised(self):
        self.assertTrue(is_separator("-----"))
        self.assertTrue(is_separator("  ---  "))

    def test_expressions_are_not_separators(self):
        self.assertFalse(is_separator("3 - 2"))
        self.assertFalse(is_separator("-5"))
        self.assertFalse(is_separator("--"))
        self.assertFalse(is_separator(""))

    def test_format_whole_number_has_no_decimal_point(self):
        self.assertEqual(format_result(13.0), "13")
        self.assertEqual(format_result(-10.0), "-10")

    def test_format_decimal_number(self):
        self.assertEqual(format_result(3.75), "3.75")

    def test_format_hides_floating_point_noise(self):
        self.assertEqual(format_result(0.1 + 0.2), "0.3")


class TestFileProcessing(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.input_path = Path(self.temp_dir.name) / "input.txt"
        self.output_path = Path(self.temp_dir.name) / "output.txt"

    def run_with_input(self, text):
        self.input_path.write_text(text, encoding="utf-8")
        counts = process_file(self.input_path, self.output_path)
        return self.output_path.read_text(encoding="utf-8"), counts

    def test_sample_input_produces_expected_output(self):
        text = "3 + 5 * 2\n-----\n(8 / 4) + 7 * 2\n-----\n10 - (2 + 3) * 4\n"
        output, counts = self.run_with_input(text)
        self.assertEqual(output, "13\n-----\n16\n-----\n-10\n")
        self.assertEqual(counts, (3, 0))

    def test_output_file_is_created(self):
        self.run_with_input("1 + 1\n")
        self.assertTrue(self.output_path.exists())

    def test_multiple_expressions_in_one_group(self):
        output, _ = self.run_with_input("1 + 1\n2 * 3\n9 / 3\n")
        self.assertEqual(output, "2\n6\n3\n")

    def test_each_result_is_on_its_own_line(self):
        output, _ = self.run_with_input("1 + 1\n-----\n2 + 2\n")
        self.assertEqual(output.splitlines(), ["2", "-----", "4"])

    def test_separator_lines_keep_their_text_and_position(self):
        output, _ = self.run_with_input("1 + 1\n-------\n2 + 2\n---\n3 + 3\n")
        self.assertEqual(output.splitlines(), ["2", "-------", "4", "---", "6"])

    def test_order_of_input_is_preserved(self):
        output, _ = self.run_with_input("5 - 1\n-----\n1 + 1\n-----\n9 * 9\n")
        self.assertEqual(output.splitlines(), ["4", "-----", "2", "-----", "81"])

    def test_invalid_expression_does_not_stop_other_expressions(self):
        text = "1 + 1\n-----\n5 / 0\n-----\n(2 + 3\n-----\n4 * 4\n"
        output, counts = self.run_with_input(text)
        self.assertEqual(
            output.splitlines(),
            [
                "2",
                "-----",
                "Error: Division by zero",
                "-----",
                "Error: Unmatched '('",
                "-----",
                "16",
            ],
        )
        self.assertEqual(counts, (4, 2))

    def test_output_has_same_number_of_lines_as_input(self):
        text = "1 + 1\n2 +\n-----\nabc\n3 * 3\n"
        output, _ = self.run_with_input(text)
        self.assertEqual(len(output.splitlines()), len(text.splitlines()))

    def test_blank_lines_are_preserved(self):
        output, counts = self.run_with_input("1 + 1\n\n2 + 2\n")
        self.assertEqual(output.splitlines(), ["2", "", "4"])
        self.assertEqual(counts, (2, 0))

    def test_windows_line_endings_are_handled(self):
        self.input_path.write_bytes(b"1 + 1\r\n-----\r\n2 + 2\r\n")
        process_file(self.input_path, self.output_path)
        output = self.output_path.read_text(encoding="utf-8")
        self.assertEqual(output.splitlines(), ["2", "-----", "4"])

    def test_empty_input_file_gives_empty_output(self):
        output, counts = self.run_with_input("")
        self.assertEqual(output.strip(), "")
        self.assertEqual(counts, (0, 0))

    def test_missing_input_file_raises_file_not_found(self):
        missing = Path(self.temp_dir.name) / "does_not_exist.txt"
        with self.assertRaises(FileNotFoundError):
            process_file(missing, self.output_path)

    def test_missing_input_file_does_not_create_output(self):
        missing = Path(self.temp_dir.name) / "does_not_exist.txt"
        with self.assertRaises(FileNotFoundError):
            process_file(missing, self.output_path)
        self.assertFalse(self.output_path.exists())


class TestMainProgram(unittest.TestCase):
    def test_main_reports_missing_input_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "input.txt"
            output = Path(temp_dir) / "output.txt"
            captured = io.StringIO()
            with mock.patch.object(main, "INPUT_FILE", missing), mock.patch.object(
                main, "OUTPUT_FILE", output
            ), redirect_stdout(captured):
                exit_code = main.main()
        self.assertEqual(exit_code, 1)
        self.assertIn("input file not found", captured.getvalue())

    def test_main_processes_valid_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.txt"
            output_path = Path(temp_dir) / "output.txt"
            input_path.write_text("2 + 2\n", encoding="utf-8")
            with mock.patch.object(main, "INPUT_FILE", input_path), mock.patch.object(
                main, "OUTPUT_FILE", output_path
            ), redirect_stdout(io.StringIO()):
                exit_code = main.main()
            self.assertEqual(exit_code, 0)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "4\n")


if __name__ == "__main__":
    unittest.main()
