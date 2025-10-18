"""Custom exception classes for PCFG parsing and training.

This module defines domain-specific exception classes that provide
clear error messages and context for debugging PCFG-related issues.
"""

from __future__ import annotations

from typing import Any, Optional


class PCFGError(Exception):
    """Base exception class for PCFG-related errors.

    This is the base class for all PCFG-specific exceptions, allowing
    for easy exception handling and providing a common interface.
    """

    def __init__(self, message: str, context: Optional[Any] = None) -> None:
        """Initialize PCFG error.

        :param message: Error message describing the issue
        :param context: Additional context information for debugging
        """
        super().__init__(message)
        self.message = message
        self.context = context


class ParseError(PCFGError):
    """Exception raised when tree parsing fails.

    This exception is raised when bracketed tree strings cannot be
    parsed due to malformed syntax, unbalanced parentheses, or other
    structural issues.
    """

    def __init__(
        self, message: str, input_string: Optional[str] = None, position: Optional[int] = None
    ) -> None:
        """Initialize parse error.

        :param message: Error message describing the parse failure
        :param input_string: The input string that failed to parse
        :param position: Character position where parsing failed
        """
        super().__init__(message, {"input_string": input_string, "position": position})
        self.input_string = input_string
        self.position = position


class GrammarError(PCFGError):
    """Exception raised when grammar-related operations fail.

    This exception is raised when grammar loading, validation, or
    processing operations encounter issues such as malformed rules,
    missing probabilities, or invalid tag definitions.
    """

    def __init__(
        self, message: str, rule: Optional[str] = None, line_number: Optional[int] = None
    ) -> None:
        """Initialize grammar error.

        :param message: Error message describing the grammar issue
        :param rule: The grammar rule that caused the error
        :param line_number: Line number in the grammar file where the error occurred
        """
        super().__init__(message, {"rule": rule, "line_number": line_number})
        self.rule = rule
        self.line_number = line_number


class ParsingError(PCFGError):
    """Exception raised when sentence parsing fails.

    This exception is raised when the CKY algorithm cannot find a valid
    parse for a sentence, or when parsing encounters numerical issues
    such as underflow or invalid probabilities.
    """

    def __init__(
        self,
        message: str,
        sentence: Optional[list[str]] = None,
        grammar_info: Optional[dict] = None,
    ) -> None:
        """Initialize parsing error.

        :param message: Error message describing the parsing failure
        :param sentence: The sentence that failed to parse
        :param grammar_info: Additional grammar information for debugging
        """
        super().__init__(message, {"sentence": sentence, "grammar_info": grammar_info})
        self.sentence = sentence
        self.grammar_info = grammar_info


class EvaluationError(PCFGError):
    """Exception raised when evaluation operations fail.

    This exception is raised when span-based evaluation encounters
    issues such as mismatched parse counts, malformed trees, or
    numerical problems in metric computation.
    """

    def __init__(
        self, message: str, sentence_index: Optional[int] = None, parse_info: Optional[dict] = None
    ) -> None:
        """Initialize evaluation error.

        :param message: Error message describing the evaluation failure
        :param sentence_index: Index of the sentence that caused the error
        :param parse_info: Additional parse information for debugging
        """
        super().__init__(message, {"sentence_index": sentence_index, "parse_info": parse_info})
        self.sentence_index = sentence_index
        self.parse_info = parse_info


class TrainingError(PCFGError):
    """Exception raised when PCFG training fails.

    This exception is raised when MLE training encounters issues such
    as empty training data, invalid tree structures, or numerical
    problems in probability computation.
    """

    def __init__(
        self, message: str, tree_index: Optional[int] = None, training_info: Optional[dict] = None
    ) -> None:
        """Initialize training error.

        :param message: Error message describing the training failure
        :param tree_index: Index of the tree that caused the error
        :param training_info: Additional training information for debugging
        """
        super().__init__(message, {"tree_index": tree_index, "training_info": training_info})
        self.tree_index = tree_index
        self.training_info = training_info


class FileError(PCFGError):
    """Exception raised when file operations fail.

    This exception is raised when file I/O operations encounter issues
    such as missing files, permission errors, or encoding problems.
    """

    def __init__(
        self, message: str, file_path: Optional[str] = None, operation: Optional[str] = None
    ) -> None:
        """Initialize file error.

        :param message: Error message describing the file operation failure
        :param file_path: Path to the file that caused the error
        :param operation: The file operation that failed (read, write, etc.)
        """
        super().__init__(message, {"file_path": file_path, "operation": operation})
        self.file_path = file_path
        self.operation = operation
