"""Core abstractions for PCFG parsing system.

This module defines essential interfaces that enable modularity,
testability, and extensibility in the parsing pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class AbstractTreeTransformer(ABC):
    """Base class for tree transformation operations.

    Enables composition of different tree transformations and
    supports dependency injection for testing.
    """

    @abstractmethod
    def transform(self, tree: Any) -> Any:
        """Transform a tree structure.

        :param tree: Input tree
        :return: Transformed tree
        """
        pass


class AbstractRuleCounter(ABC):
    """Base class for rule counting operations.

    Provides interface for counting grammatical rules from parse trees.
    Enables mocking for unit tests and supports different counting strategies.
    """

    @abstractmethod
    def count_rules(
        self, trees: List[Any], train_mode: bool = True
    ) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
        """Count binary and lexical rules from trees.

        :param trees: List of parsed trees
        :param train_mode: Whether in training mode
        :return: Tuple of (binary_rules, lexical_rules)
        """
        pass


class AbstractProbabilityCalculator(ABC):
    """Base class for probability calculation operations.

    Provides interface for converting rule counts to log probabilities.
    Supports different calculation strategies and smoothing techniques.
    """

    @abstractmethod
    def calculate_log_probabilities(
        self, rule_counts: Dict[str, Dict[str, int]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate log probabilities from rule counts.

        :param rule_counts: Rule count dictionary
        :return: Log probability dictionary
        :raises TrainingError: If calculation fails
        """
        pass


class ConfigurationManager:
    """Centralized configuration management.

    Manages all system constants and configuration values.
    Provides single source of truth for configuration parameters.
    """

    # Unknown word/tag constants
    UNK_TERMINAL = "<UNK-T>"
    UNK_NONTERMINAL = "<UNK-NT>"
    UNK_NONTERMINAL_X = "<UNK-NT>^X"

    # Default values
    DEFAULT_TRAIN_SPLIT = 0.9
    DEFAULT_OUTPUT_FILE = "output.parses"
    DEFAULT_MODEL_FILE = "model.pcfg"
    DEFAULT_EVAL_FILE = "output.eval"

    # File encodings
    DEFAULT_ENCODING = "utf-8"

    # Logging configuration
    DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def get_unk_token(cls, token_type: str) -> str:
        """Get appropriate UNK token for type.

        :param token_type: Type of token ("terminal", "preterminal",
        "nonterminal")
        :return: Appropriate UNK token
        """
        mapping = {
            "terminal": cls.UNK_TERMINAL,
            "preterminal": cls.UNK_NONTERMINAL_X,
            "nonterminal": cls.UNK_NONTERMINAL,
        }
        return mapping.get(token_type, cls.UNK_NONTERMINAL)
