"""Maximum Likelihood Estimation trainer for Probabilistic Context-Free Grammars.

This module implements MLE training for PCFGs with robust unknown word handling.
It processes bracketed constituency trees, collapses unary chains for stability,
counts binary and lexical rules, and outputs a model with log probabilities.

Key Features:
    - Unary chain collapsing for grammar stability
    - Unknown word handling with train/heldout split
    - Rule counting with proper normalization
    - Log probability output for numerical stability
    - Comprehensive error handling and validation

CLI Usage:
    python -m pcfg.mle --trees path/to/trees.txt --split 0.9 --out model.pcfg

Example:
    >>> trees = ["(ROOT (S (NP John) (VP runs)))"]
    >>> G_log_probs, X_log_probs = train_pcfg(trees, split=0.9)
    >>> write_model_pcfg(G_log_probs, X_log_probs, Path("model.pcfg"))
"""

from __future__ import annotations

import argparse
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from .exceptions import FileError, TrainingError
from .protocols import (
    AbstractProbabilityCalculator,
    AbstractRuleCounter,
    ConfigurationManager,
)
from .tree import Tree

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases for clarity
RuleCounts = Dict[str, Dict[str, int]]  # LHS -> { RHS -> count }
RuleLogProbs = Dict[str, Dict[str, float]]  # LHS -> { RHS -> log_prob }

# Unknown word/tag constants
UNK_T = "<UNK-T>"
UNK_NT = "<UNK-NT>"
UNK_NT_X = "<UNK-NT>^X"


class RuleCounter(AbstractRuleCounter):
    """Rule counting implementation for PCFG training.

    Counts binary and lexical rules from parsed trees with support
    for unknown word handling and training/heldout data processing.
    """

    def __init__(self):
        """Initialize rule counter with configuration."""
        self.config = ConfigurationManager()

    def count_rules(
        self, trees: List[Tree], train_mode: bool = True
    ) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
        """Count binary and lexical rules from trees.

        :param trees: List of parsed trees
        :param train_mode: Whether in training mode
        :return: Tuple of (binary_rules, lexical_rules)
        """
        G: Dict[str, Dict[str, int]] = defaultdict(dict)
        X: Dict[str, Dict[str, int]] = defaultdict(dict)
        vocab: Set[str] = set()
        tags: Set[str] = set()

        for tree in trees:
            self._count_tree_rules(tree, tree.c, "", train_mode, vocab, tags, G, X)

        return G, X

    def _count_tree_rules(
        self,
        node: Tree,
        curr_tag: str,
        unary_acc: str,
        train: bool,
        vocab: Set[str],
        tags: Set[str],
        G: Dict[str, Dict[str, int]],
        X: Dict[str, Dict[str, int]],
    ) -> str:
        """Count rules for a single tree node.

        :param node: Current tree node
        :param curr_tag: Current grammatical tag
        :param unary_acc: Accumulated unary chain prefix
        :param train: Whether this is training data
        :param vocab: Vocabulary set (modified in place)
        :param tags: Tag set (modified in place)
        :param G: Binary rule counts (modified in place)
        :param X: Lexical rule counts (modified in place)
        :return: Processed tag for this node
        """
        # Attach collapsed unary prefix if present
        if unary_acc:
            curr_tag = f"{unary_acc}+{curr_tag}"

        if node.is_leaf():
            return curr_tag

        # Handle unary layer
        if node.is_unary_layer():
            child = node.ch[0]
            return self._count_tree_rules(child, child.c, curr_tag, train, vocab, tags, G, X)

        # Handle preterminal
        if self._is_preterminal(node):
            return self._count_preterminal(node, curr_tag, train, vocab, tags, X)

        # Handle binary nonterminal
        if self._is_binary(node):
            return self._count_binary(node, curr_tag, train, vocab, tags, G, X)

        # Out-of-CNF arity: ignore gracefully
        logger.warning(f"Ignoring node with {len(node.ch)} children: {node.c}")
        return curr_tag

    def _is_binary(self, t: Tree) -> bool:
        """Check if tree node has exactly two children.

        :param t: Tree node
        :return: True if node has exactly 2 children
        """
        return len(t.ch) == 2

    def _is_preterminal(self, t: Tree) -> bool:
        """Check if tree node is preterminal (has one leaf child).

        :param t: Tree node
        :return: True if node has exactly 1 child and that child is a leaf
        """
        return len(t.ch) == 1 and t.ch[0].is_leaf()

    def _count_preterminal(
        self,
        node: Tree,
        curr_tag: str,
        train: bool,
        vocab: Set[str],
        tags: Set[str],
        X: Dict[str, Dict[str, int]],
    ) -> str:
        """Count lexical rules for preterminal node.

        :param node: Preterminal tree node
        :param curr_tag: Current grammatical tag
        :param train: Whether this is training data
        :param vocab: Vocabulary set (modified in place)
        :param tags: Tag set (modified in place)
        :param X: Lexical rule counts (modified in place)
        :return: Processed tag for this node
        """
        child = node.ch[0]
        lhs = self._maybe_tag(curr_tag, tags, train, "preterminal")
        rhs_word = child.c
        rhs = rhs_word if train else (rhs_word if rhs_word in vocab else self.config.UNK_TERMINAL)
        if train:
            vocab.add(rhs_word)
        X.setdefault(lhs, {})
        X[lhs][rhs] = X[lhs].get(rhs, 0) + 1
        return lhs

    def _count_binary(
        self,
        node: Tree,
        curr_tag: str,
        train: bool,
        vocab: Set[str],
        tags: Set[str],
        G: Dict[str, Dict[str, int]],
        X: Dict[str, Dict[str, int]],
    ) -> str:
        """Count binary rules for binary node.

        :param node: Binary tree node
        :param curr_tag: Current grammatical tag
        :param train: Whether this is training data
        :param vocab: Vocabulary set (modified in place)
        :param tags: Tag set (modified in place)
        :param G: Binary rule counts (modified in place)
        :param X: Lexical rule counts (modified in place)
        :return: Processed tag for this node
        """
        lhs = self._maybe_tag(curr_tag, tags, train, "nonterminal")
        left = self._count_tree_rules(node.ch[0], node.ch[0].c, "", train, vocab, tags, G, X)
        right = self._count_tree_rules(node.ch[1], node.ch[1].c, "", train, vocab, tags, G, X)
        rhs = f"{left} {right}"
        G.setdefault(lhs, {})
        G[lhs][rhs] = G[lhs].get(rhs, 0) + 1
        return lhs

    def _maybe_tag(self, tag: str, tags: Set[str], train: bool, kind: str) -> str:
        """Apply unknown tag handling based on training mode.

        :param tag: Original tag
        :param tags: Set of known tags (modified in place if training)
        :param train: Whether this is training data
        :param kind: Type of tag ("terminal", "preterminal", "nonterminal")
        :return: Processed tag (possibly mapped to UNK)
        """
        unk = self.config.get_unk_token(kind)

        if kind == "preterminal":
            tag = f"{tag}^X"

        if train:
            tags.add(tag)
            return tag

        return tag if tag in tags else unk


class ProbabilityCalculator(AbstractProbabilityCalculator):
    """Probability calculation implementation for PCFG training.

    Converts rule counts to log probabilities with proper normalization
    and error handling for zero-count rules.
    """

    def calculate_log_probabilities(
        self, rule_counts: Dict[str, Dict[str, int]]
    ) -> Dict[str, Dict[str, float]]:
        """Convert rule counts to log probabilities.

        :param rule_counts: Rule count dictionary
        :return: Rule log probability dictionary
        :raises TrainingError: If any rule has zero total count
        """
        logps: Dict[str, Dict[str, float]] = {}
        for lhs, rhs_counts in rule_counts.items():
            total = sum(rhs_counts.values())
            if total == 0:
                raise TrainingError(
                    f"Rule {lhs} has zero total count",
                    training_info={"rule": lhs, "rhs_counts": rhs_counts},
                )
            logps[lhs] = {rhs: math.log(cnt / total) for rhs, cnt in rhs_counts.items()}
        return logps


def train_pcfg(
    trees: List[str],
    split: float,
) -> Tuple[RuleLogProbs, RuleLogProbs]:
    """Train a PCFG from bracketed tree strings using Maximum Likelihood Estimation.

    This function implements MLE training for PCFGs with the following steps:
    1. Parse and normalize trees (collapse unary chains)
    2. Split data into training and heldout sets
    3. Count rules on training data
    4. Process heldout data to identify unknown words/tags
    5. Convert counts to log probabilities

    :param trees: List of bracketed tree strings
    :param split: Fraction for training set (rest used for UNK mapping)
    :return: Tuple of (binary_rule_log_probs, lexical_rule_log_probs)
    :raises ValueError: If split is not in (0, 1) or trees list is empty
    :raises ValueError: If any tree string is malformed
    """
    if not trees:
        raise TrainingError("Cannot train PCFG from empty tree list")

    if not (0 < split < 1):
        raise TrainingError(f"Split must be in (0, 1), got {split}")

    logger.info(f"Training PCFG from {len(trees)} trees with split {split}")

    # Preprocess: collapse unary for stability
    try:
        parsed = [Tree.from_string(t).collapse_unary() for t in trees]
    except ValueError as e:
        raise TrainingError(
            f"Failed to parse tree strings: {e}", training_info={"num_trees": len(trees)}
        ) from e

    n_train = int(len(parsed) * split)
    train, heldout = parsed[:n_train], parsed[n_train:]

    logger.info(f"Training set: {len(train)} trees, Heldout set: {len(heldout)} trees")

    # Initialize components for rule counting and probability calculation
    rule_counter = RuleCounter()
    prob_calculator = ProbabilityCalculator()

    # Count rules on training data
    G, X = rule_counter.count_rules(train, train_mode=True)

    # Process heldout to populate UNK pathways without altering counts
    rule_counter.count_rules(heldout, train_mode=False)

    logger.info(
        f"Binary rules: {sum(len(rules) for rules in G.values())}, "
        f"Lexical rules: {sum(len(rules) for rules in X.values())}"
    )

    return (
        prob_calculator.calculate_log_probabilities(G),
        prob_calculator.calculate_log_probabilities(X),
    )


def write_model_pcfg(G: RuleLogProbs, X: RuleLogProbs, path: Path) -> None:
    """Write PCFG model to file in canonical format.

    Output format:
        G <LHS> : <RHS> <log_prob>
        X <PRETERM> : <WORD> <log_prob>

    :param G: Binary rule log probabilities
    :param X: Lexical rule log probabilities
    :param path: Output file path
    :raises OSError: If the file cannot be written
    """
    try:
        lines: List[str] = []

        # Write binary rules
        for lhs, rhs_map in G.items():
            for rhs, lp in rhs_map.items():
                lines.append(f"G {lhs} : {rhs} {lp:.6f}")

        # Write lexical rules
        for lhs, rhs_map in X.items():
            for rhs, lp in rhs_map.items():
                lines.append(f"X {lhs} : {rhs} {lp:.6f}")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info(f"Wrote PCFG model with {len(lines)} rules to {path}")

    except OSError as e:
        logger.error(f"Failed to write PCFG model to {path}: {e}")
        raise FileError(
            f"Failed to write PCFG model to {path}: {e}", file_path=str(path), operation="write"
        )


# ---------------- CLI ----------------


# ---------------- CLI Interface ----------------


def _read_trees_file(path: Path) -> List[str]:
    """Read bracketed trees from file.

    Expected format: trees separated by whitespace, each starting with (ROOT

    :param path: Path to trees file
    :return: List of tree strings
    :raises FileNotFoundError: If the file does not exist
    :raises UnicodeDecodeError: If the file cannot be decoded as UTF-8
    """
    try:
        text = path.read_text(encoding="utf-8").replace("\n", " ")
        parts = [p for p in text.split("(ROOT ") if p]
        trees = [f"(ROOT {p}".strip() for p in parts]
        logger.info(f"Read {len(trees)} trees from {path}")
        return trees
    except FileNotFoundError:
        logger.error(f"Trees file not found: {path}")
        raise FileError(f"Trees file not found: {path}", file_path=str(path), operation="read")
    except UnicodeDecodeError as e:
        logger.error(f"Failed to decode trees file {path}: {e}")
        raise FileError(
            f"Failed to decode trees file {path}: {e}", file_path=str(path), operation="read"
        )


def main(argv: List[str] | None = None) -> None:
    """Main CLI entry point for PCFG training.

    :param argv: Command line arguments (for testing)
    """
    parser = argparse.ArgumentParser(
        description="Train a PCFG using Maximum Likelihood Estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s --trees trees.txt --split 0.9 --out model.pcfg
  %(prog)s --trees data/trees.txt --split 0.8 --out grammar.pcfg
        """,
    )
    parser.add_argument("--trees", required=True, help="Path to bracketed trees file")
    parser.add_argument(
        "--split", type=float, default=0.9, help="Training set fraction (default: 0.9)"
    )
    parser.add_argument(
        "--out", default="model.pcfg", help="Output model file path (default: model.pcfg)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        trees = _read_trees_file(Path(args.trees))
        G, X = train_pcfg(trees, split=args.split)
        write_model_pcfg(G, X, Path(args.out))

        # Print summary statistics
        total_binary_rules = sum(len(rules) for rules in G.values())
        total_lexical_rules = sum(len(rules) for rules in X.values())
        print(
            f"Training complete: {total_binary_rules} binary rules, {total_lexical_rules} lexical rules"
        )

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
