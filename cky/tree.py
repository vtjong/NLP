"""Lightweight constituency tree with robust parsing and transformations.

This module defines a minimal Tree structure used by MLE training, CKY
parsing, and evaluation. It includes a recursive-descent parser from
bracketed strings and supports modular tree transformations.

Key Features:
- Immutable tree structure for safe algorithm use
- Deterministic parsing with clear error messages
- Modular transformation system for tree processing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .exceptions import ParseError
from .protocols import AbstractTreeTransformer

# Configure logging
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Tree:
    """Constituency tree node.

    :param c: Category/lexeme label
    :param ch: Ordered list of children
    """

    c: str
    ch: List["Tree"] = field(default_factory=list)

    # ---------- Construction helpers ----------

    @staticmethod
    def from_string(s: str) -> "Tree":
        """Parse a bracketed tree string into a Tree.

        Grammar:
            TREE := ATOM | '(' ATOM { TREE } ')'
            ATOM := any token without whitespace or parentheses

        :param s: Input string
        :return: Parsed Tree
        :raises ValueError: If the string is malformed
        """
        s = s.strip()
        if not s:
            raise ParseError("Cannot parse empty string", input_string=s)

        try:
            tree, rest = _read_tree(s)
            rest = rest.strip()
            if rest:
                raise ParseError(
                    f"Unexpected trailing input after tree: {rest[:40]}", input_string=s
                )
            return tree
        except Exception as e:
            logger.error(f"Failed to parse tree string: {s[:50]}...")
            raise ParseError(f"Parse error: {e}", input_string=s) from e

    def __str__(self) -> str:
        if not self.ch:
            return self.c
        return "(" + self.c + " " + " ".join(str(t) for t in self.ch) + ")"

    # ---------- Structural predicates ----------

    def is_leaf(self) -> bool:
        """Return True if the node has no children."""
        return not self.ch

    def is_unary_layer(self) -> bool:
        """Return True if this level is unary (one child that is non-leaf)."""
        if len(self.ch) != 1:
            return False
        child = self.ch[0]
        return not child.is_leaf()

    # ---------- Transformations ----------

    def collapse_unary(self) -> "Tree":
        """Collapse unary chains using the UnaryCollapser transformer.

        :return: New Tree with unary layers collapsed
        """
        collapser = UnaryCollapser()
        return collapser.transform(self)

    def uniquify_lexemes(self) -> "Tree":
        """Make lexemes unique using the LexemeUniquifier transformer.

        :return: New Tree with renamed leaves and same structure
        """
        uniquifier = LexemeUniquifier()
        return uniquifier.transform(self)


class UnaryCollapser(AbstractTreeTransformer):
    """Tree transformer that collapses unary chains.

    Reduces grammar complexity by merging consecutive unary productions
    into single rules with concatenated labels.
    """

    def transform(self, tree: Tree) -> Tree:
        """Collapse unary chains by concatenating labels with '+'.

        This transformation is crucial for PCFG training as it reduces
        the number of unary rules and improves grammar stability.

        Example:
            (A (B (C x))) -> (A+B+C x)

        :param tree: Input tree
        :return: New Tree with unary layers collapsed
        """
        if tree.is_leaf():
            return tree

        if tree.is_unary_layer():
            # Collapse current label with child's label
            child = tree.ch[0]
            collapsed_child = self.transform(child)
            # Merge labels
            merged_label = f"{tree.c}+{collapsed_child.c}"
            return Tree(merged_label, collapsed_child.ch)

        # General case: collapse each child
        return Tree(tree.c, [self.transform(c) for c in tree.ch])


class LexemeUniquifier(AbstractTreeTransformer):
    """Tree transformer that makes lexemes unique.

    Ensures each leaf token has a unique identifier to prevent
    ambiguity in span-based evaluation metrics.
    """

    def transform(self, tree: Tree) -> Tree:
        """Make repeated leaf lexemes unique with suffix _k.

        This transformation is essential for evaluation as it ensures
        that each leaf token has a unique identifier, preventing
        ambiguity in span-based evaluation.

        Example:
            leaves a, a, b -> a_1, a_2, b_1

        :param tree: Input tree
        :return: New Tree with renamed leaves and same structure
        """
        counts: Dict[str, int] = {}
        _, new_tree = self._uniquify(tree, counts)
        return new_tree

    def _uniquify(self, t: Tree, counts: Dict[str, int]) -> Tuple[str, Tree]:
        """Recursively uniquify lexemes in the tree.

        :param t: Input tree
        :param counts: Dictionary tracking lexeme counts
        :return: Tuple of (original_label, uniquified_tree)
        """
        if t.is_leaf():
            lex = t.c
            counts[lex] = counts.get(lex, 0) + 1
            unique = f"{lex}_{counts[lex]}"
            return unique, Tree(unique, [])

        new_children: List[Tree] = []
        for ch in t.ch:
            _, child_new = self._uniquify(ch, counts)
            new_children.append(child_new)

        return t.c, Tree(t.c, new_children)


def _read_atom(s: str) -> Tuple[str, str]:
    if not s or s[0] in "() ":
        raise ParseError("Expected atom but found delimiter or empty string", input_string=s)

    i = 0
    n = len(s)
    while i < n and s[i] not in "() \t\r\n":
        i += 1

    atom = s[:i]
    remaining = s[i:].lstrip()
    return atom, remaining


def _read_tree(s: str) -> Tuple[Tree, str]:
    s = s.lstrip()
    if not s:
        raise ParseError("Empty input", input_string=s)

    if s[0] != "(":
        # Leaf/atom
        atom, rest = _read_atom(s)
        return Tree(atom, []), rest

    # Non-terminal
    s = s[1:].lstrip()  # consume '('
    if not s:
        raise ParseError("Empty parentheses", input_string=s)

    label, s = _read_atom(s)
    children: List[Tree] = []
    s = s.lstrip()

    while s and s[0] != ")":
        child, s = _read_tree(s)
        children.append(child)
        s = s.lstrip()

    if not s or s[0] != ")":
        raise ParseError("Unbalanced parentheses", input_string=s)

    s = s[1:].lstrip()  # consume ')'
    return Tree(label, children), s


def _uniquify(t: Tree, counts: Dict[str, int]) -> Tuple[str, Tree]:
    if t.is_leaf():
        lex = t.c
        counts[lex] = counts.get(lex, 0) + 1
        unique = f"{lex}_{counts[lex]}"
        return unique, Tree(unique, [])
    new_children: List[Tree] = []
    for ch in t.ch:
        _, child_new = _uniquify(ch, counts)
        new_children.append(child_new)
    return t.c, Tree(t.c, new_children)
