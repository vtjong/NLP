"""CKY parser for PCFGs with backpointers and bracketed-tree output.

This module implements the Cocke-Kasami-Younger (CKY) algorithm for parsing
Probabilistic Context-Free Grammars (PCFGs). It provides efficient parsing
of tokenized sentences using dynamic programming with backpointers for
reconstructing parse trees.

Key Features:
    - CKY algorithm in Chomsky Normal Form (CNF)
    - Backpointer-based tree reconstruction

CLI Usage:
    python -m pcfg.cky --model model.pcfg --test test.txt --out output.parses

Example:
    >>> grammar = Grammar.from_file("model.pcfg")
    >>> sentence = ["The", "cat", "runs"]
    >>> log_prob, tree = parse_sentence(sentence, grammar)
    >>> print(f"Log probability: {log_prob:.3f}")
    >>> print(f"Parse tree: {tree}")
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .exceptions import FileError, GrammarError, ParsingError
from .tree import Tree

# Configure logging
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Grammar:
    """In-memory PCFG representation in probability space (not log).

    This class encapsulates a Probabilistic Context-Free Grammar with
    binary and lexical rules, along with tag indexing for efficient
    tensor operations during parsing.

    Attributes:
        binary: Mapping from (left_tag, right_tag) pairs to parent
                probabilities
        lexical: Mapping from words to preterminal probabilities
        tags: Sorted list of all grammatical tags
        tag_to_idx: Mapping from tag strings to integer indices
        idx_to_tag: Mapping from integer indices to tag strings
        root_tag_idxs: Mapping from root tag names to their indices
    """

    # Binary rules: (left_tag, right_tag) -> {parent_tag -> probability}
    binary: Dict[Tuple[str, str], Dict[str, float]]
    # Lexical rules: word -> {preterminal_tag -> probability}
    lexical: Dict[str, Dict[str, float]]
    # Tag indexing for efficient tensor operations
    tags: List[str]
    tag_to_idx: Dict[str, int]
    idx_to_tag: Dict[int, str]
    root_tag_idxs: Dict[str, int]

    @classmethod
    def from_file(cls, path: Path) -> "Grammar":
        """Load a Grammar from a PCFG model file.

        :param path: Path to the PCFG model file
        :return: Loaded Grammar instance
        :raises FileNotFoundError: If the model file does not exist
        :raises ValueError: If the model file format is invalid
        """
        return _read_model(path)


def _read_model(path: Path) -> Grammar:
    """Load a PCFG model from file and construct Grammar object.

    Expected format:
        G <LHS> : <LEFT> <RIGHT> <log_prob>
        X <PRETERM> : <WORD> <log_prob>

    :param path: Path to the PCFG model file
    :return: Grammar instance
    :raises FileNotFoundError: If the model file does not exist
    :raises ValueError: If the model file format is invalid
    """
    try:
        pos: set[str] = set()
        binary: Dict[Tuple[str, str], Dict[str, float]] = {}
        lexical: Dict[str, Dict[str, float]] = {}

        for line_num, line in enumerate(path.read_text(encoding="utf-8").strip().splitlines(), 1):
            parts = line.strip().split()
            if not parts:
                continue

            if len(parts) < 5:
                raise GrammarError(
                    f"Invalid line format at line {line_num}: {line}", line_number=line_num
                )

            kind = parts[0]
            if kind == "G":
                # G LHS : LEFT RIGHT logp
                if len(parts) != 6 or parts[2] != ":":
                    raise GrammarError(
                        f"Invalid binary rule format at line {line_num}: {line}",
                        line_number=line_num,
                    )
                lhs, left, right, logp_str = parts[1], parts[3], parts[4], parts[5]
                try:
                    logp = float(logp_str)
                except ValueError:
                    raise GrammarError(
                        f"Invalid log probability at line {line_num}: {logp_str}",
                        line_number=line_num,
                    )
                p = math.exp(logp)
                pos.update([lhs, left, right])
                binary.setdefault((left, right), {})[lhs] = p

            elif kind == "X":
                # X PRETERM : WORD logp
                if len(parts) != 5 or parts[2] != ":":
                    raise GrammarError(
                        f"Invalid lexical rule format at line {line_num}: {line}",
                        line_number=line_num,
                    )
                lhs, word, logp_str = parts[1], parts[3], parts[4]
                try:
                    logp = float(logp_str)
                except ValueError:
                    raise GrammarError(
                        f"Invalid log probability at line {line_num}: {logp_str}",
                        line_number=line_num,
                    )
                p = math.exp(logp)
                pos.add(lhs)
                lexical.setdefault(word, {})[lhs] = p

            else:
                logger.warning(f"Unknown rule type '{kind}' at line {line_num}: {line}")

        tags = sorted(pos)
        tag_to_idx = {t: i for i, t in enumerate(tags)}
        idx_to_tag = {i: t for t, i in tag_to_idx.items()}
        root_tag_idxs = {t: tag_to_idx[t] for t in tags if "ROOT" in t and "|" not in t}

        logger.info(
            f"Loaded grammar with {len(tags)} tags, {len(binary)} binary rules, {len(lexical)} lexical entries"
        )
        return Grammar(binary, lexical, tags, tag_to_idx, idx_to_tag, root_tag_idxs)

    except FileNotFoundError:
        logger.error(f"Model file not found: {path}")
        raise FileError(f"Model file not found: {path}", file_path=str(path), operation="read")
    except UnicodeDecodeError as e:
        logger.error(f"Failed to decode model file {path}: {e}")
        raise FileError(
            f"Failed to decode model file {path}: {e}", file_path=str(path), operation="read"
        )


# ---------------- CKY Algorithm Core ----------------


def _cky_base(
    sentence: List[str],
    num_tags: int,
    lex: Dict[str, Dict[str, float]],
    tag_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize CKY trellis and backpointer arrays.

    Sets up the base case for lexical rules (length-1 spans) in the
    dynamic programming table.

    :param sentence: Tokenized input sentence
    :param num_tags: Number of grammatical tags
    :param lex: Lexical rule probabilities
    :param tag_to_idx: Tag to index mapping
    :return: Tuple of (trellis, backpointer) arrays
    """
    n = len(sentence)
    trellis = np.zeros((n, n, num_tags), dtype=float)
    bkptr = np.empty((n, n, num_tags), dtype=object)
    bkptr.fill(None)

    for j, w in enumerate(sentence):
        dist = lex.get(w, {})  # UNK handling via upstream data preparation
        for t, p in dist.items():
            trellis[0, j, tag_to_idx[t]] = p

    logger.debug(f"Initialized CKY trellis for sentence of length {n}")
    return trellis, bkptr


def _cky_inductive(
    trellis: np.ndarray,
    bkptr: np.ndarray,
    i: int,
    j: int,
    k: int,
    grammar: Grammar,
) -> None:
    """Apply CKY inductive step for span [j, j+i+1).

    Updates the trellis and backpointer arrays for span length i+1,
    starting at position j, using split point k.

    :param trellis: Dynamic programming probability table
    :param bkptr: Backpointer table for tree reconstruction
    :param i: Span length minus 1
    :param j: Starting position
    :param k: Split point
    :param grammar: Grammar rules
    """
    row, col = i - k - 1, j + k + 1
    for (left_t, right_t), parent_map in grammar.binary.items():
        l_idx = grammar.tag_to_idx[left_t]
        r_idx = grammar.tag_to_idx[right_t]
        left_p = trellis[k, j, l_idx]
        right_p = trellis[row, col, r_idx]

        if left_p == 0.0 or right_p == 0.0:
            continue

        prod = left_p * right_p
        for parent_t, rule_p in parent_map.items():
            p_idx = grammar.tag_to_idx[parent_t]
            cand = rule_p * prod
            if cand > trellis[i, j, p_idx]:
                trellis[i, j, p_idx] = cand
                bkptr[i, j, p_idx] = (
                    cand,  # best probability for this cell/parent
                    (k, j),  # left span pointer
                    (row, col),  # right span pointer
                    l_idx,  # left tag index
                    r_idx,  # right tag index
                )


def _best_root(
    n: int, trellis: np.ndarray, grammar: Grammar
) -> Tuple[Optional[Tuple[int, int, int]], float]:
    """Find the best root parse for the complete sentence.

    Searches for the highest probability root tag that spans the entire
    sentence (span [0, n)).

    :param n: Sentence length
    :param trellis: Dynamic programming probability table
    :param grammar: Grammar with root tag definitions
    :return: Tuple of (best_root_pointer, best_probability)
    """
    best_p = 0.0
    best_idx: Optional[int] = None

    for _, ridx in grammar.root_tag_idxs.items():
        p = trellis[n - 1, 0, ridx]
        if p > best_p:
            best_p = p
            best_idx = ridx

    if best_idx is None or best_p == 0.0:
        logger.warning(f"No valid root parse found for sentence of length {n}")
        return None, 0.0

    return (n - 1, 0, best_idx), best_p


def _children(
    ptr: Tuple[int, int, int], bkptr: np.ndarray
) -> Tuple[Tuple[int, int], Tuple[int, int], int, int]:
    """Extract child pointers and tag indices from backpointer.

    :param ptr: Pointer tuple (i, j, tag_idx)
    :param bkptr: Backpointer array
    :return: Tuple of (left_ptr, right_ptr, left_tag_idx, right_tag_idx)
    :raises ValueError: If backpointer is missing for non-leaf cell
    """
    i, j, tag_idx = ptr
    record = bkptr[i, j, tag_idx]
    if record is None:
        raise ValueError(f"Missing backpointer for cell ({i}, {j}, {tag_idx})")
    _, l_ptr, r_ptr, l_idx, r_idx = record
    return l_ptr, r_ptr, l_idx, r_idx


def _to_tree(
    ptr: Tuple[int, int, int], bkptr: np.ndarray, idx_to_tag: Dict[int, str], sent: List[str]
) -> str:
    """Reconstruct bracketed tree string from backpointers.

    :param ptr: Pointer tuple (i, j, tag_idx)
    :param bkptr: Backpointer array
    :param idx_to_tag: Index to tag mapping
    :param sent: Original sentence tokens
    :return: Bracketed tree string
    :raises ValueError: If backpointer reconstruction fails
    """
    i, j, tag_idx = ptr
    label = idx_to_tag[tag_idx]

    if i == 0:
        # Leaf node
        return f"({label} {sent[j]})"

    try:
        l_ptr, r_ptr, l_idx, r_idx = _children((i, j, tag_idx), bkptr)
        lt = _to_tree((*l_ptr, l_idx), bkptr, idx_to_tag, sent)
        rt = _to_tree((*r_ptr, r_idx), bkptr, idx_to_tag, sent)
        return f"({label} {lt} {rt})"
    except ValueError as e:
        raise ValueError(f"Tree reconstruction failed at ({i}, {j}, {tag_idx}): {e}") from e


def parse_sentence(sentence: List[str], grammar: Grammar) -> Tuple[float, str]:
    """Parse one tokenized sentence using the CKY algorithm.

    This function implements the Cocke-Kasami-Younger algorithm for parsing
    Probabilistic Context-Free Grammars. It uses dynamic programming to find
    the highest probability parse tree for the input sentence.

    :param sentence: Tokenized input sentence
    :param grammar: PCFG grammar rules and probabilities
    :return: Tuple of (log_probability, bracketed_tree_string or 'FAIL')
    :raises ValueError: If sentence is empty or grammar is invalid
    """
    if not sentence:
        raise ParsingError("Cannot parse empty sentence", sentence=sentence)

    if not grammar.root_tag_idxs:
        raise ParsingError(
            "Grammar has no root tags defined",
            grammar_info={"root_tags": list(grammar.root_tag_idxs.keys())},
        )

    n = len(sentence)
    logger.debug(f"Parsing sentence of length {n}: {' '.join(sentence)}")

    # Initialize CKY tables
    trellis, bkptr = _cky_base(sentence, len(grammar.tags), grammar.lexical, grammar.tag_to_idx)

    # Fill dynamic programming table
    for i in range(1, n):
        for j in range(0, n - i):
            for k in range(0, i):
                _cky_inductive(trellis, bkptr, i, j, k, grammar)

    # Find best root parse
    root_ptr, root_p = _best_root(n, trellis, grammar)
    if root_ptr is None:
        logger.warning(f"No valid parse found for sentence: {' '.join(sentence)}")
        return float("nan"), "FAIL"

    log_prob = math.log(root_p)
    tree_str = _to_tree(root_ptr, bkptr, grammar.idx_to_tag, sentence)

    logger.debug(f"Parse completed with log probability: {log_prob:.3f}")
    return log_prob, tree_str


# ---------------- I/O & CLI ----------------


def _read_test_sentences(path: Path, vocab: Dict[str, Dict[str, float]]) -> List[List[str]]:
    """Read test sentences and map unknown words to UNK token.

    :param path: Path to test sentences file
    :param vocab: Vocabulary from training data
    :return: List of tokenized sentences with UNK mapping
    :raises FileNotFoundError: If the test file does not exist
    :raises UnicodeDecodeError: If the file cannot be decoded as UTF-8
    """
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()

        def map_unk(tok: str) -> str:
            """Map unknown tokens to UNK symbol."""
            return tok if tok in vocab else "<UNK-T>"

        sentences = [[map_unk(t) for t in line.split()] for line in lines if line.strip()]
        logger.info(f"Read {len(sentences)} test sentences from {path}")
        return sentences

    except FileNotFoundError:
        logger.error(f"Test sentences file not found: {path}")
        raise FileError(
            f"Test sentences file not found: {path}", file_path=str(path), operation="read"
        )
    except UnicodeDecodeError as e:
        logger.error(f"Failed to decode test sentences file {path}: {e}")
        raise FileError(
            f"Failed to decode test sentences file {path}: {e}",
            file_path=str(path),
            operation="read",
        )


def write_parses(out_path: Path, results: List[Tuple[float, str]]) -> None:
    """Write parsing results to output file.

    :param out_path: Output file path
    :param results: List of (log_probability, tree_string) tuples
    :raises OSError: If the file cannot be written
    """
    try:
        with out_path.open("w", encoding="utf-8") as fh:
            for i, (ll, tree) in enumerate(results):
                fh.write(f"LL{i}: {ll}\n")
                fh.write(tree + "\n")
        logger.info(f"Wrote {len(results)} parse results to {out_path}")
    except OSError as e:
        logger.error(f"Failed to write parse results to {out_path}: {e}")
        raise FileError(
            f"Failed to write parse results to {out_path}: {e}",
            file_path=str(out_path),
            operation="write",
        )


def main(argv: List[str] | None = None) -> None:
    """Main CLI entry point for CKY parsing.

    :param argv: Command line arguments (for testing)
    """
    parser = argparse.ArgumentParser(
        description="CKY parser for Probabilistic Context-Free Grammars",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s --model model.pcfg --test test.txt --out output.parses
  %(prog)s --model grammar.pcfg --test sentences.txt --out results.txt
        """,
    )
    parser.add_argument("--model", required=True, help="Path to PCFG model file")
    parser.add_argument("--test", required=True, help="Path to tokenized test sentences file")
    parser.add_argument(
        "--out", default="output.parses", help="Output parses file path (default: output.parses)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        grammar = Grammar.from_file(Path(args.model))
        sentences = _read_test_sentences(Path(args.test), grammar.lexical)

        logger.info(f"Starting to parse {len(sentences)} sentences")
        results = []
        for i, sentence in enumerate(sentences):
            try:
                log_prob, tree = parse_sentence(sentence, grammar)
                results.append((log_prob, tree))
                if (i + 1) % 100 == 0:
                    logger.info(f"Parsed {i + 1}/{len(sentences)} sentences")
            except Exception as e:
                logger.error(f"Failed to parse sentence {i}: {' '.join(sentence)} - {e}")
                results.append((float("nan"), "FAIL"))

        write_parses(Path(args.out), results)

        # Print summary statistics
        successful_parses = sum(1 for _, tree in results if tree != "FAIL")
        avg_log_prob = (
            sum(log_prob for log_prob, _ in results if not math.isnan(log_prob))
            / successful_parses
            if successful_parses > 0
            else float("nan")
        )
        print(
            f"Parsing complete: {successful_parses}/{len(results)} successful, avg log prob: {avg_log_prob:.3f}"
        )

    except Exception as e:
        logger.error(f"Parsing failed: {e}")
        raise


if __name__ == "__main__":
    main()
