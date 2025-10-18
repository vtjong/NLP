"""Parse evaluation by span charts (precision/recall/F1 per sentence).

This module provides comprehensive evaluation capabilities for constituency
parsing by comparing system-generated parses against gold standard parses.
It normalizes trees by collapsing unary chains and uniquifying leaves, projects
them to span charts, and computes precision, recall, and F1 scores.

Key Features:
    - Span-based evaluation using binary upper-triangular charts
    - Robust tree normalization (unary collapse + lexeme uniquification)
    - Per-sentence and aggregate metrics computation
    - CLI interface for batch evaluation

CLI Usage:
    python -m pcfg.eval --sys output.parses --gold gold.txt --out output.eval

Example:
    >>> sys_parses = ["(S (NP John) (VP runs))"]
    >>> gold_parses = ["(S (NP John) (VP runs))"]
    >>> precisions, recalls, f1s = evaluate(sys_parses, gold_parses)
    >>> print(f"F1: {f1s[0]:.3f}")
    F1: 1.000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .exceptions import EvaluationError, FileError
from .tree import Tree

# Configure logging
logger = logging.getLogger(__name__)


def _read_sys_parses(path: Path) -> List[str]:
    """Read system parse output file, filtering out log-likelihood lines.

    :param path: Path to system output file
    :return: List of parse tree strings
    :raises FileNotFoundError: If the file does not exist
    :raises UnicodeDecodeError: If the file cannot be decoded as UTF-8
    """
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        filtered_lines = [ln for ln in lines if ln and not ln.startswith("LL")]
        logger.info(f"Read {len(filtered_lines)} system parses from {path}")
        return filtered_lines
    except FileNotFoundError:
        logger.error(f"System parses file not found: {path}")
        raise FileError(
            f"System parses file not found: {path}", file_path=str(path), operation="read"
        )
    except UnicodeDecodeError as e:
        logger.error(f"Failed to decode system parses file {path}: {e}")
        raise FileError(
            f"Failed to decode system parses file {path}: {e}",
            file_path=str(path),
            operation="read",
        )


def _read_gold_parses(path: Path) -> List[str]:
    """Read gold standard parse file.

    :param path: Path to gold standard file
    :return: List of gold parse tree strings
    :raises FileNotFoundError: If the file does not exist
    :raises UnicodeDecodeError: If the file cannot be decoded as UTF-8
    """
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        filtered_lines = [ln for ln in lines if ln]
        logger.info(f"Read {len(filtered_lines)} gold parses from {path}")
        return filtered_lines
    except FileNotFoundError:
        logger.error(f"Gold parses file not found: {path}")
        raise FileError(
            f"Gold parses file not found: {path}", file_path=str(path), operation="read"
        )
    except UnicodeDecodeError as e:
        logger.error(f"Failed to decode gold parses file {path}: {e}")
        raise FileError(
            f"Failed to decode gold parses file {path}: {e}", file_path=str(path), operation="read"
        )


def _spans_list(t: Tree) -> List[str]:
    """Collect leaves (lexemes) depth-first in order.

    :param t: Input tree
    :return: List of leaf lexemes in depth-first order
    """
    if t.is_leaf():
        return [t.c]
    spans: List[str] = []
    for ch in t.ch:
        spans.extend(_spans_list(ch))
    return spans


def _chart_for_tree(t: Tree) -> np.ndarray:
    """Convert a tree to a binary upper-triangular chart of spans.

    Creates a binary matrix where chart[i, j] = 1 if there exists a span
    from position i to position i+j in the tree.

    :param t: Input tree
    :return: Binary upper-triangular chart matrix
    """
    # Build lexeme index map
    ordered = _spans_list(t)
    idx: Dict[str, int] = {lex: i for i, lex in enumerate(ordered)}
    n = len(ordered)
    chart = np.zeros((n, n), dtype=int)

    def visit(node: Tree) -> None:
        """Recursively mark spans in the chart."""
        if node.is_leaf():
            return
        leaves = _spans_list(node)
        i0 = idx[leaves[0]]
        length = len(leaves) - 1
        chart[i0, length] = 1
        for ch in node.ch:
            visit(ch)

    visit(t)
    return chart


def _compute_counts(sys_chart: np.ndarray, gold_chart: np.ndarray) -> Tuple[int, int, int]:
    """Compute span counts for precision/recall calculation.

    :param sys_chart: System parse chart
    :param gold_chart: Gold standard chart
    :return: Tuple of (true_positives, system_spans, gold_spans)
    """
    tp = int(np.sum((sys_chart != 0) & (gold_chart != 0)))
    sys_n = int(np.sum(sys_chart != 0))
    gold_n = int(np.sum(gold_chart != 0))
    return tp, sys_n, gold_n


def _safe_div(a: float, b: float) -> float:
    """Safely divide two numbers, returning 0.0 if denominator is zero.

    :param a: Numerator
    :param b: Denominator
    :return: a/b if b != 0, else 0.0
    """
    return (a / b) if b else 0.0


def evaluate(
    sys_parses: Sequence[str], gold_parses: Sequence[str]
) -> Tuple[List[float], List[float], List[float]]:
    """Compute per-sentence precision, recall, and F1 scores.

    This function evaluates constituency parsing performance by comparing
    system-generated parses against gold standard parses using span-based
    metrics. Trees are normalized by collapsing unary chains and uniquifying
    lexemes before evaluation.

    :param sys_parses: System-generated parse trees (one per line)
    :param gold_parses: Gold standard parse trees (one per line)
    :return: Tuple of (precisions, recalls, f1s) lists aligned to input order
    :raises ValueError: If the number of system and gold parses don't match
    :raises ValueError: If any parse string is malformed
    """
    if len(sys_parses) != len(gold_parses):
        raise ValueError(
            f"Mismatched parse counts: {len(sys_parses)} system vs {len(gold_parses)} gold"
        )

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for i, (sys_t, gold_t) in enumerate(zip(sys_parses, gold_parses)):
        try:
            sys_tree = Tree.from_string(sys_t).collapse_unary().uniquify_lexemes()
            gold_tree = Tree.from_string(gold_t).collapse_unary().uniquify_lexemes()
            sys_chart = _chart_for_tree(sys_tree)
            gold_chart = _chart_for_tree(gold_tree)
            tp, sys_n, gold_n = _compute_counts(sys_chart, gold_chart)
            p = _safe_div(tp, sys_n)
            r = _safe_div(tp, gold_n)
            f = _safe_div(2 * p * r, p + r)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
        except ValueError as e:
            logger.error(f"Failed to parse sentence {i}: {e}")
            raise EvaluationError(
                f"Parse error at sentence {i}: {e}",
                sentence_index=i,
                parse_info={"sys_parse": sys_t, "gold_parse": gold_t},
            ) from e

    logger.info(f"Evaluated {len(precisions)} sentences")
    return precisions, recalls, f1s


def write_eval(
    out_path: Path, precisions: List[float], recalls: List[float], f1s: List[float]
) -> None:
    """Write evaluation results to output file.

    :param out_path: Output file path
    :param precisions: List of precision scores
    :param recalls: List of recall scores
    :param f1s: List of F1 scores
    :raises OSError: If the file cannot be written
    """
    try:
        with out_path.open("w", encoding="utf-8") as fh:
            for p, r, f in zip(precisions, recalls, f1s):
                fh.write(f"P {p:.6f};R {r:.6f};F {f:.6f}\n")
        logger.info(f"Wrote evaluation results to {out_path}")
    except OSError as e:
        logger.error(f"Failed to write evaluation results to {out_path}: {e}")
        raise FileError(
            f"Failed to write evaluation results to {out_path}: {e}",
            file_path=str(out_path),
            operation="write",
        )


def main(argv: List[str] | None = None) -> None:
    """Main CLI entry point for parse evaluation.

    :param argv: Command line arguments (for testing)
    """
    parser = argparse.ArgumentParser(
        description="Evaluate constituency parses against gold standard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  %(prog)s --sys output.parses --gold gold.txt --out results.eval
  %(prog)s --sys parses.txt --gold gold.txt --out eval.txt
        """,
    )
    parser.add_argument(
        "--sys", required=True, help="System parse output file (format: tree per line)"
    )
    parser.add_argument(
        "--gold", required=True, help="Gold standard parse file (format: tree per line)"
    )
    parser.add_argument(
        "--out", default="output.eval", help="Output evaluation file path (default: output.eval)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        sys_parses = _read_sys_parses(Path(args.sys))
        gold_parses = _read_gold_parses(Path(args.gold))
        p, r, f = evaluate(sys_parses, gold_parses)
        write_eval(Path(args.out), p, r, f)

        # Print summary statistics
        avg_p = sum(p) / len(p) if p else 0.0
        avg_r = sum(r) / len(r) if r else 0.0
        avg_f = sum(f) / len(f) if f else 0.0
        print(f"Evaluation complete: Avg P={avg_p:.3f}, R={avg_r:.3f}, F1={avg_f:.3f}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
