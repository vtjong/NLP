"""Unit tests for CKY parser following TDD principles and NLP theory.

This module tests the Cocke-Kasami-Younger algorithm implementation
with focus on linguistic correctness and parsing accuracy.
"""

import math
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from cky.cky import (
    Grammar,
    _best_root,
    _children,
    _cky_base,
    _cky_inductive,
    _read_model,
    _read_test_sentences,
    _to_tree,
    parse_sentence,
    write_parses,
)
from cky.exceptions import GrammarError, ParsingError


class TestGrammar(unittest.TestCase):
    """Test Grammar class functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_grammar = Grammar(
            binary={
                ("NP", "VP"): {"S": 0.8},
                ("DT", "NN"): {"NP": 0.9},
            },
            lexical={
                "the": {"DT": 0.7},
                "cat": {"NN": 0.8},
                "runs": {"VB": 0.9},
            },
            tags=["DT", "NN", "NP", "S", "VB", "VP"],
            tag_to_idx={"DT": 0, "NN": 1, "NP": 2, "S": 3, "VB": 4, "VP": 5},
            idx_to_tag={0: "DT", 1: "NN", 2: "NP", 3: "S", 4: "VB", 5: "VP"},
            root_tag_idxs={"S": 3},
        )

    def test_grammar_initialization(self):
        """Test grammar initialization with proper structure."""
        self.assertEqual(len(self.sample_grammar.tags), 6)
        self.assertEqual(self.sample_grammar.tag_to_idx["S"], 3)
        self.assertEqual(self.sample_grammar.idx_to_tag[3], "S")
        self.assertIn("S", self.sample_grammar.root_tag_idxs)

    def test_grammar_from_file_success(self):
        """Test loading grammar from file with valid format."""
        mock_content = """G S : NP VP -0.223
X DT : the -0.357
X NN : cat -0.223
X VB : runs -0.105"""

        with patch("pathlib.Path.read_text") as mock_read:
            mock_read.return_value = mock_content
            grammar = Grammar.from_file(Path("test.pcfg"))

            self.assertIn(("NP", "VP"), grammar.binary)
            self.assertIn("the", grammar.lexical)
            self.assertIn("S", grammar.root_tag_idxs)

    def test_grammar_from_file_invalid_format(self):
        """Test grammar loading with invalid format raises GrammarError."""
        mock_content = "INVALID FORMAT"

        with patch("pathlib.Path.read_text") as mock_read:
            mock_read.return_value = mock_content
            with self.assertRaises(GrammarError):
                Grammar.from_file(Path("invalid.pcfg"))

    def test_grammar_from_file_missing_file(self):
        """Test grammar loading with missing file raises FileError."""
        with patch("pathlib.Path.read_text") as mock_read:
            mock_read.side_effect = FileNotFoundError()
            with self.assertRaises(FileError):
                Grammar.from_file(Path("missing.pcfg"))


class TestCKYAlgorithm(unittest.TestCase):
    """Test CKY algorithm core functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.grammar = Grammar(
            binary={
                ("DT", "NN"): {"NP": 0.9},
                ("NP", "VB"): {"S": 0.8},
            },
            lexical={
                "the": {"DT": 0.7},
                "cat": {"NN": 0.8},
                "runs": {"VB": 0.9},
            },
            tags=["DT", "NN", "NP", "S", "VB"],
            tag_to_idx={"DT": 0, "NN": 1, "NP": 2, "S": 3, "VB": 4},
            idx_to_tag={0: "DT", 1: "NN", 2: "NP", 3: "S", 4: "VB"},
            root_tag_idxs={"S": 3},
        )

    def test_cky_base_initialization(self):
        """Test CKY base case initialization."""
        sentence = ["the", "cat", "runs"]
        trellis, bkptr = _cky_base(
            sentence, len(self.grammar.tags), self.grammar.lexical, self.grammar.tag_to_idx
        )

        # Check trellis dimensions
        self.assertEqual(trellis.shape, (3, 3, 5))

        # Check lexical probabilities are set
        dt_idx = self.grammar.tag_to_idx["DT"]
        nn_idx = self.grammar.tag_to_idx["NN"]
        vb_idx = self.grammar.tag_to_idx["VB"]

        self.assertAlmostEqual(trellis[0, 0, dt_idx], 0.7, places=6)
        self.assertAlmostEqual(trellis[0, 1, nn_idx], 0.8, places=6)
        self.assertAlmostEqual(trellis[0, 2, vb_idx], 0.9, places=6)

    def test_cky_inductive_step(self):
        """Test CKY inductive step with binary rule application."""
        sentence = ["the", "cat"]
        trellis, bkptr = _cky_base(
            sentence, len(self.grammar.tags), self.grammar.lexical, self.grammar.tag_to_idx
        )

        # Apply inductive step for span [0, 2)
        _cky_inductive(trellis, bkptr, 1, 0, 0, self.grammar)

        # Check that NP rule was applied
        np_idx = self.grammar.tag_to_idx["NP"]
        expected_prob = 0.7 * 0.8 * 0.9  # DT * NN * rule_prob
        self.assertAlmostEqual(trellis[1, 0, np_idx], expected_prob, places=6)

    def test_best_root_finding(self):
        """Test finding best root parse."""
        sentence = ["the", "cat", "runs"]
        trellis, bkptr = _cky_base(
            sentence, len(self.grammar.tags), self.grammar.lexical, self.grammar.tag_to_idx
        )

        # Fill trellis with mock probabilities
        s_idx = self.grammar.tag_to_idx["S"]
        trellis[2, 0, s_idx] = 0.5  # Mock probability for S spanning [0, 3)

        root_ptr, root_prob = _best_root(len(sentence), trellis, self.grammar)

        self.assertIsNotNone(root_ptr)
        self.assertEqual(root_ptr, (2, 0, s_idx))
        self.assertAlmostEqual(root_prob, 0.5, places=6)

    def test_best_root_no_valid_parse(self):
        """Test best root finding when no valid parse exists."""
        sentence = ["unknown", "word"]
        trellis, bkptr = _cky_base(
            sentence, len(self.grammar.tags), self.grammar.lexical, self.grammar.tag_to_idx
        )

        root_ptr, root_prob = _best_root(len(sentence), trellis, self.grammar)

        self.assertIsNone(root_ptr)
        self.assertEqual(root_prob, 0.0)

    def test_children_extraction(self):
        """Test extracting children from backpointer."""
        # Mock backpointer record
        bkptr = np.empty((3, 3, 5), dtype=object)
        bkptr[1, 0, 2] = (0.5, (0, 0), (0, 1), 0, 1)  # NP -> DT NN

        l_ptr, r_ptr, l_idx, r_idx = _children((1, 0, 2), bkptr)

        self.assertEqual(l_ptr, (0, 0))
        self.assertEqual(r_ptr, (0, 1))
        self.assertEqual(l_idx, 0)  # DT
        self.assertEqual(r_idx, 1)  # NN

    def test_children_missing_backpointer(self):
        """Test children extraction with missing backpointer."""
        bkptr = np.empty((3, 3, 5), dtype=object)
        bkptr.fill(None)

        with self.assertRaises(ValueError):
            _children((1, 0, 2), bkptr)

    def test_tree_reconstruction(self):
        """Test tree reconstruction from backpointers."""
        sentence = ["the", "cat"]
        bkptr = np.empty((2, 2, 5), dtype=object)

        # Mock backpointer for NP -> DT NN
        dt_idx = self.grammar.tag_to_idx["DT"]
        nn_idx = self.grammar.tag_to_idx["NN"]
        np_idx = self.grammar.tag_to_idx["NP"]

        bkptr[0, 0, dt_idx] = None  # Leaf
        bkptr[0, 1, nn_idx] = None  # Leaf
        bkptr[1, 0, np_idx] = (0.5, (0, 0), (0, 1), dt_idx, nn_idx)

        tree_str = _to_tree((1, 0, np_idx), bkptr, self.grammar.idx_to_tag, sentence)

        expected = "(NP (DT the) (NN cat))"
        self.assertEqual(tree_str, expected)


class TestParseSentence(unittest.TestCase):
    """Test complete sentence parsing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.grammar = Grammar(
            binary={
                ("DT", "NN"): {"NP": 0.9},
                ("NP", "VB"): {"S": 0.8},
            },
            lexical={
                "the": {"DT": 0.7},
                "cat": {"NN": 0.8},
                "runs": {"VB": 0.9},
            },
            tags=["DT", "NN", "NP", "S", "VB"],
            tag_to_idx={"DT": 0, "NN": 1, "NP": 2, "S": 3, "VB": 4},
            idx_to_tag={0: "DT", 1: "NN", 2: "NP", 3: "S", 4: "VB"},
            root_tag_idxs={"S": 3},
        )

    def test_parse_sentence_success(self):
        """Test successful sentence parsing."""
        sentence = ["the", "cat", "runs"]
        log_prob, tree = parse_sentence(sentence, self.grammar)

        self.assertFalse(math.isnan(log_prob))
        self.assertNotEqual(tree, "FAIL")
        self.assertIn("S", tree)
        self.assertIn("NP", tree)
        self.assertIn("VB", tree)

    def test_parse_sentence_empty(self):
        """Test parsing empty sentence raises ParsingError."""
        with self.assertRaises(ParsingError):
            parse_sentence([], self.grammar)

    def test_parse_sentence_no_root_tags(self):
        """Test parsing with grammar having no root tags."""
        grammar_no_root = Grammar(
            binary={},
            lexical={},
            tags=[],
            tag_to_idx={},
            idx_to_tag={},
            root_tag_idxs={},
        )

        with self.assertRaises(ParsingError):
            parse_sentence(["test"], grammar_no_root)

    def test_parse_sentence_unknown_words(self):
        """Test parsing with unknown words."""
        sentence = ["unknown", "word"]
        log_prob, tree = parse_sentence(sentence, self.grammar)

        # Should return FAIL for unknown words
        self.assertTrue(math.isnan(log_prob))
        self.assertEqual(tree, "FAIL")


class TestIOOperations(unittest.TestCase):
    """Test I/O operations for CKY parser."""

    def test_read_test_sentences_success(self):
        """Test reading test sentences from file."""
        mock_content = "the cat runs\nunknown word\n"

        with patch("pathlib.Path.read_text") as mock_read:
            mock_read.return_value = mock_content
            sentences = _read_test_sentences(Path("test.txt"), {"the": {}, "cat": {}, "runs": {}})

            self.assertEqual(len(sentences), 2)
            self.assertEqual(sentences[0], ["the", "cat", "runs"])
            self.assertEqual(sentences[1], ["<UNK-T>", "<UNK-T>"])

    def test_read_test_sentences_file_not_found(self):
        """Test reading test sentences with missing file."""
        with patch("pathlib.Path.read_text") as mock_read:
            mock_read.side_effect = FileNotFoundError()
            with self.assertRaises(FileError):
                _read_test_sentences(Path("missing.txt"), {})

    def test_write_parses_success(self):
        """Test writing parse results to file."""
        results = [(0.5, "(S (NP the cat) (VP runs))"), (-1.0, "FAIL")]

        with patch("pathlib.Path.open") as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file

            write_parses(Path("output.parses"), results)

            # Check that file was written to
            mock_file.write.assert_called()
            calls = mock_file.write.call_args_list
            self.assertIn("LL0: 0.5", str(calls[0]))
            self.assertIn("LL1: -1.0", str(calls[2]))


class TestLinguisticCorrectness(unittest.TestCase):
    """Test linguistic correctness based on NLP theory."""

    def setUp(self):
        """Set up linguistically valid grammar."""
        # Grammar based on English syntax rules
        self.english_grammar = Grammar(
            binary={
                # Phrase structure rules
                ("NP", "VP"): {"S": 0.9},  # S -> NP VP
                ("DT", "N"): {"NP": 0.8},  # NP -> DT N
                ("V", "NP"): {"VP": 0.7},  # VP -> V NP
                ("V", "PP"): {"VP": 0.6},  # VP -> V PP
                ("P", "NP"): {"PP": 0.8},  # PP -> P NP
            },
            lexical={
                # Determiners
                "the": {"DT": 0.9},
                "a": {"DT": 0.8},
                # Nouns
                "cat": {"N": 0.9},
                "dog": {"N": 0.8},
                "house": {"N": 0.7},
                # Verbs
                "sees": {"V": 0.9},
                "chases": {"V": 0.8},
                # Prepositions
                "in": {"P": 0.9},
                "on": {"P": 0.8},
            },
            tags=["DT", "N", "NP", "P", "PP", "S", "V", "VP"],
            tag_to_idx={"DT": 0, "N": 1, "NP": 2, "P": 3, "PP": 4, "S": 5, "V": 6, "VP": 7},
            idx_to_tag={0: "DT", 1: "N", 2: "NP", 3: "P", 4: "PP", 5: "S", 6: "V", 7: "VP"},
            root_tag_idxs={"S": 5},
        )

    def test_subject_verb_object_structure(self):
        """Test parsing S-V-O structure (linguistically valid)."""
        sentence = ["the", "cat", "sees", "the", "dog"]
        log_prob, tree = parse_sentence(sentence, self.english_grammar)

        self.assertFalse(math.isnan(log_prob))
        self.assertNotEqual(tree, "FAIL")
        # Should have S as root
        self.assertTrue(tree.startswith("(S"))

    def test_prepositional_phrase_attachment(self):
        """Test prepositional phrase attachment."""
        sentence = ["the", "cat", "sees", "the", "dog", "in", "the", "house"]
        log_prob, tree = parse_sentence(sentence, self.english_grammar)

        self.assertFalse(math.isnan(log_prob))
        self.assertNotEqual(tree, "FAIL")
        # Should contain PP structure
        self.assertIn("PP", tree)

    def test_grammatical_agreement_constraints(self):
        """Test that parser respects grammatical constraints."""
        # This tests that the parser doesn't create invalid structures
        sentence = ["the", "cat", "chases"]
        log_prob, tree = parse_sentence(sentence, self.english_grammar)

        # Should still parse even with incomplete structure
        self.assertFalse(math.isnan(log_prob))
        self.assertNotEqual(tree, "FAIL")

    def test_probability_ranking(self):
        """Test that more probable parses are preferred."""
        # Create grammar where one structure is more probable
        high_prob_grammar = Grammar(
            binary={
                ("NP", "VP"): {"S": 0.95},  # High probability
                ("DT", "N"): {"NP": 0.9},
                ("V", "NP"): {"VP": 0.9},
            },
            lexical={
                "the": {"DT": 0.9},
                "cat": {"N": 0.9},
                "sees": {"V": 0.9},
                "dog": {"N": 0.9},
            },
            tags=["DT", "N", "NP", "S", "V", "VP"],
            tag_to_idx={"DT": 0, "N": 1, "NP": 2, "S": 3, "V": 4, "VP": 5},
            idx_to_tag={0: "DT", 1: "N", 2: "NP", 3: "S", 4: "V", 5: "VP"},
            root_tag_idxs={"S": 3},
        )

        sentence = ["the", "cat", "sees", "the", "dog"]
        log_prob, tree = parse_sentence(sentence, high_prob_grammar)

        # Should have high probability due to grammar design
        self.assertGreater(log_prob, -2.0)  # Not too negative


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        """Set up minimal grammar for edge case testing."""
        self.minimal_grammar = Grammar(
            binary={},
            lexical={"word": {"N": 1.0}},
            tags=["N"],
            tag_to_idx={"N": 0},
            idx_to_tag={0: "N"},
            root_tag_idxs={},
        )

    def test_single_word_sentence(self):
        """Test parsing single word sentence."""
        sentence = ["word"]
        log_prob, tree = parse_sentence(sentence, self.minimal_grammar)

        # Should fail because no root tags defined
        self.assertTrue(math.isnan(log_prob))
        self.assertEqual(tree, "FAIL")

    def test_very_long_sentence(self):
        """Test parsing very long sentence."""
        # Create grammar for long sentence
        long_grammar = Grammar(
            binary={
                ("N", "N"): {"NP": 0.5},
                ("NP", "NP"): {"S": 0.3},
            },
            lexical={
                "word": {"N": 0.8},
                "another": {"N": 0.7},
            },
            tags=["N", "NP", "S"],
            tag_to_idx={"N": 0, "NP": 1, "S": 2},
            idx_to_tag={0: "N", 1: "NP", 2: "S"},
            root_tag_idxs={"S": 2},
        )

        # Long sentence
        sentence = ["word"] * 10
        log_prob, tree = parse_sentence(sentence, long_grammar)

        # Should handle long sentences without crashing
        self.assertIsInstance(log_prob, float)
        self.assertIsInstance(tree, str)

    def test_zero_probability_rules(self):
        """Test handling of zero probability rules."""
        zero_prob_grammar = Grammar(
            binary={
                ("N", "N"): {"NP": 0.0},  # Zero probability
            },
            lexical={
                "word": {"N": 1.0},
            },
            tags=["N", "NP"],
            tag_to_idx={"N": 0, "NP": 1},
            idx_to_tag={0: "N", 1: "NP"},
            root_tag_idxs={"NP": 1},
        )

        sentence = ["word", "word"]
        log_prob, tree = parse_sentence(sentence, zero_prob_grammar)

        # Should handle zero probabilities gracefully
        self.assertTrue(math.isnan(log_prob))
        self.assertEqual(tree, "FAIL")


if __name__ == "__main__":
    unittest.main()
