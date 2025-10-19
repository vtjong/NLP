"""Unit tests for MLE training module following TDD principles and NLP theory.

This module tests Maximum Likelihood Estimation for PCFG training
with focus on linguistic rule extraction and probability estimation.
"""

import math
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from cky.exceptions import FileError, TrainingError
from cky.mle import (
    ConfigurationManager,
    ProbabilityCalculator,
    RuleCounter,
    _read_trees_file,
    train_pcfg,
    write_model_pcfg,
)
from cky.tree import Tree


class TestConfigurationManager(unittest.TestCase):
    """Test configuration management."""

    def test_unk_token_mapping(self):
        """Test UNK token mapping for different types."""
        self.assertEqual(ConfigurationManager.get_unk_token("terminal"), "<UNK-T>")
        self.assertEqual(ConfigurationManager.get_unk_token("preterminal"), "<UNK-NT>^X")
        self.assertEqual(ConfigurationManager.get_unk_token("nonterminal"), "<UNK-NT>")
        self.assertEqual(ConfigurationManager.get_unk_token("unknown_type"), "<UNK-NT>")

    def test_default_values(self):
        """Test default configuration values."""
        self.assertEqual(ConfigurationManager.DEFAULT_TRAIN_SPLIT, 0.9)
        self.assertEqual(ConfigurationManager.DEFAULT_OUTPUT_FILE, "output.parses")
        self.assertEqual(ConfigurationManager.DEFAULT_MODEL_FILE, "model.pcfg")
        self.assertEqual(ConfigurationManager.DEFAULT_ENCODING, "utf-8")


class TestRuleCounter(unittest.TestCase):
    """Test rule counting functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.rule_counter = RuleCounter()

        # Sample trees for testing
        self.simple_tree = Tree.from_string("(S (NP (DT the) (NN cat)) (VP (VB runs)))")
        self.unary_tree = Tree.from_string("(S (NP (DT the) (NN cat)))")
        self.binary_tree = Tree.from_string("(S (NP the) (VP runs))")

    def test_count_rules_training_mode(self):
        """Test rule counting in training mode."""
        trees = [self.simple_tree]
        G, X = self.rule_counter.count_rules(trees, train_mode=True)

        # Should have binary rules
        self.assertIn("S", G)
        self.assertIn("NP", G)
        self.assertIn("VP", G)

        # Should have lexical rules
        self.assertIn("DT", X)
        self.assertIn("NN", X)
        self.assertIn("VB", X)

    def test_count_rules_heldout_mode(self):
        """Test rule counting in heldout mode with UNK handling."""
        # First train on some trees
        train_trees = [self.simple_tree]
        self.rule_counter.count_rules(train_trees, train_mode=True)

        # Then process heldout with unknown words
        heldout_tree = Tree.from_string("(S (NP (DT the) (NN unknown_word)))")
        G, X = self.rule_counter.count_rules([heldout_tree], train_mode=False)

        # Should map unknown words to UNK
        self.assertIn("<UNK-T>", X.get("NN", {}))

    def test_unary_chain_collapsing(self):
        """Test that unary chains are properly collapsed."""
        # Tree with unary chain: S -> NP -> DT
        unary_tree = Tree.from_string("(S (NP (DT the)))")
        G, X = self.rule_counter.count_rules([unary_tree], train_mode=True)

        # Should collapse unary chains
        self.assertIn("NP+DT", X)

    def test_binary_rule_extraction(self):
        """Test extraction of binary rules."""
        binary_tree = Tree.from_string("(S (NP the) (VP runs))")
        G, X = self.rule_counter.count_rules([binary_tree], train_mode=True)

        # Should extract binary rule S -> NP VP
        self.assertIn("S", G)
        self.assertIn("NP VP", G["S"])

    def test_lexical_rule_extraction(self):
        """Test extraction of lexical rules."""
        lexical_tree = Tree.from_string("(S (NP (DT the) (NN cat)))")
        G, X = self.rule_counter.count_rules([lexical_tree], train_mode=True)

        # Should extract lexical rules
        self.assertIn("DT", X)
        self.assertIn("NN", X)
        self.assertIn("the", X["DT"])
        self.assertIn("cat", X["NN"])


class TestProbabilityCalculator(unittest.TestCase):
    """Test probability calculation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.calculator = ProbabilityCalculator()

    def test_calculate_log_probabilities_success(self):
        """Test successful probability calculation."""
        rule_counts = {
            "S": {"NP VP": 10, "VP NP": 5},
            "NP": {"DT NN": 20, "NN": 5},
        }

        log_probs = self.calculator.calculate_log_probabilities(rule_counts)

        # Check S probabilities
        s_probs = log_probs["S"]
        self.assertAlmostEqual(s_probs["NP VP"], math.log(10 / 15), places=6)
        self.assertAlmostEqual(s_probs["VP NP"], math.log(5 / 15), places=6)

        # Check NP probabilities
        np_probs = log_probs["NP"]
        self.assertAlmostEqual(np_probs["DT NN"], math.log(20 / 25), places=6)
        self.assertAlmostEqual(np_probs["NN"], math.log(5 / 25), places=6)

    def test_calculate_log_probabilities_zero_count(self):
        """Test probability calculation with zero count rule."""
        rule_counts = {
            "S": {"NP VP": 0},  # Zero count
        }

        with self.assertRaises(TrainingError):
            self.calculator.calculate_log_probabilities(rule_counts)

    def test_probability_normalization(self):
        """Test that probabilities sum to 1.0."""
        rule_counts = {
            "S": {"NP VP": 3, "VP NP": 2},
        }

        log_probs = self.calculator.calculate_log_probabilities(rule_counts)
        s_probs = log_probs["S"]

        # Convert back to probabilities and check normalization
        total_prob = sum(math.exp(p) for p in s_probs.values())
        self.assertAlmostEqual(total_prob, 1.0, places=6)


class TestTrainPCFG(unittest.TestCase):
    """Test PCFG training functionality."""

    def test_train_pcfg_success(self):
        """Test successful PCFG training."""
        trees = [
            "(S (NP (DT the) (NN cat)) (VP (VB runs)))",
            "(S (NP (DT a) (NN dog)) (VP (VB barks)))",
        ]

        G, X = train_pcfg(trees, split=0.8)

        # Should have binary rules
        self.assertIsInstance(G, dict)
        self.assertGreater(len(G), 0)

        # Should have lexical rules
        self.assertIsInstance(X, dict)
        self.assertGreater(len(X), 0)

    def test_train_pcfg_empty_trees(self):
        """Test training with empty tree list."""
        with self.assertRaises(TrainingError):
            train_pcfg([], split=0.9)

    def test_train_pcfg_invalid_split(self):
        """Test training with invalid split ratio."""
        trees = ["(S (NP the) (VP runs))"]

        with self.assertRaises(TrainingError):
            train_pcfg(trees, split=0.0)  # Invalid split

        with self.assertRaises(TrainingError):
            train_pcfg(trees, split=1.0)  # Invalid split

    def test_train_pcfg_malformed_trees(self):
        """Test training with malformed tree strings."""
        trees = ["(S (NP the) (VP runs))", "malformed tree"]

        with self.assertRaises(TrainingError):
            train_pcfg(trees, split=0.9)

    def test_train_heldout_split(self):
        """Test training with heldout split for UNK handling."""
        trees = [
            "(S (NP (DT the) (NN cat)) (VP (VB runs)))",
            "(S (NP (DT a) (NN dog)) (VP (VB barks)))",
            "(S (NP (DT the) (NN unknown_word)) (VP (VB runs)))",
        ]

        G, X = train_pcfg(trees, split=0.7)  # 2 train, 1 heldout

        # Should handle unknown words in heldout
        self.assertIsInstance(G, dict)
        self.assertIsInstance(X, dict)


class TestLinguisticRuleExtraction(unittest.TestCase):
    """Test linguistic rule extraction based on NLP theory."""

    def test_phrase_structure_rules(self):
        """Test extraction of phrase structure rules."""
        trees = [
            "(S (NP (DT the) (NN cat)) (VP (VB runs)))",
            "(S (NP (DT a) (NN dog)) (VP (VB barks)))",
        ]

        G, X = train_pcfg(trees, split=1.0)

        # Should extract S -> NP VP rule
        self.assertIn("S", G)
        self.assertIn("NP VP", G["S"])

        # Should extract NP -> DT NN rule
        self.assertIn("NP", G)
        self.assertIn("DT NN", G["NP"])

    def test_lexical_categories(self):
        """Test extraction of lexical categories."""
        trees = [
            "(S (NP (DT the) (NN cat)) (VP (VB runs)))",
            "(S (NP (DT a) (NN dog)) (VP (VB barks)))",
        ]

        G, X = train_pcfg(trees, split=1.0)

        # Should extract lexical rules for each category
        self.assertIn("DT", X)
        self.assertIn("NN", X)
        self.assertIn("VB", X)

        # Should have correct word-tag mappings
        self.assertIn("the", X["DT"])
        self.assertIn("cat", X["NN"])
        self.assertIn("runs", X["VB"])

    def test_probability_estimation(self):
        """Test probability estimation from counts."""
        trees = [
            "(S (NP (DT the) (NN cat)) (VP (VB runs)))",
            "(S (NP (DT the) (NN dog)) (VP (VB runs)))",
            "(S (NP (DT a) (NN cat)) (VP (VB barks)))",
        ]

        G, X = train_pcfg(trees, split=1.0)

        # DT should have higher probability for "the" (2/3) vs "a" (1/3)
        dt_probs = X["DT"]
        self.assertGreater(dt_probs["the"], dt_probs["a"])

        # NN should have equal probability for "cat" and "dog" (1/2 each)
        nn_probs = X["NN"]
        self.assertAlmostEqual(nn_probs["cat"], nn_probs["dog"], places=6)

    def test_unary_chain_handling(self):
        """Test handling of unary chains in linguistic trees."""
        trees = [
            "(S (NP (DT the) (NN cat)))",  # S -> NP
            "(S (VP (VB runs)))",  # S -> VP
        ]

        G, X = train_pcfg(trees, split=1.0)

        # Should collapse unary chains
        self.assertIn("S", G)
        # Should have S -> NP and S -> VP rules
        self.assertIn("NP", G["S"])
        self.assertIn("VP", G["S"])


class TestIOOperations(unittest.TestCase):
    """Test I/O operations for MLE training."""

    def test_read_trees_file_success(self):
        """Test reading trees from file."""
        mock_content = "(S (NP the cat) (VP runs)) (S (NP a dog) (VP barks))"

        with patch("pathlib.Path.read_text") as mock_read:
            mock_read.return_value = mock_content
            trees = _read_trees_file(Path("trees.txt"))

            self.assertEqual(len(trees), 2)
            self.assertTrue(trees[0].startswith("(S"))
            self.assertTrue(trees[1].startswith("(S"))

    def test_read_trees_file_not_found(self):
        """Test reading trees from missing file."""
        with patch("pathlib.Path.read_text") as mock_read:
            mock_read.side_effect = FileNotFoundError()
            with self.assertRaises(FileError):
                _read_trees_file(Path("missing.txt"))

    def test_write_model_pcfg_success(self):
        """Test writing PCFG model to file."""
        G = {"S": {"NP VP": -0.5}}
        X = {"DT": {"the": -0.3}}

        with patch("pathlib.Path.write_text") as mock_write:
            write_model_pcfg(G, X, Path("model.pcfg"))

            # Check that file was written
            mock_write.assert_called_once()
            content = mock_write.call_args[0][0]
            self.assertIn("G S : NP VP", content)
            self.assertIn("X DT : the", content)

    def test_write_model_pcfg_write_error(self):
        """Test writing PCFG model with write error."""
        G = {"S": {"NP VP": -0.5}}
        X = {"DT": {"the": -0.3}}

        with patch("pathlib.Path.write_text") as mock_write:
            mock_write.side_effect = OSError("Write failed")
            with self.assertRaises(FileError):
                write_model_pcfg(G, X, Path("model.pcfg"))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_single_tree_training(self):
        """Test training with single tree."""
        trees = ["(S (NP the) (VP runs))"]

        G, X = train_pcfg(trees, split=1.0)

        # Should handle single tree
        self.assertIsInstance(G, dict)
        self.assertIsInstance(X, dict)

    def test_very_long_trees(self):
        """Test training with very long trees."""
        long_tree = "(S " + " ".join(["(NP word)"] * 20) + ")"
        trees = [long_tree]

        G, X = train_pcfg(trees, split=1.0)

        # Should handle long trees without crashing
        self.assertIsInstance(G, dict)
        self.assertIsInstance(X, dict)

    def test_empty_tree_strings(self):
        """Test training with empty tree strings."""
        trees = ["", "   ", "(S (NP the) (VP runs))"]

        with self.assertRaises(TrainingError):
            train_pcfg(trees, split=0.9)

    def test_malformed_tree_structures(self):
        """Test training with malformed tree structures."""
        trees = [
            "(S (NP the) (VP runs))",  # Valid
            "(S (NP the",  # Malformed - missing closing paren
        ]

        with self.assertRaises(TrainingError):
            train_pcfg(trees, split=0.9)


if __name__ == "__main__":
    unittest.main()
