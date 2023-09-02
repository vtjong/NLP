import sys
import numpy as np

class TreeMaker:
    def __init__(self, idx_pos, tree_roots_idx):
        self.idx_pos = idx_pos
        self.tree_roots_idx = tree_roots_idx

    def get_best_root(self, *args):
        """
        [get_best_root()] computes the most likely root after viterbi is completed and the associated probability [max]. If one cannot be found, outputs [fail_flag] indicating as such. 
        """
        n, trellis = args
        best_root_pr, best_root_idx = 0, str()
        fail_flag = True
        for root, root_idx in self.tree_roots_idx.items():
            contender_pr = trellis[-1][0][root_idx]
            if contender_pr!= 0:
                fail_flag = False
                if contender_pr > best_root_pr: # Update new best
                    best_root_pr, best_root_idx = contender_pr , root_idx
        root = (n - 1, 0, best_root_idx)
        return root, best_root_pr, fail_flag
    
    def get_children(self, ptr, tag, bkptr):
        """
        [get_children()] gets left and right children info.
        """
        children = bkptr[ptr[0], ptr[1], tag]
        return children[1], children[2], children[3], children[-1]
    
    def tree_traverse(self, ptr, tag, bkptr, sentence):
        """
        [tree_traverse()] traverses through all nodes and builds up trees recursively.
        """
        i, j = ptr[0], ptr[1]
        tree = "(" + self.idx_pos[tag]

        # Base case
        if i == 0: return tree + " " + str(sentence[j]) + ")"

        # Recursively get subtrees
        lt_ptr, rt_ptr, lt_lab, rt_lab = self.get_children(ptr, tag, bkptr)
        lt_tree = self.tree_traverse(lt_ptr, lt_lab, bkptr, sentence)
        rt_tree = self.tree_traverse(rt_ptr, rt_lab, bkptr, sentence)
        tree += " " + lt_tree + " " + rt_tree + ")"

        return tree

    def tree_handler(self, bkptr, root_info, sentence):
        """
        [tree_handler()] handles all the backtracing, tree building logic for given a tree with [root], accessing children through [bckptr].
        """
        n, root_pos = len(sentence), self.idx_pos[root_info[2]]

        # Begin tree
        tree = "(" + root_pos

        # Single node case
        if n<=1: return tree + " " + str(sentence[0]) + ")"

        # General case
        lt_ptr, rt_ptr, lt_lab, rt_lab = self.get_children(root_info[:2], root_info[2], bkptr)
        root_left_tree, root_right_tree = self.tree_traverse(lt_ptr, lt_lab, bkptr, sentence), self.tree_traverse(rt_ptr, rt_lab, bkptr, sentence)
        tree += " " + root_left_tree + " " + root_right_tree + ")"

        return tree

class CKY:
    def __init__(self):
        self.pos = set()
        self.vocab = set()
        self.word_pr = {}
        self.pos_pr = {}
        self.parse_pcfg()
        self.num_tags = len(self.pos)
        self.parse_testfile()
        self.treemaker = TreeMaker(self.idx_pos, self.tree_roots_idx)
        self.test_prs = [self.cky(sentence) for sentence in self.sentences]
        self.save_it_up() 
    
    def make_dict(self, *args):
        dict, key1, key2, val = args
        dict.setdefault(key1, {})
        dict[key1][key2] = val

    def cky_base(self, sentence):
        """
        [cky_base()] initializes cky data structs and fills in base probabilities.
        """
        num_words = len(sentence)
        trellis = np.zeros((num_words, num_words, self.num_tags)) # chart contains probs from lower trig mat
        bckptr = np.empty_like(trellis, dtype=object)
        bckptr.fill((-1, (-1, -1), (-1, -1), -1, -1))   

        # Iterate through all sentences and test words 
        for i in range(num_words):
            word = sentence[i]
            for key in self.word_pr[word]:
                key_ind = self.pos_idx[key]
                trellis[0, i, key_ind] = self.word_pr[word][key]
        return trellis, bckptr

    def cky_ind(self, trellis, bckptr, *args):
        """
        [cky_ind()] conducts the inductive step for a given sentence.
        """
        # We are going to use log probabilities so we should add them
        # When we use log probabilities, they are all negative, but the less negative the better, so we still compare by doing if contender > current one
        i,j,k = args
        row, col = i - k - 1, j + k + 1
        
        for (left_ch, right_ch) in self.pos_pr:
            left_idx, right_idx = self.pos_idx[left_ch], self.pos_idx[right_ch]
            for node, node_pr in self.pos_pr[(left_ch, right_ch)].items():
                # node_pr = self.pos_pr[(left_ch, right_ch)][node]
                left_pr, right_pr = trellis[k, j, left_idx], trellis[row, col, right_idx]
                contender_pr = node_pr * left_pr * right_pr
                
                # Now we have two conditions to check
                #   (1) valid prob left_prob > 0 and right_prob > 0 (bcuz we initialize to -1)
                #   (2) contender prob is better than OPT 
                #       - if so, change OPT to it
                #       - assign backpointer to its previous one
                node_idx = self.pos_idx[node]
                valid_prob = left_pr > 0 and right_pr > 0
                contender_is_sup = contender_pr > trellis[i, j, node_idx] 

                if (valid_prob and contender_is_sup):
                    trellis[i, j, node_idx] = contender_pr
                    bckptr[i, j, node_idx] = (contender_pr, (k, j), (row, col),left_idx, right_idx) 

    def cky(self, sentence):
        """
        [cky()] performs the CKY algorithm, using the lower triangle method.
        """
        # Base step
        trellis, bkptr = self.cky_base(sentence)
        num_words = len(sentence)

        # Inductive step
        for i in range(1, num_words):
            for j in range(num_words - i):
                for k in range(i):
                    self.cky_ind(trellis, bkptr, i, j, k)
        
        # Handle logic to find root
        root, root_pr, fail_flag = self.treemaker.get_best_root(num_words, trellis)

        # Retrace sequence if one exists, otherwise "FAIL" and "nan"
        if fail_flag: 
            tree, root_pr = "FAIL", "nan"
        else: 
            tree = self.treemaker.tree_handler(bkptr, root, sentence)
        
        tree_pr = np.log(float(root_pr))
        return tree_pr, tree
            
    def parse_pcfg(self):
        '''
        [parse_pcfg()] reads in grammar file and generates structures to store pos tags, vocab, and probabilities of both. 
        '''  
        input_file = sys.argv[1]
        with open(input_file, "r") as file:
            file = file.read().strip().split("\n")
        for line in file:
            line = line.split(" ")
            if line[0] == "G":
                node, left_child, right_child, log_prob = line[1], line[3], line[4], np.exp(float(line[-1]))
                self.pos.update((node, left_child, right_child))
                self.make_dict(self.pos_pr, (left_child, right_child), node, log_prob)
            elif line[0] == "X":
                preterm, term, log_prob = line[1], line[3], np.exp(float(line[-1]))
                self.pos.add(preterm)
                self.make_dict(self.word_pr, term, preterm, log_prob)
        self.pos = list(self.pos)
        self.pos_idx = {key: idx for (key, idx) in zip(self.pos, list(range(len(self.pos))))}
        self.idx_pos = {idx:key for (key, idx) in self.pos_idx.items()}
        self.tree_roots_idx = {pos:idx for (pos,idx) in self.pos_idx.items() if "ROOT" in pos and "|" not in pos}
        self.tree_roots = list(self.tree_roots_idx.keys())

    def parse_testfile(self):
        """
        [parse_testfile()] reads in testfile and generates a list of test sentences, saved in [self.sentences].
        """
        test_file = sys.argv[2]
        with open(test_file) as test_file:
            sentences = test_file.read().strip().split("\n")

        self.sentences = [[word if word in self.word_pr else "<UNK-T>" for word in sentence.split(" ")] for sentence in sentences]

    def save_it_up(self):
        """
        [save_it_up()] dumps all trees and stats into output.parses.
        """
        with open("output.parses", "w") as out_file:
            count = len(self.test_prs)
            print(count)
            for idx in range(count):
                pr = str(self.test_prs[idx][0])
                print(pr)
                tree = self.test_prs[idx][1]
                out_file.write("LL" + str(idx) + ": " + pr + "\n")
                out_file.write(tree + "\n")

def main():
    CKY()

if __name__ == '__main__':
    main()
