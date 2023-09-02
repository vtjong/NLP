import sys
import re
import numpy as np
import itertools
from tree import Tree

class MLE:
    def __init__(self):
        self.G = {}
        self.X = {}
        self.tags = set()
        self.vocab = set()
        self.parse_file()
        self.save_it_up() 
    
    def update_counts(self, *args):
        dict, key = args
        dict.setdefault(key, 0)
        dict[key] += 1

    def update_pair(self, *args):
        dict, key, val = args
        dict.setdefault(key, {})
        self.update_counts(dict[key], val)

    def update_probs(self, dict):
        total_counts = np.sum(np.array(list(dict.values())))
        dict = {key: np.log(float(val)/total_counts) for (key, val) in dict.items()}
        return dict
    
    def matrix_probs(self, dict):
        for key in list(dict.keys()):
            dict[key] = self.update_probs(dict[key])
        
    def update_tags(self, tags, tag, train, terminal):
        if terminal == "terminal":
            unk_val = "<UNK-T>"
        elif terminal== "preterminal": 
            unk_val = "<UNK-NT>^X"
            tag = tag + "^X"
        else:
            unk_val = "<UNK-NT>"
        if train:
            tags.add(tag)
        else: 
            tag = tag if tag in tags else unk_val

        return tag


    def is_unary(self, node):
        children = node.ch
        number_children = len(children)
        if number_children != 1: return 0

        child = children[0]
        grand_children = child.ch
        number_grand_children = len(grand_children)
        if number_grand_children == 0: return 0
        return 1


    def traverse(self, tree, curr_tag, unitary_tag, train):
        '''
        [traverse()] traverses through tree [tree] with string tag [curr_tag] and builds up grammar G and X lists recursively. If [train] is true, then [traverse] adds lexeme to self.vocab, otherwise it replaces it with "<UNK-T>".
        '''

        # curr_tag = curr_tag if '+' or "UNK" in curr_tag else tree.c
        if unitary_tag != "":
            curr_tag = unitary_tag + "+" + curr_tag

        children = tree.ch
        num_children = len(children)

        # unary, terminal, preterminal
        if num_children == 1:
            child = children[0]
            child_tag = child.c

            # unary level
            if self.is_unary(tree):
                return self.traverse(child, child_tag, curr_tag, train)

            # preterminal
            curr_tag = self.update_tags(self.tags, curr_tag, train, "preterminal")
            child_tag = self.update_tags(self.vocab, child_tag, train, "terminal")
            self.update_pair(self.X, curr_tag, child_tag)
            return curr_tag

        # nonterminal
        if num_children == 2:
            curr_tag = self.update_tags(self.tags, curr_tag, train, "nonterminal")
            left_tag = self.traverse(children[0], children[0].c, "", train)
            right_tag = self.traverse(children[1], children[1].c, "", train)
            self.update_pair(self.G, curr_tag, left_tag + " " + right_tag)
            return curr_tag

        return ""


    def tree_traverse(self, tree_list, train):
        '''
        [tree_traverse()] takes care of the tree reading logic and calls traverse for each tree, corresponding to a sentence. 
        '''
        for tree_str in tree_list:
            tree = Tree()   
            tree.read(tree_str)
            self.traverse(tree, tree.c, "", train)

            
    def parse_file(self):
        '''
        [parse_file()] 

        - Read in the entire file and then replace the newline characters with spaces by using string replace or regex
        - Split on the "(ROOT" and store each one in a list
            - But you have to prepend that in bcuz splitting on "(ROOT" gets rid of it and we still need that
        - tree.Tree.read() returns an empty string, but you need to first create a tree object and then 
        '''  
        
        # Read in trees 
        input_file, split_val = sys.argv[1], float(sys.argv[2])

        # Parse trees and recurse
        with open(input_file) as file:
            entire_file = file.read().replace("\n", " ").split("(ROOT ")
            file_list = ["(ROOT " + tree for tree in entire_file if tree!= ""]
            train_list = file_list[:int(len(file_list) * split_val)]
            test_list =  file_list[int(len(file_list) * split_val):]

            self.tree_traverse(train_list, True)
            self.tree_traverse(test_list, False)

            self.matrix_probs(self.G)
            self.matrix_probs(self.X)
            
    def checker(self):
        print(self.G)
        print(self.X)
    
    def save_it_up(self):
        """
        [save_it_up()] dumps all stats in model.pcfg.
        """
        with open("model.pcfg", "w") as file:
            for tag_from in list(self.G.keys()):
                for tag_to in self.G[tag_from].keys():
                    file.write("G " + tag_from + " : " + tag_to + " " +
                            str(self.G[tag_from][tag_to]))
                    file.write("\n")

            for transition in self.X.keys():
                for transition_key in self.X[transition].keys():
                    file.write("X " + transition + " : " + transition_key + " " + str(self.X[transition][transition_key]))
                    file.write("\n")
        
def main():
    MLE()

if __name__ == '__main__':
    main()
