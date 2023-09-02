import sys
import numpy as np
import itertools 

class MLE:
    def __init__(self):
        self.emissions = {}
        self.transitions = {}
        self.mle()
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
        dict = {key: float(val)/total_counts for (key, val) in dict.items()}
        return dict
    
    def unking(self, dict):
        for key in set(dict.keys()):
            self.update_counts(dict[key], "<unk>")
    
    def add_one_smoothing(self, *args):
        dict, transitions_keys = args
        for key in set(dict.keys()):
            for transition in transitions_keys:
                self.update_counts(dict[key], transition)
    
    def matrix_probs(self):
        for key in list(self.emissions.keys()):
            self.emissions[key] = self.update_probs(self.emissions[key])
        for key in list(self.transitions.keys()):
            self.transitions[key] = self.update_probs(self.transitions[key])

    def mle(self):
        '''
        [mle()] reads in all CL arguments and generates matrices containing emission and transition probabilities, handling unking, add-one smoothing, and probability calculations.
        '''
        input_file = sys.argv[1]
        with open(input_file) as file:
            lines = file.read().strip().split("\n")
            transitions_keys = set()
            beg_sent = "<s>"
            end_sent = "</s>"
            
            for line in lines:
                states = line.split(";")
                prev_state_key = beg_sent  # start of sentence

                for state in states:
                    state = (state.split()[0], state.split()[1])
                    curr_state_key = state[0]
                    curr_state_val = state[1]
                    transitions_keys.add(curr_state_key)
                    self.update_pair(self.emissions, curr_state_key, curr_state_val)
                    self.update_pair(self.transitions, prev_state_key, curr_state_key)
                    prev_state_key = curr_state_key
                
                # end of sentence
                transitions_keys.add(end_sent) 
                self.update_pair(self.transitions, prev_state_key, end_sent) 
            
            self.unking(self.emissions)
            self.add_one_smoothing(self.transitions, transitions_keys)
            self.matrix_probs()
        
    def save_it_up(self):
        """
        [save_it_up()] dumps all stats in model.hmm.
        """
        with open("model.hmm", "w") as file:
            for emission in list(self.emissions.keys()):
                for emission_key in self.emissions[emission].keys():
                    file.write("E " + emission + " " + emission_key + " : " +
                            str(self.emissions[emission][emission_key]))
                    file.write("\n")
                            
            for transition in self.transitions.keys():
                for transition_key in self.transitions[transition].keys():
                    file.write("T " + transition + " " + transition_key + " : " + str(self.transitions[transition][transition_key]))
                    file.write("\n")

def main():
    MLE()

if __name__ == '__main__':
    main()
