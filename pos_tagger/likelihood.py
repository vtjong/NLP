import sys
import numpy as np

class Likelihood:
    def __init__(self):
        self.beg_sent = "<s>"
        self.end_sent = "</s>"
        self.emissions = {}
        self.transitions = {}
        self.observations = set()
        self.pos = {}
        self.sequences = []
        self.parse_hmm()
        self.parse_testfile()
        self.alpha = np.zeros((len(self.emissions) + 2, len(self.sequences)))
        self.beta = np.zeros_like(self.alpha)
        self.unker()
        self.forward_likelihood = self.forward()
        self.alpha = self.alpha[1:len(self.alpha)-1]
        self.backward_likelihood = self.backward()
        self.beta = self.beta[1:len(self.beta)-1]
        self.save_it_up() 

    def parse_hmm(self):
        """
        [parse_hmm()] reads in model from model.hmm and initializes data structures 
        containing observations, latent states, emissions, and transitions.
        """
        model_file = sys.argv[1]
        with open(model_file) as file:
            lines = file.read().strip().split("\n")
        for line in lines:
            E_or_T, word_from, word_to, junk, prob = tuple(line.split())

            if E_or_T == "E":
                self.observations.add(word_to)
                self.emissions.setdefault(word_from, {})
                self.emissions[word_from][word_to] = float(prob)
            elif E_or_T == "T":
                self.transitions.setdefault(word_from, {})
                self.transitions[word_from][word_to] = float(prob)

        self.pos = {idx: val for idx, val in enumerate(self.emissions, start=1)}

    def parse_testfile(self):
        """
        [parse_testfile()] reads in testfile and generates a list of test 
        sequences into [self.sequences].
        """
        test_file = sys.argv[2]
        with open(test_file) as test_file:
            self.sequences = test_file.read().strip().split()

    def unker(self):
        """
        [unker()] unks words not in self.observations
        """
        self.sequences = [
            word if word in self.observations else "<unk>"
            for word in self.sequences
        ]

    def forward_beg(self, counter, word):
        """
        [forward_beg(counter, word)] is a helper function that handles the beg 
        case of forward logic for the forward-backward algorithm.
        """
        for idx in self.pos.keys():
            key = self.pos[idx]
            emission = 0
            if word in self.emissions[key].keys():
                emission = self.emissions[key][word]
                self.alpha[idx, counter] = emission * self.transitions[self.beg_sent][key]

    def forward_gen(self, counter, word):
        """
        [forward_gen(counter, word)] is a helper function that handles the general 
        case of forward logic for the forward-backward algorithm.
        """
        for idx in self.pos.keys():
            state = self.pos[idx]
            var = 0
            for prev_idx in self.pos.keys():
                prev_state = self.pos[prev_idx]
                emission = 0
                if word in self.emissions[state].keys():
                    emission = self.emissions[state][word] 
                var += self.alpha[prev_idx, counter-1] * self.transitions[prev_state][state]
                self.alpha[idx, counter] = var * emission

    def forward(self):
        """
        [forward()] handles the forward logic for the forward-backward algorithm 
        and calculates likehoods in alpha. 
        """
        counter = 0
        for word in self.sequences:
            # beg states
            if counter == 0:
                self.forward_beg(0, word)
            # general case states
            else:
                self.forward_gen(counter, word)
            counter += 1
        counter -= 1

        # end case states
        alpha_end = np.array([
            self.alpha[idx, counter] * self.transitions[self.pos[idx]][self.end_sent]
            for idx in range(1, len(self.pos))
        ])

        self.alpha[0, counter] = np.sum(alpha_end)
        return np.log(self.alpha[0, counter])
    

    def backward_gen(self, counter, prev_word):
        """
        [backward_gen(counter, prev_word)] is a helper function that handles the 
        general case of backward logic for the forward-backward algorithm.
        """
        for idx in self.pos.keys():
            state = self.pos[idx]
            var = 0
            for prev_idx in self.pos.keys():
                prev_state = self.pos[prev_idx]
                emission = 0
                if prev_word in self.emissions[prev_state].keys():
                    emission = self.emissions[prev_state][prev_word] 
                var += (
                    emission *
                    self.transitions[state][prev_state] *
                    self.beta[prev_idx, counter + 1]
                )
            self.beta[idx, counter] = var

    def backward(self):
        """
        [backward()] handles the backward logic for the forward-backward 
        algorithm and calculates likehoods in beta. 
        """
        init_counter = self.beta.shape[1] - 1
        counter = init_counter
        self.sequences.reverse()
        for word in self.sequences:
            # end case states
            if counter == init_counter:
                for idx in self.pos.keys():
                    self.beta[idx, counter] += self.transitions[self.pos[idx]][self.end_sent]
            # general case
            elif counter != init_counter:
                self.backward_gen(counter, prev_word)
            prev_word = word
            counter -= 1

        # beg states
        for index in self.pos.keys():
            key = self.pos[index]
            if prev_word in self.emissions[key]:
                emission = self.emissions[key][prev_word]
            else:
                emission = 0
            last_row = self.beta.shape[0] - 1
            beta_val = self.transitions["<s>"][key] * emission * self.beta[index, 0]
            self.beta[last_row, 0] += beta_val
        return np.log(self.beta[last_row, 0])

    def save_it_up(self):
        """
        [save_it_up()] dumps all stats in hmm.likelihood.
        """
        with open("hmm.likelihood", "w") as f:
            f.write("Forward: " + "\n" + str(self.forward_likelihood) +
                    "\n" + str(self.alpha) + "\n")
            f.write("Backward: " + "\n" + str(self.backward_likelihood) + 
                    "\n" + str(self.beta) + "\n")

def main():
    Likelihood()

if __name__ == '__main__':
    main()

