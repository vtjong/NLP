import sys
import numpy as np

class Decoding:
    def __init__(self):
        self.beg_sent = "<s>"
        self.end_sent = "</s>"
        self.emissions = {}
        self.transitions = {}
        self.observations = set()
        self.sequences = []
        self.best_path = []
        self.parse_hmm()
        self.parse_testfile()
        self.unker()
        self.trellis = np.zeros((len(self.pos) + 2, len(self.sequences)))
        self.backtrace = np.zeros_like(self.trellis)
        self.log_likelihood = 0
        self.viterbi()
        self.save_it_up() 

    def parse_hmm(self):
        """
        [parse_hmm()] reads in model from model.hmm and initializes data 
        structures containing observations, pos, emissions, and transitions.
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

        # self.pos = {idx: val for (idx, val) in zip(list(range(1, len(self.emissions) + 1)), list(self.emissions.keys()))}
        self.pos = {idx: key for idx, key in enumerate(self.emissions, start=1)}

    def parse_testfile(self):
        """
        [parse_testfile()] reads in testfile and generates a list of test 
        sequences into [self.test_seq_list].
        """
        test_file = sys.argv[2]
        with open(test_file) as test_file:
            self.sequences = test_file.read().strip().split()

    def unker(self):
        """
        [unker()] unks words not in self.observations
        """
        self.sequences = [word if word in self.observations
                            else "<unk>" for word in self.sequences]

    def viterbi_verifier(self, *args):
        prev_idx, prev_state, curr_state, counter, emission, trellis, bestpathpointer = args
        contender = (
            self.trellis[prev_idx, counter - 1] *
            self.transitions[prev_state][curr_state] *
            emission
        )
        cond = contender > trellis
        trellis = contender if cond else trellis
        bestpathpointer = int(prev_idx) if cond else bestpathpointer
        return trellis, bestpathpointer
    
    def viterbi_beg(self, word, count = 0):
        """
        [viterbi_beg(word, count = 0)] handles beg state logic for viterbi.
        """
        for idx in self.pos.keys():
            pos = self.pos[idx]
            emission = 0
            if word in self.emissions[pos].keys():
                emission = self.emissions[pos][word]
            self.trellis[idx, count] = emission * self.transitions[self.beg_sent][pos]

    def viterbi_gen(self, *args):
        """
        [viterbi_gen()] handles the core logic for finding best state path.
        - Need to loop over all previous counter emission values to fill current 
        emission values
        """
        word, curr_state, counter, trellis, bestpathpointer = args

        for prev_idx in self.pos.keys():   
            prev_state = self.pos[prev_idx]
            emission = 0
            if word in self.emissions[curr_state].keys():
                emission = self.emissions[curr_state][word]
            trellis, bestpathpointer = self.viterbi_verifier(prev_idx, 
                                                            prev_state, 
                                                            curr_state, 
                                                            counter, 
                                                            emission, 
                                                            trellis, 
                                                            bestpathpointer)
        return trellis, bestpathpointer
        
    def viterbi_end(self, counter):
        trellis = -1     
        bestpathpointer = 0   
        for idx in self.pos.keys():
            pos = self.pos[idx]
            trellis, bestpathpointer = self.viterbi_verifier(
                idx,
                pos,
                self.end_sent,
                counter,
                1,
                trellis,
                bestpathpointer)
        return trellis, bestpathpointer

    def viterbi(self):
        """
        [viterbi()] uses viterbi algorithm to determine best sequence--
        the sequence with the best LL. 
        - Each cell of the trellis, v_t(j), represents the probability that the 
        HMM is in state j after seeing the first t observations and passing 
        through the most probable state sequence q1,...,qt−1, given HMM λ = (A, B). 
        The value of each cell vt(j) is computed by recursively taking the most 
        probable path that could lead us to this cell. 
        - The algorithm returns the state path through the HMM that assigns 
        maximum likelihood to the observation sequence.

        - [counter] represents time stamp t_i of our iterations
        """
        counter = 0
        for word in self.sequences:
            # beg states
            if counter == 0:
                self.viterbi_beg(word)
            # general case states
            else:
                for curr_idx in self.pos.keys():  # get emission (is obs in label)
                    curr_state = self.pos[curr_idx]
                    temp = self.viterbi_gen(word, curr_state, counter, -1, 0)
                    self.trellis[curr_idx, counter] = temp[0]
                    self.backtrace[curr_idx, counter] = temp[1]
            counter += 1
        
        # end state (end of sentence emissions)
        trellis, bestpathpointer = self.viterbi_end(counter)
        counter -= 1
        self.trellis[0, counter] = trellis
        self.backtrace[0, counter] = bestpathpointer
        self.backtrace.astype(int)

        # calculate log likelihood of best path
        self.log_likelihood = np.log(self.trellis[0, counter])

        # backtrace using backpointers to get best path
        best_sequence = [self.pos[int(bestpathpointer)]]
        num_cols = len(self.sequences)
        for i in range(num_cols - 1, 0, -1):
            bestpathpointer = int(self.backtrace[bestpathpointer, i])
            best_sequence.append(self.pos[bestpathpointer])
        best_sequence.reverse()
        self.best_path = best_sequence
        
    def save_it_up(self):
        """
        [save_it_up()] dumps all stats in hmm.decoding.
        """
        with open("hmm.decoding", "w") as file:
            s = ""
            for i in range(len(self.best_path)):
                s += " " + self.best_path[i]
            file.write("Best Sequence :" + s + "\n")
            file.write("Best LL : " + str(self.log_likelihood))

def main():
    dc = Decoding()

if __name__ == '__main__':
    main()