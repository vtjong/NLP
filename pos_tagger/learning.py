import sys
import numpy as np

from likelihood import Likelihood

class Learning:
    def __init__(self):
        self.epochs = 3
        self.beg_sent = "<s>"
        self.end_sent = "</s>"
        self.emissions, self.transitions, self.pos_tags, self.vocab, self.vocab_set = self.parse_hmm()
        self.sequences = self.parse_testfile()
        self.num_seqs = len(self.sequences)
        self.forward_backward()
        self.save_it_up() 

    def parse_hmm(self):
        """
        [parse_hmm()] reads in model.hmm and generates initial emission and 
        tranmission matrices based on the indices of the pos tags and vocab labels. 
        """
        model_file = sys.argv[1]
        with open(model_file) as file:
            lines = file.read().strip().split("\n")

        # Collect pos tags and vocabs into pos_set and vocab_set
        pos_set = {line.split()[1] for line in lines if line.split()[0] == "E"}
        vocab_set = {line.split()[2] for line in lines if line.split()[0] == "E"}
        # Create pos_tags_to_idx and vocab_to_idx dicts by mapping tags/vocab to idx values
        num_tags, num_vocab = len(pos_set), len(vocab_set)
        self.num_tags, self.num_vocab = num_tags, num_vocab 
        pos_tags_to_idx = {tag: idx for (tag, idx) in zip(list(pos_set), list(range(num_tags)))}
        vocab_to_idx = {vocab: idx for (vocab, idx) in zip(list(vocab_set), list(range(num_vocab)))}

        # Use pos_tags and vocab dicts to create word-key'd emission and 
        # transition matrices with idx values; note that number of vocab words 
        # is also the total time (T) in textbook formulas
        emissions = np.zeros((num_tags, num_vocab))
        transitions = np.zeros((num_tags + 2, num_tags + 2))

        for line in lines:
            E_or_T, pos_tag, vocab, junk, prob = tuple(line.split())
            if E_or_T == "E":
                emissions[pos_tags_to_idx[pos_tag], vocab_to_idx[vocab]] = float(prob)
            elif E_or_T == "T":
                if pos_tag == self.beg_sent:
                    pos_tag_idx = 0
                elif pos_tag == self.end_sent:
                    pos_tag_idx = num_tags + 1
                else:
                    pos_tag_idx = pos_tags_to_idx[pos_tag] + 1
                
                if vocab == self.beg_sent:
                    vocab_idx = 0
                elif vocab == self.end_sent:
                    vocab_idx = num_tags + 1
                else:
                    vocab_idx = pos_tags_to_idx[vocab] + 1
            # elif E_or_T == "T":
            #     pos_tag_idx = 0 if pos_tag== self.beg_sent else (num_tags + 1 if pos_tag == self.end_sent else pos_tags_to_idx[pos_tag] + 1)
            #     vocab_idx = 0 if vocab==self.beg_sent else (num_tags + 1 if vocab == self.end_sent else pos_tags_to_idx[vocab] + 1)
            transitions[pos_tag_idx, vocab_idx] = float(prob)

        return emissions, transitions, pos_tags_to_idx, vocab_to_idx, vocab_set

    def unker(self, sequences):
        """
        [unker()] unks words not in vocabulary set [self.vocab_set].
        """
        return [[word if word in self.vocab_set else "<unk>" for word in sequence] for sequence in sequences]
    
    def parse_testfile(self):
        """
        [parse_testfile()] reads in testfile and generates a list of test 
        sequences into [self.sequences], which is then unked by [self.unker(sequences)].
        Note that each sequence is a list of string observations, so 
        [self.sequences] is a list of lists. 
        """
        test_file = sys.argv[2]
        with open(test_file) as test_file:
            sequences = test_file.read().strip().split("\n")
        sequences = [[sequence.split(" ")][0] for sequence in sequences]
        sequences = self.unker(sequences)
        return sequences
    
    def forward(self, sequence, seq_idx, T_total):
        """
        [forward(sequence, seq_idx, T_total)] updates the [seq_idx]th entry in 
        alpha matrix [self.alpha_history] and calculates log likelihood prob for 
        a particular sequence number [seq_idx].
        """
        # Initialize Step: t= 0 case
        t = 0
        alpha = np.zeros((self.num_tags + 2, T_total))
        test_word = str(sequence[t])
        test_word_idx = self.vocab[test_word]
        alpha[1 : self.num_tags + 1, t] = np.array([
            self.transitions[0, j + 1] * self.emissions[j, test_word_idx]
            for j in range(self.num_tags)
        ])

        # Recursive Step
        for t in range(1, T_total):
            test_word = str(sequence[t])
            test_word_idx = self.vocab[test_word]
            for j in range(self.num_tags):
                for i in range(self.num_tags):
                    alpha[j + 1, t] += (
                        alpha[i + 1, t - 1] *
                        self.transitions[i + 1, j + 1] *
                        self.emissions[j, test_word_idx]
                    )

        # Termination Step: t = T_total case
        prob = 0
        for i in range(self.num_tags):
            val = alpha[i+1, T_total-1] * self.transitions[i+1, -1]
            alpha[-1, T_total-1] += val
            prob += val
        
        # Update global probabilities and alpha matrix
        self.loglikelihood += np.log(prob)
        self.alpha_history.append(alpha)
        self.alpha_history_sums.append(np.sum(alpha))

    def backward(self, sequence, seq_idx, T_total):
        """
        [backward(sequence, seq_idx, T_total)] updates beta matrix the 
        [seq_idx]th entry in [self.beta_history].
        """
        # Initialize Step: same shape as beta at time stamp seq_idx; t= T_total case
        t = -1 
        beta = np.zeros((self.num_tags + 2, T_total))
        beta[1 : self.num_tags+1, -1] = self.transitions[1 : self.num_tags+1,-1]

        # Recursive Step
        t_rec_init = T_total - 2
        for t in range(t_rec_init, -1, -1):
            test_word = str(sequence[t+1])
            test_word_idx = self.vocab[test_word]
            for j in range(self.num_tags):
                for i in range(self.num_tags):
                    beta[i + 1, t] += self.transitions[i + 1, j + 1] * self.emissions[j, test_word_idx] * beta[j + 1, t + 1]

        # Termination Step: t = 0
        t = 0
        test_word_idx = self.vocab[str(sequence[t])]
        for j in range(self.num_tags):
            beta[0,0] += beta[j + 1, 0] * self.transitions[0, j + 1] * self.emissions[j, test_word_idx]  

        # Update global beta matrix  
        self.beta_history.append(beta)
        self.beta_history_sums.append(np.sum(beta))

    def gamma(self, sequence, seq_idx, T_total):
        """
        [gamma(sequence, seq_idx, T_total)] performs the E-step in the algorithm, 
        updating the [seq_idx]th entry in the [self.gamma_history] matrix. 
        """    
        alpha, beta =  self.alpha_history[seq_idx], self.beta_history[seq_idx]
        gamma = np.zeros((self.num_tags, T_total))

        for t in range(T_total):
            for i in range(1, self.num_tags+1):
                    gamma[i - 1, t] = alpha[i, t] * beta[i, t] / alpha[-1, -1]
        
        # Update global gamma history matrix
        self.gamma_history.append(gamma)
        self.gamma_history_sums.append(np.sum(gamma))

    def tau(self, sequence, seq_idx, T_total):
        """
        [tau(sequence, seq_idx, T_total)] performs the E-step in the algorithm, 
        updating the [seq_idx]th entry in the [self.tau_history] matrix. 
        """    
        alpha, beta = self.alpha_history[seq_idx], self.beta_history[seq_idx]
        tau = np.zeros((T_total + 1, self.num_tags + 2, self.num_tags + 2))
        denom = alpha[-1, -1]
        tot_tags = self.num_tags + 2
        seq_size = T_total
        obs_seq = sequence
        T = self.transitions
        E = self.emissions
        t = 0
        while t < T_total + 1:
            if t < T_total:
                ob = (
                    self.vocab[str(obs_seq[t])]
                    if str(obs_seq[t]) in self.vocab
                    else self.vocab["<unk>"]
                )
            for i in range(tot_tags):
                for j in range(tot_tags):
                    T_val = T[i, j]
                    if t == 0:
                        if i == 0:
                            alpha_val = 1
                        else:
                            alpha_val = 0
                    else:
                        alpha_val = alpha[i, t - 1]

                    if t == seq_size:
                        if j == tot_tags - 1:
                            E_val = 1
                            beta_val = 1
                        else:
                            E_val = 0
                            beta_val = 0
                    else:
                        E_val = E[j - 1, ob] if j > 0 and j <= len(E) else 0
                        beta_val = beta[j, t]
                    numer = T_val * E_val * alpha_val * beta_val
                    tau[t, i, j] = numer / denom
            t += 1
        
        # Update global tau matrix  
        self.tau_history.append(tau)
        self.tau_history_sums.append(np.sum(tau))
    
    def row_norm(self, mat):
        """
        [row_norm()] conducts row-wise normalization for a given matrix [mat].
        """
        sum_of_rows = np.array([x if x!= 0 else np.inf for x in mat.sum(axis=1)])
        mat /= sum_of_rows[:, np.newaxis]
        return mat

    def M_step(self):
        """
        [M_step()] executes the M_step update for the emission and transition matrices. 
        """
        idx_vocab_dict = {self.vocab[k] : k for k in self.vocab}
        temp_t = np.zeros_like(self.transitions)
        temp_e = np.zeros_like(self.emissions)
        for i in range(self.num_tags + 2):
            for j in range(self.num_tags + 2):
                numer, denom = 0, 0
                for seq_idx in range(self.num_seqs):
                    tau = self.tau_history[seq_idx]
                    for t in range(len(tau)):
                        numer += tau[t, i, j]
                        for word_idx in range(len(tau[t, i])):
                            denom += tau[t, i, word_idx]
                temp_t[i, j] = numer / denom if denom != 0 else 0

            # Exclude beg and end sentence cases
            if i != 0 and i != self.num_tags + 1:
                for k in range(len(self.emissions[i - 1])):
                    vocab_word = str(idx_vocab_dict[k])
                    numer, denom = 0, 0
                    for seq_idx in range(self.num_seqs):
                        sequence = self.sequences[seq_idx]
                        gamma = self.gamma_history[seq_idx]
                        for t in range(len(gamma[i - 1])):
                            vocab_t = str(sequence[t])
                            if vocab_t == vocab_word:
                                numer += gamma[i - 1, t]
                            elif vocab_word == "<unk>" and vocab_t not in self.vocab:
                                numer += gamma[i - 1, t]
                            denom += gamma[i - 1, t]
                    temp_e[i - 1, k] = numer / denom if denom != 0 else 0
       
        return self.row_norm(temp_e), self.row_norm(temp_t)
        # print("emissions", np.sum(self.emissions))
        # print("transitions", np.sum(self.transitions))

    def forward_backward(self):
        """
        [forward_backward()] performs Baum-Welch, a special class of EM 
        (expectation-maximization) algorithms. 
        """
        # Initialize alphas, betas, gammas, and taus across time stamps
        self.alpha_history, self.beta_history = [], []
        self.alpha_history_sums, self.beta_history_sums = [], []

        self.tau_history, self.gamma_history = [], []
        self.tau_history_sums, self.gamma_history_sums = [], []

        self.loglikelihood = 0

        # Training for self.epochs number of iterations
        for epoch in range(self.epochs + 1):
            self.loglikelihood = 0
            if epoch != 0: print("Iteration: ", epoch) 
            for seq_idx in range(self.num_seqs):
                sequence = self.sequences[seq_idx]
                T_total = len(sequence)
                self.forward(sequence, seq_idx, T_total)
                self.backward(sequence, seq_idx, T_total)
                self.gamma(sequence, seq_idx, T_total)
                self.tau(sequence, seq_idx, T_total)
            # print(self.alpha_history)
            if epoch == 3:
                self.save_it_up()
            else:
                self.emissions, self.transitions = self.M_step()
        print("self.alpha_history_sum", self.alpha_history_sums)
        print("self.beta_history_sum", self.beta_history_sums)
        print("self.gamma_history_sum", self.gamma_history_sums)
        print("self.tau_history_sum", self.tau_history_sums)


    def save_it_up(self):
        flipped_tags = dict([(v + 1, k) for k, v in self.pos_tags.items()])
        flipped_tags[0] = "<s>"
        flipped_tags[len(flipped_tags)] = "</s>"
        flipped_labels = dict([(v, k) for k, v in self.vocab.items()])

        E_rows, E_cols = self.emissions.shape
        T_rows, T_cols = self.transitions.shape
        with open("hmm_params.learning", "w") as f:
            for i in range(E_rows):
                for j in range(E_cols):
                    f.write(
                        "E " + flipped_tags[i+1] + " " + flipped_labels[j] +
                        " : " + str(self.emissions[i][j]) + "\n"
                    )
            for i in range(T_rows - 1):
                for j in range(1, T_cols):
                    f.write(
                        "T " + flipped_tags[i] + " " + flipped_tags[j] +
                        " : " + str(self.transitions[i][j]) + "\n"
                    )

def main():
    ln = Learning()

if __name__ == '__main__':
    main()
