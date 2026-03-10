import numpy as np
import re
from collections import Counter
import random

class Word2VecSGNS:
    """
    Word2Vec Skip-gram with Negative Sampling (SGNS)
    """
    def __init__(self, text, embedding_dim, lr=0.025, min_count=1, neg_samples=5,
                 window_size=2, subsample_t=1e-3):
        """
        Initializes the Word2Vec model.

        Args:
            text (str): Raw input text
            embedding_dim (int): Dimension of word embeddings
            lr (float): Initial learning rate
            min_count (int): Minimum frequency of words to include in vocabulary
            neg_samples (int): Number of negative samples per positive pair
            window_size (int): Maximum skip-gram window size
            subsample_t (float): Threshold for subsampling frequent words
        """
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.min_count = min_count
        self.neg_samples = neg_samples
        self.window_size = window_size

        # text processing
        self.tokens = self.tokenize(text)
        # vocabulary and subsampling
        self.build_vocab(subsample_t)
        # unigram distribution
        self.build_unigram_table()

        # embeddings initialization
        self.vocab_size = len(self.word2idx)
        self.limit = 0.5 / self.embedding_dim
        self.W_in = np.random.uniform(low=-self.limit, high=self.limit,
                                      size=(self.vocab_size, self.embedding_dim))
        self.W_out = np.zeros((self.vocab_size, self.embedding_dim))

    # ------------------------------- Vocabulary -------------------------------
    @staticmethod
    def tokenize(text):
        """
        separates words from the text
        """
        text = text.lower()
        text = text.replace("-", " ").replace("—", " ")
        text = re.sub(r"[^a-z\s]", "", text)
        return text.split()

    def build_vocab(self, subsample_t=1e-3):
        """
        generates vocabulary
        """
        counter = Counter(self.tokens)
        initial_vocab = list(counter.keys())

        # subsampling frequent words
        initial_freqs = np.array([counter[w] for w in initial_vocab], dtype=np.float64)
        total_freqs = initial_freqs.sum()
        rel_freqs = initial_freqs / total_freqs
        drop_probs = {w: max(0.0, 1 - np.sqrt(subsample_t / rel_freqs[i]))
                      for i, w in enumerate(initial_vocab)}
        self.tokens = [w for w in self.tokens if random.random() > drop_probs[w]
                       and counter[w] >= self.min_count] # discarding rare words

        counter = Counter(self.tokens)
        self.vocab = list(counter.keys())
        self.freqs = np.array(list(counter.values()), dtype=np.float64)
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}


    # ---------------------------- Negative Sampling ---------------------------
    def build_unigram_table(self, table_size=1_000_000):
        """
        Precomputes a unigram table for negative sampling.
        """
        pow_freqs = self.freqs ** 0.75
        probs = pow_freqs / pow_freqs.sum()
        counts = np.round(probs * table_size).astype(int)
        table = []
        for idx, c in enumerate(counts):
            table += [idx] * c
        self.unigram_table = np.array(table)

    def sample_negatives(self, center_idx, context_idx):
        """
        Samples negative word indices for a given positive pair
        """
        samples = np.random.choice(self.unigram_table, size=self.neg_samples*2)
        negative_ids = [s for s in samples if s != center_idx and s != context_idx]
        return negative_ids[:self.neg_samples]


    # -------------------------------- Training --------------------------------
    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -10, 10)
        return 1 / (1 + np.exp(-x))

    def train_pair(self, center_idx, context_idx, neg_ids, lr):
        """
        Forward pass, backprop, and parameter update for a single (center, context) pair.
        """
        loss = 0.0

        # positive sample
        v_ce = self.W_in[center_idx]
        u_co = self.W_out[context_idx]

        score_pos = np.dot(u_co, v_ce)
        sigma_pos = self.sigmoid(score_pos)
        grad_pos = sigma_pos - 1

        grad_v_ce = grad_pos * u_co
        grad_u_co = grad_pos * v_ce

        loss += -np.log(sigma_pos + 1e-10)

        # negative samples
        u_ns = self.W_out[neg_ids]

        score_neg = np.dot(u_ns, v_ce)
        sigma_neg = self.sigmoid(score_neg)
        grad_neg = sigma_neg

        grad_v_ce += np.sum((grad_neg[:, None] * u_ns), axis=0)
        grad_u_ne = grad_neg[:, None] * v_ce

        loss += -np.sum(np.log(1 - sigma_neg + 1e-10))

        self.W_out[context_idx] -= lr * grad_u_co
        self.W_in[center_idx] -= lr * grad_v_ce
        self.W_out[neg_ids] -= lr * grad_u_ne

        return loss

    def train(self, epochs=5):
        """
        Trains the model over multiple epochs.
        """
        total_words = len(self.tokens) * epochs
        processed_words = 0

        for epoch in range(epochs):
            np.random.shuffle(self.tokens)
            total_loss = 0.0
            lr = self.lr
            for i, center_word in enumerate(self.tokens):
                # learning rate decay
                lr = max(self.lr * (1 - processed_words / total_words), self.lr * 0.0001)
                center_idx = self.word2idx[center_word]

                # dynamic window generation
                current_window = np.random.randint(1, self.window_size + 1)
                for j in range(-current_window, current_window + 1):
                    if j == 0:
                        continue

                    context_pos = i + j
                    if 0 <= context_pos < len(self.tokens):
                        context_word = self.tokens[context_pos]
                        context_idx = self.word2idx[context_word]
                        negative_ids = self.sample_negatives(center_idx, context_idx)

                        loss = self.train_pair(center_idx, context_idx, negative_ids, lr)
                        total_loss += loss

                processed_words += 1

            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")



    # ------------------------------ Utilities ---------------------------------
    def contains(self, word):
        return word in self.vocab

    def get_embedding(self, word):
        idx = self.word2idx[word]
        return self.W_in[idx]

    def get_embedding_matrix(self):
        return self.W_in

    def get_vocab(self):
        return self.vocab