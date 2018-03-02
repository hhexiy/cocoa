import numpy as np
import random
from itertools import izip_longest, izip
from cocoa.core.util import read_pickle, write_pickle
import pdb

class DialogueBatcher(object):
    def __init__(self, vocab, split_type, shuffle=True):
        self.vocab = vocab
        self.shuffle = shuffle
        self.data = self.create_batches(split_type)
        self.num_per_epoch = len(self.data)

    def create_batches(self, split_type):
        raw_data = read_pickle("data/{}_batches.pkl".format(split_type))
        data = []
        for example in raw_data:
            source_tokens = example[0]
            target_tokens = example[1]

            source = [self.vocab.word_to_ind[t] for t in source_tokens]
            target = [self.vocab.word_to_ind[t] for t in target_tokens]
            data.append((source, target))
        return data

    def get_batch(self):
        return random.choice(self.data)

