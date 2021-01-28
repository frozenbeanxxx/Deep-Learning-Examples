
import os
import torch
import torch.nn as nn 
import numpy as np 
from torch.nn.utils import clip_grad_norm_

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    def __len__(self):
        return len(self.word2idx)

class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()
    def get_data(self, path, batch_size=20):
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
        # Tokenize the file content
        ids = torch.LongTensor(tokens)
        token = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        num_batches = ids.size(0) // batch_size
        ids = ids[:num_batches*batch_size]
        return ids.view(batch_size, -1)

def language_model():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    embed_size = 128
    hidden_size = 1024
    num_layers = 1
    num_epochs = 5
    num_samples = 1000     # number of words to be sampled
    batch_size = 20
    seq_length = 30
    learning_rate = 0.002

    # Load "Penn Treebank" dataset
    corpus = Corpus()
    ids = corpus.get_data('data/train.txt', batch_size)
    vocab_size = len(corpus.dictionary)
    num_batches = ids.size(1) // seq_length

    class RNNLM(nn.Module):
        def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
            super(RNNLM, self).__init__()
            self.embed = nn.Embedding(vocab_size, embed_size)
            self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_size, vocab_size)
        def forward(self, x, h):
            x = self.embed(x)
            out, (h, c) = self.lstm(x, h)
            out = out.reshape(out.size(0)*out.size(1), out.size(2))
            out = self.linear(out)
            return out, (h, c)
    model = RNNLM(vocab_size, embed_size, hidden_size, num_layers).to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)




