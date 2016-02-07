#%%
import time
import math
import sys
import argparse
import cPickle as pickle
import codecs
import os

import numpy as np
from chainer import cuda, Variable, FunctionSet
from CharRNN import CharRNN, make_initial_state

sys.stdout = codecs.getwriter('utf_8')(sys.stdout)


RNN_MODEL = os.environ.get("RNN_MODEL")
RNN_MODEL_FILENAME = os.environ.get("RNN_MODEL_FILENAME")
RNN_VOCABULARY_FILENAME = os.environ.get("RNN_VOCABULARY_FILENAME")
RNN_SEED = int(os.environ.get("RNN_SEED"))
RNN_SAMPLE = int(os.environ.get("RNN_SAMPLE"))
RNN_PRIMETEXT = os.environ.get("RNN_PRIMETEXT")
RNN_LENGTH = int(os.environ.get("RNN_LENGTH"))
RNN_GPU = int(os.environ.get("RNN_GPU", 0))




CURRENT_PATH = os.path.dirname(os.path.abspath(__file__)) # Chainer path
MODEL_PATH = RNN_MODEL + "/"
MODEL_FILE_PATH = MODEL_PATH + RNN_MODEL_FILENAME
CHECKPOINT_PATH = MODEL_PATH + "checkpoints/"
VOCAB_PATH = MODEL_PATH + RNN_VOCABULARY_FILENAME

print("--- Path Summary ---")
print("Chainer Path: " + CURRENT_PATH)
print("Model Path: " + MODEL_PATH)
print("Checkpoint Path: " + CHECKPOINT_PATH)
print("--------------------")

def run():



    np.random.seed(RNN_SEED)

    # load vocabulary
    vocab = pickle.load(open(VOCAB_PATH, 'rb'))
    ivocab = {}
    for c, i in vocab.items():
        ivocab[i] = c

    # load model
    model = pickle.load(open(MODEL_FILE_PATH, 'rb'))
    n_units = model.embed.W.data.shape[1]

    if RNN_GPU >= 0:
        cuda.get_device(RNN_GPU).use()
        model.to_gpu()

    # initialize generator
    state = make_initial_state(n_units, batchsize=1, train=False)
    if RNN_GPU >= 0:
        for key, value in state.items():
            value.data = cuda.to_gpu(value.data)

    prev_char = np.array([0], dtype=np.int32)
    if RNN_GPU >= 0:
        prev_char = cuda.to_gpu(prev_char)

    if len(RNN_PRIMETEXT) > 0:
        for i in unicode(RNN_PRIMETEXT, 'utf-8'):
            sys.stdout.write(i)
            prev_char = np.ones((1,), dtype=np.int32) * vocab[i]
            if RNN_GPU >= 0:
                prev_char = cuda.to_gpu(prev_char)

            state, prob = model.forward_one_step(prev_char, prev_char, state, train=False)

    for i in xrange(RNN_LENGTH):
        state, prob = model.forward_one_step(prev_char, prev_char, state, train=False)

        if RNN_SAMPLE > 0:
            probability = cuda.to_cpu(prob.data)[0].astype(np.float64)
            probability /= np.sum(probability)
            index = np.random.choice(range(len(probability)), p=probability)
        else:
            index = np.argmax(cuda.to_cpu(prob.data))
        sys.stdout.write(ivocab[index])

        prev_char = np.array([index], dtype=np.int32)
        if RNN_GPU >= 0:
            prev_char = cuda.to_gpu(prev_char)

    print