import time
import math
import sys
import argparse
import cPickle as pickle
import copy
import os
import codecs
import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
from CharRNN import CharRNN, make_initial_state

RNN_DATA_DIR = os.environ.get("RNN_DATA_DIR")
RNN_TRAINING_FILE = os.environ.get("RNN_TRAINING_FILE")
RNN_CHECKPOINT_DIR = os.environ.get("RNN_CHECKPOINT_DIR")
RNN_GPU = int(os.environ.get("RNN_GPU"))
RNN_RNN_SIZE = int(os.environ.get("RNN_RNN_SIZE"))
RNN_LEARNING_RATE = float(os.environ.get("RNN_LEARNING_RATE"))
RNN_LEARNING_RATE_DECAY = float(os.environ.get("RNN_LEARNING_RATE_DECAY"))
RNN_LEARNING_RATE_DECAY_AFTER = int(os.environ.get("RNN_LEARNING_RATE_DECAY_AFTER"))
RNN_DECAY_RATE = float(os.environ.get("RNN_DECAY_RATE"))
RNN_DROPOUT = float(os.environ.get("RNN_DROPOUT"))
RNN_SEQ_LENGTH = int(os.environ.get("RNN_SEQ_LENGTH"))
RNN_BATCHSIZE = int(os.environ.get("RNN_BATCHSIZE"))
RNN_EPOCHS = int(os.environ.get("RNN_EPOCHS"))
RNN_GRAD_CLIP = int(os.environ.get("RNN_GRAD_CLIP"))
RNN_INIT_FROM = str(os.environ.get("RNN_INIT_FROM"))



# Determine paths
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__)) # Chainer path
MODEL_PATH = CURRENT_PATH + "/Models/" + os.path.splitext(os.path.basename(RNN_TRAINING_FILE))[0] + "/" # Model path
CHECKPOINT_PATH = MODEL_PATH + "checkpoints/"
VOCAB_PATH = MODEL_PATH + "vocabulary.bin"


print("--- Path Summary ---")
print("Chainer Path: " + CURRENT_PATH)
print("Model Path: " + MODEL_PATH)
print("Checkpoint Path: " + CHECKPOINT_PATH)
print("--------------------")

# Create directories
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
if not os.path.exists(CHECKPOINT_PATH):
    os.mkdir(CHECKPOINT_PATH)


# Load input data
def load_data():
    vocab = {}
    words = codecs.open(RNN_TRAINING_FILE, 'rb', 'utf-8').read()
    words = list(words)
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset[i] = vocab[word]
    print 'corpus length:', len(words)
    print 'vocab size:', len(vocab)
    return dataset, words, vocab


def do_checkpoint(model, epoch):
    model_copy = copy.deepcopy(model).to_cpu()

    pickle.dump(model_copy, open(CHECKPOINT_PATH + "CharRNN-Epoch-%s.model" % epoch, 'wb'))
    pickle.dump(model_copy, open(MODEL_PATH + "CharRNN-Latest.model", 'wb'))





train_data, words, vocab = load_data()
pickle.dump(vocab, open(VOCAB_PATH, 'wb'))

if len(RNN_INIT_FROM) > 0:
    model = pickle.load(open(RNN_INIT_FROM, 'rb'))
else:
    model = CharRNN(len(vocab), RNN_RNN_SIZE)

if RNN_GPU >= 0:
    cuda.get_device(RNN_GPU).use()
    model.to_gpu()

optimizer = optimizers.RMSprop(lr=RNN_LEARNING_RATE, alpha=RNN_DECAY_RATE, eps=1e-8)
optimizer.setup(model)

whole_len    = train_data.shape[0]
jump         = whole_len / RNN_BATCHSIZE
epoch        = 0
start_at     = time.time()
cur_at       = start_at
state        = make_initial_state(RNN_RNN_SIZE, batchsize=RNN_BATCHSIZE)
if RNN_GPU >= 0:
    accum_loss   = Variable(cuda.zeros(()))
    for key, value in state.items():
        value.data = cuda.to_gpu(value.data)
else:
    accum_loss   = Variable(np.zeros((), dtype=np.float32))

print 'going to train {} iterations'.format(jump * RNN_EPOCHS)
for i in xrange(jump * RNN_EPOCHS):

    start = time.time()
    x_batch = np.array([train_data[(jump * j + i) % whole_len]
                        for j in xrange(RNN_BATCHSIZE)])
    y_batch = np.array([train_data[(jump * j + i + 1) % whole_len]
                        for j in xrange(RNN_BATCHSIZE)])
    print(time.time() - start)

    if RNN_GPU >=0:
        x_batch = cuda.to_gpu(x_batch)
        y_batch = cuda.to_gpu(y_batch)

    state, loss_i = model.forward_one_step(x_batch, y_batch, state, dropout_ratio=RNN_DROPOUT)
    accum_loss   += loss_i

    if (i + 1) % RNN_SEQ_LENGTH == 0:  # Run truncated BPTT
        now = time.time()
        print '{}/{}, train_loss = {}, time = {:.2f}'.format((i+1) / RNN_SEQ_LENGTH, jump, accum_loss.data / RNN_SEQ_LENGTH, now-cur_at)
        cur_at = now

        optimizer.zero_grads()
        accum_loss.backward()
        accum_loss.unchain_backward()  # truncate
        if RNN_GPU >= 0:
            accum_loss = Variable(cuda.zeros(()))
        else:
            accum_loss = Variable(np.zeros((), dtype=np.float32))

        optimizer.clip_grads(RNN_GRAD_CLIP)
        optimizer.update()

    # Save model to disk for checkpoints
    if (i + 1) % 10000 == 0:
        do_checkpoint(model, float(i)/jump)

    if (i + 1) % jump == 0:
        epoch += 1

        # Save model
        do_checkpoint(model, epoch)

        if epoch >= RNN_LEARNING_RATE_DECAY_AFTER:
            optimizer.lr *= RNN_LEARNING_RATE_DECAY
            print 'decayed learning rate by a factor {} to {}'.format(RNN_LEARNING_RATE_DECAY, optimizer.lr)

    sys.stdout.flush()