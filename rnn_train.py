#! /usr/bin/env python
from rnn_utils import *
from datetime import datetime
from rnn_gru_theano import GRUTheano
from rnn_gru_theano import RNNTheano

LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
VOCABULARY_SIZE = int(os.environ.get("VOCABULARY_SIZE", "2000"))
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "48"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "128"))
NEPOCH = int(os.environ.get("NEPOCH", "20"))
MODEL_OUTPUT_FILE = os.environ.get("MODEL_OUTPUT_FILE")
INPUT_DATA_FILE = os.environ.get("INPUT_DATA_FILE", "./data/input.txt")
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "30"))
RNN_MODEL = os.environ.get("RNN_MODEL", "RNNTheano")


MODELS_PATH = os.environ.get("MODELS_PATH", "./Models")

print("-----------------Training Parameters-----------------")
print("LEARNING_RATE: {0}".format(LEARNING_RATE))
print("VOCABULARY_SIZE: {0}".format(VOCABULARY_SIZE))
print("EMBEDDING_DIM: {0}".format(EMBEDDING_DIM))
print("HIDDEN_DIM: {0}".format(HIDDEN_DIM))
print("NEPOCH: {0}".format(NEPOCH))
print("MODEL_OUTPUT_FILE: {0}".format(MODEL_OUTPUT_FILE))
print("INPUT_DATA_FILE: {0}".format(INPUT_DATA_FILE))
print("PRINT_EVERY: {0}".format(PRINT_EVERY))
print("MODELS_PATH: {0}".format(MODELS_PATH))
print("RNN_MODEL: {0}".format(RNN_MODEL))
print("-----------------------------------------------------")


# if no model file is set, create new
if not MODEL_OUTPUT_FILE:

  ts = datetime.now().strftime("%Y-%m-%d-%H-%M")

  MODEL_FILE_NAME =  "%s-%s-%s-%s-%s" % (RNN_MODEL, VOCABULARY_SIZE, EMBEDDING_DIM, HIDDEN_DIM, ts)
  MODEL_OUTPUT_FILE = "%s/%s.dat" % (MODELS_PATH, MODEL_FILE_NAME,)
  print("Creating new model: {0}".format(MODEL_OUTPUT_FILE))

  # Load data
  x_train, y_train, word_to_index, index_to_word = load_data(INPUT_DATA_FILE, VOCABULARY_SIZE)

  # Build model
  if RNN_MODEL == "GRUTheano":
    model = GRUTheano(VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM, bptt_truncate=-1)
  elif RNN_MODEL == "RNNTheano":
    model = RNNTheano(VOCABULARY_SIZE, hidden_dim=HIDDEN_DIM, bptt_truncate=4)

else:
  print("Using existing model: {0}".format(MODEL_OUTPUT_FILE))

  # Load data
  x_train, y_train, word_to_index, index_to_word = load_data(INPUT_DATA_FILE, VOCABULARY_SIZE)

  # Load existing model
  if RNN_MODEL == "GRUTheano":
    model = load_model_parameters_theano(MODEL_OUTPUT_FILE, modelClass=GRUTheano)
  elif RNN_MODEL == "RNNTheano":
    model = load_model_parameters_theano(MODEL_OUTPUT_FILE, modelClass=RNNTheano)

# Benchmark SGD step time
t1 = time.time()
model.sgd_step(x_train[10], y_train[10], LEARNING_RATE)
t2 = time.time()
print("SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.))
sys.stdout.flush()


# We do this every few examples to understand what's going on
def sgd_callback(model, num_examples_seen):
  dt = datetime.now().isoformat()
  loss = model.calculate_loss(x_train[:10000], y_train[:10000])
  print("\n%s (%d)" % (dt, num_examples_seen))
  print("--------------------------------------------------")
  print("Loss: %f" % loss)
  generate_sentences(model, 10, index_to_word, word_to_index)
  save_model_parameters_theano(model, MODEL_OUTPUT_FILE)
  print("\n")
  sys.stdout.flush()

for epoch in range(NEPOCH):
  train_with_sgd(model, x_train, y_train, learning_rate=LEARNING_RATE, nepoch=1, decay=0.9, 
    callback_every=PRINT_EVERY, callback=sgd_callback)

