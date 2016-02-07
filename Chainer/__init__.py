__author__ = 'Per-Arne'

class Trainer:


    def __init__(self):
        self.RNN_DATA_DIR = os.environ.get("RNN_DATA_DIR")
        self.RNN_TRAINING_FILE = os.environ.get("RNN_TRAINING_FILE")
        self.RNN_CHECKPOINT_DIR = os.environ.get("RNN_CHECKPOINT_DIR")
        self.RNN_GPU = int(os.environ.get("RNN_GPU"))
        self.RNN_RNN_SIZE = int(os.environ.get("RNN_RNN_SIZE"))
        self.RNN_LEARNING_RATE = float(os.environ.get("RNN_LEARNING_RATE"))
        self.RNN_LEARNING_RATE_DECAY = float(os.environ.get("RNN_LEARNING_RATE_DECAY"))
        self.RNN_LEARNING_RATE_DECAY_AFTER = int(os.environ.get("RNN_LEARNING_RATE_DECAY_AFTER"))
        self.RNN_DECAY_RATE = float(os.environ.get("RNN_DECAY_RATE"))
        self.RNN_DROPOUT = float(os.environ.get("RNN_DROPOUT"))
        self.RNN_SEQ_LENGTH = int(os.environ.get("RNN_SEQ_LENGTH"))
        self.RNN_BATCHSIZE = int(os.environ.get("RNN_BATCHSIZE"))
        self.RNN_EPOCHS = int(os.environ.get("RNN_EPOCHS"))
        self.RNN_GRAD_CLIP = int(os.environ.get("RNN_GRAD_CLIP"))
        self.RNN_INIT_FROM = str(os.environ.get("RNN_INIT_FROM"))

        # Determine paths
        self.CURRENT_PATH = os.path.dirname(os.path.abspath(__file__)) # Chainer path
        self.MODEL_PATH = self.CURRENT_PATH + "/Models/" + os.path.splitext(os.path.basename(self.RNN_TRAINING_FILE))[0] + "/" # Model path
        self.CHECKPOINT_PATH = self.MODEL_PATH + "checkpoints/"
        self.VOCAB_PATH = self.MODEL_PATH + "vocabulary.bin"

        # Multiprocessing Shared Memory
        self.c_type_shared_memory = None




        print("--- Path Summary ---")
        print("Chainer Path: " + self.CURRENT_PATH)
        print("Model Path: " + self.MODEL_PATH)
        print("Checkpoint Path: " + self.CHECKPOINT_PATH)
        print("--------------------")


        # Create directories
        if not os.path.exists(self.MODEL_PATH):
            os.mkdir(self.MODEL_PATH)
        if not os.path.exists(self.CHECKPOINT_PATH):
            os.mkdir(self.CHECKPOINT_PATH)


    def do_checkpoint(self, model, epoch):
        model_copy = copy.deepcopy(model).to_cpu()

        pickle.dump(model_copy, open(self.CHECKPOINT_PATH + "CharRNN-Epoch-%s.model" % epoch, 'wb'))
        pickle.dump(model_copy, open(self.MODEL_PATH + "CharRNN-Latest.model", 'wb'))


    def worker(self, i):
        shared_train_data = np.frombuffer(self.c_type_shared_memory.get_obj())



    def load_data(self):
        vocab = {}
        words = codecs.open(self.RNN_TRAINING_FILE, 'rb', 'utf-8').read()
        words = list(words)
        dataset = np.ndarray((len(words),), dtype=np.int32)
        for i, word in enumerate(words):
            if word not in vocab:
                vocab[word] = len(vocab)
            dataset[i] = vocab[word]
        print('corpus length:', len(words))
        print('vocab size:', len(vocab))
        return dataset, words, vocab

    def train(self):

        train_data, words, vocab = self.load_data()




        pickle.dump(vocab, open(self.VOCAB_PATH, 'wb'))

        if len(self.RNN_INIT_FROM) > 0:
            model = pickle.load(open(self.RNN_INIT_FROM, 'rb'))
        else:
            model = CharRNN(len(vocab), self.RNN_RNN_SIZE)

        if self.RNN_GPU >= 0:
            cuda.get_device(self.RNN_GPU).use()
            model.to_gpu()

        optimizer = optimizers.RMSprop(lr=self.RNN_LEARNING_RATE, alpha=self.RNN_DECAY_RATE, eps=1e-8)
        optimizer.setup(model)

        whole_len    = train_data.shape[0]
        jump         = whole_len / self.RNN_BATCHSIZE
        epoch        = 0
        start_at     = time.time()
        cur_at       = start_at
        state        = make_initial_state(self.RNN_RNN_SIZE, batchsize=self.RNN_BATCHSIZE)
        if self.RNN_GPU >= 0:
            accum_loss   = Variable(cuda.zeros(()))
            for key, value in state.items():
                value.data = cuda.to_gpu(value.data)
        else:
            accum_loss   = Variable(np.zeros((), dtype=np.float32))

        print('going to train {} iterations'.format(jump * self.RNN_EPOCHS))

        self.c_type_shared_memory = multiprocess.Array('d', train_data, lock=False)

        #shared_train_data = multiprocess.Array(ctypes.c_int32, train_data, lock=False)
        pool = multiprocess.Pool(processes=4)
        ids = [i for i in xrange(jump * self.RNN_EPOCHS)]
        pool.map(self.worker, ids)










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
