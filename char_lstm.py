import time
import mxnet as mx

context = mx.cpu()
batch_size = 512
sequence_length = 128
num_embed = 30
num_hidden = 512
num_layers = 2
learning_rate = 20
grads_clip = 0.25

with open("caf_code.txt") as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars) + 1
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

class RNNModel(mx.gluon.Block):
    def __init__(self, vocab_size, num_embed, num_hidden, num_layers,
                 dropout=0.5, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self._encoder = mx.gluon.nn.Embedding(
                vocab_size, num_embed,
                weight_initializer=mx.init.Uniform(0.1))
            self._rnn = mx.gluon.rnn.LSTM(num_hidden, num_layers)
            self._dropout = mx.gluon.nn.Dropout(dropout)
            self._decoder = mx.gluon.nn.Dense(vocab_size)
        self._num_hidden = num_hidden

    def forward(self, inputs, hidden):
        embed = self._encoder(inputs)
        output, hidden = self._rnn(embed, hidden)
        output = self._dropout(output)
        decoded = self._decoder(output.reshape((-1, self._num_hidden)))
        return decoded, hidden

    def begin_state(self, *args, **kwargs):
        return self._rnn.begin_state(*args, **kwargs)

model = RNNModel(vocab_size, num_embed, num_hidden, num_layers)
loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()

def rnn_batch(data, batch_size):
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data

def get_batch(source, i, seq):
    seq_len = min(seq, source.shape[0] - i - 1)
    data = source[i: i + seq_len]
    target = source[i + 1: i + seq_len + 1]
    return data, target.reshape((-1,))

def train(dataset):
    global learning_rate
    epoch = 0
    best_L = float("Inf")
    epochs_no_progress = 0
    trainer = mx.gluon.Trainer(model.collect_params(), 'sgd',
                               {'learning_rate': learning_rate,
                                'momentum': 0, 'wd': 0})
    while learning_rate >= 1e-3:
        ts = time.time()
        total_L = 0.0
        hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size,
                                   ctx=context)
        for i in range(0, dataset.shape[0] - 1, sequence_length):
            data, label = get_batch(dataset, i, sequence_length)
            hidden = [i.detach() for i in hidden]
            with mx.autograd.record():
                output, hidden = model(data, hidden)
                L = loss(output, label)
                L = L / (batch_size * sequence_length)
                L.backward()
            grads = [i.grad(context) for i in model.collect_params().values()]
            mx.gluon.utils.clip_global_norm(grads, grads_clip)
            trainer.step(1)
            total_L += mx.nd.sum(L).asscalar()
        epoch += 1

        avg_L = total_L / ((dataset.shape[0] - 1) / sequence_length)
        print("[Epoch %d]  learning_rate %f  loss %f  epochs_no_progress %d  dur %.2fs" %
            (epoch, learning_rate, avg_L, epochs_no_progress, time.time() - ts), flush=True)

        if avg_L < best_L:
            best_L = avg_L
            epochs_no_progress = 0
            model.save_params("char_lstm.params")
        elif epochs_no_progress < 10:
            epochs_no_progress += 1
        else:
            epochs_no_progress = 0
            learning_rate *= 0.25
            trainer.set_learning_rate(learning_rate)

if __name__ == "__main__":
    dataset = mx.nd.array([char_indices[c] for c in text], ctx=context)
    dataset = rnn_batch(dataset, batch_size)
    model.initialize(mx.init.Xavier(), ctx=context)
    train(dataset)
