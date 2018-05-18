import sys
import random
from char_lstm import *

if __name__ == "__main__":
    if len(sys.argv) > 1:
        step = int(sys.argv[1])
    else:
        step = sys.maxsize

    input_str = random.choice(chars)

    model.load_params("char_lstm.params", context)

    hidden = model.begin_state(func=mx.nd.zeros, batch_size=1, ctx=context)
    for i in range(step):
        sample = mx.nd.array([char_indices[c] for c in input_str], ctx=context)
        sample = rnn_batch(sample, 1)
        output, hidden = model(sample, hidden)
        probs = mx.nd.softmax(output, axis=1)
        index = mx.nd.random.multinomial(probs)
        gen_char = indices_char[index[-1].asscalar()]
        input_str = input_str[1:] + gen_char
        print(gen_char, end="", flush=True)
