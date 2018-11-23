import keras
from keras.layers import Input, LSTM, Embedding, Bidirectional
from keras_contrib.layers import CRF
from keras.models import Model


def build_lstm_crf_model(num_cates, seq_len, vocab_size, model_opts=dict()):
    opts = {
        'emb_size': 256,
        'emb_trainable': True,
        'emb_matrix': None,
        'lstm_units': 256,
        'optimizer': keras.optimizers.Adam()
    }
    opts.update(model_opts)

    input_seq = Input(shape=(seq_len,), dtype='int32')
    if opts.get('emb_matrix') is not None:
        embedding = Embedding(vocab_size, opts['emb_size'], 
                              weights=[opts['emb_matrix']],
                              trainable=opts['emb_trainable'])
    else:
        embedding = Embedding(vocab_size, opts['emb_size'])
    x = embedding(input_seq)
    lstm = LSTM(opts['lstm_units'], return_sequences=True)
    x = Bidirectional(lstm)(x)
    crf = CRF(num_cates, sparse_target=True)
    output = crf(x)

    model = Model(input_seq, output)
    model.compile(opts['optimizer'], loss=crf.loss_function, metrics=[crf.accuracy])
    return model
