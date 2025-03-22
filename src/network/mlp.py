from keras.layers import Layer, Dense, Dropout
from keras.activations import gelu

class MLP(Layer):
    def __init__(self, units, dropout_rate):
        super(MLP, self).__init__()
        self.mlp1 = Dense(units*2, activation=gelu)
        self.dropout1 = Dropout(rate=dropout_rate)
        self.mlp2 = Dense(units, activation=gelu)
        self.dropout2 = Dropout(rate=dropout_rate)

    def call(self, embeddings):
        out = self.mlp1(embeddings)
        out = self.dropout1(out)
        out = self.mlp2(out)
        out = self.dropout2(out)

        return out
    