import tensorflow as tf
from keras.layers import Layer, Dense, Embedding


class PositionalEncoding(Layer):
    def __init__(self, n_patch: int, model_dim: int):
        super(PositionalEncoding, self).__init__()
        print('[INFO] Creating Positional Encoding...')
        self.n_patches = n_patch
        self.linear_projection = Dense(model_dim)
        self.positional_encoding = Embedding(n_patch, model_dim)


    def call(self, patches: tf.Tensor):
        print('[INFO] Linear projection on flattened patches...')
        flattened_patches = self.linear_projection(patches)
        out = flattened_patches + self.positional_encoding(tf.range(0, self.n_patches, delta=1))

        print(f'[INFO] Dimension of data: {out.shape}')
        return out
    

if __name__ == '__main__':
    N_PATCHES = 64
    MODEL_DIM = 32
    # PositionalEncoding(N_PATCHES, MODEL_DIM)(X_train)