from keras import Model
from keras.layers import Input, LayerNormalization, MultiHeadAttention, Flatten
from data_pipeline.load_dataset import Patches
from network.positional_encoding import PositionalEncoding
from network.mlp import MLP


def create_model(input_shape, patch_size, n_patches, units, n_layers, model_dim, n_heads, n_classes, dropout_rate):
    mlp_encoder = MLP(units, dropout_rate)
    mlp_classification = MLP(n_classes, dropout_rate)
    patches = Patches(patch_size)
    pe = PositionalEncoding(n_patches, model_dim)

    input_layer = Input(shape=(input_shape))
    x = patches(input_layer)
    x = pe(x)

    for _ in range(n_layers):
        l1 = LayerNormalization()(x)
        mha = MultiHeadAttention(num_heads=n_heads, key_dim=model_dim, value_dim=model_dim)(l1, l1)
        x2 = l1 + mha
        l2 = LayerNormalization()(x2)
        x = x2 + mlp_encoder(l2)

    flattened_embeddings = Flatten()(x)
    logits = mlp_classification(flattened_embeddings)

    model = Model(inputs=input_layer, outputs=logits)

    return model


"""
if __name__ == '__main__':
    model = create_model(input_shape=(32,32), 
                 units=64, 
                 n_layers=2,
                 n_patch=64,
                 model_dim=32,
                 n_heads=4,
                 n_classes=10
                 )
    print(model.summary())
"""
    


        



    


