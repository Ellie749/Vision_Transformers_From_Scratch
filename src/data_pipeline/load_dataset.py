from keras.datasets import cifar100
from keras.layers import Layer
import tensorflow as tf

class Patches(Layer):
    def __init__(self, patch_size: int) -> None:
        super().__init__()
        print('[INFO] Creating patches...')
        self.patch_size = patch_size

    def call(self, images: tf.Tensor) -> tf.Tensor:
        print('[INFO] Extracting patches...')
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        out =  tf.reshape(patches, [tf.shape(images)[0], -1, patches.shape[-1]])
        print(f'[INFO] Patches shape: {out.shape}')
        return out

def preprocess(patch_size):
    """
    Load and Create Patches from dataset.
    """
    print('[INFO] Loading data...')
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    print(f'X_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'y_test shape: {y_test.shape}')

    patches = Patches(patch_size)
    X_train_patches = patches(X_train)
    X_test_patches = patches(X_test)

    return X_train_patches, X_test_patches, y_train, y_test


if __name__ == "__main__":
    preprocess(4)



