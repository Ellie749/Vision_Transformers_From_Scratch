import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import cifar100
from keras.layers import Layer


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
            padding='VALID')

        out = tf.reshape(patches, [tf.shape(images)[0], -1, patches.shape[-1]])
        print(f'[INFO] Patches shape: {out.shape}')
        return out


def load_data():
    print('[INFO] Loading data...')
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()
    print(f'X_train shape: {X_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'y_test shape: {y_test.shape}')

    return X_train, X_test, y_train, y_test


def preprocess(patch_size):
    """
    Load and Create Patches from dataset.

    Args:
        patch_size: size of each patch (numebr of patches = (img_size/patch_size)^2)

    Returns:
        X_train_patches, X_test_patches, y_train, y_test: patches and labels
    """
    X_train, X_test, y_train, y_test = load_data()

    # patches = Patches(patch_size)
    # X_train_patches = patches(X_train)
    # X_test_patches = patches(X_test)

    # viz_patches(X_train[0], X_train_patches[0])

    return X_train, X_test, y_train, y_test


def viz_patches(img, patches):
    plt.imshow(img)
    plt.show()

    n = int(np.sqrt(64))
    plt.figure()
    plt.title('patches')
    for i, patch in enumerate(patches):
        ax = plt.subplot(n, n, i+1)
        img = np.array(tf.reshape(patch, [4,4,3])).astype('uint8')
        plt.imshow(img)
        plt.axis('off')
    plt.show()


'''
if __name__ == "__main__":
    PATCH_SIZE = 4
    X_train, X_test, y_train, y_test = preprocess(PATCH_SIZE)
''' 
    
   

