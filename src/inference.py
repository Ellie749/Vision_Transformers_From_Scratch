import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import cv2
import numpy as np
import tensorflow as tf
from network.architecture import create_model
from data_pipeline.load_dataset import Patches

PATCH_SIZE = 4
N_PATCHES = 64
MODEL_DIM = 32
IMAGE_SIZE = (32, 32, 3)
N_LAYERS = 2
N_HEADS = 4
N_CLASSES = 100
DROPOUT_RATE = 0.001
IMG_PATH = 'src/test_data_01.jpg'
LABEL_PATH = 'src/labels.txt'


def main():
    with open(LABEL_PATH) as f:
        content = f.read()
    labels = [label.strip().strip('"') for label in content.split(',') if label.strip()]

    model = create_model(
            input_shape= IMAGE_SIZE,
            patch_size= PATCH_SIZE,
            n_patches= N_PATCHES,
            units= MODEL_DIM,
            n_layers= N_LAYERS,
            n_heads= N_HEADS,
            n_classes= N_CLASSES,
            model_dim= MODEL_DIM,
            dropout_rate= DROPOUT_RATE
        )
    print(model.summary())
    # print(model(tf.zeros((1, *IMAGE_SIZE))))
    img = cv2.imread(IMG_PATH)
    test_img = cv2.resize(img, (32, 32))
    test_img = tf.convert_to_tensor(test_img)

    model.load_weights('src/weights/checkpoint_experiment1')

    logits = model.predict(tf.expand_dims(test_img, axis=0))
    probs = tf.nn.softmax(logits).numpy()
    pred = labels[np.argmax(probs, axis=1)[0]]
    cv2.putText(img, pred, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('test_image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()