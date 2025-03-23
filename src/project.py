from data_pipeline.load_dataset import preprocess
from network.architecture import create_model
from model.train import run_experiments
from visualization.utils import plot_metrics

PATCH_SIZE = 4
N_PATCHES = 64
MODEL_DIM = 32
IMAGE_SIZE = (32, 32, 3)
N_LAYERS = 2
N_HEADS = 4
N_CLASSES = 100
BATCH_SIZE = 64
EPOCHS = 50
DROPOUT_RATE = 0.001


def main():
    X_train, X_test, y_train, y_test = preprocess(PATCH_SIZE)
    
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
    H = run_experiments(model, X_train, y_train, X_test, y_test, BATCH_SIZE, EPOCHS)
    plot_metrics(H)


    
if __name__ == '__main__':
    main()