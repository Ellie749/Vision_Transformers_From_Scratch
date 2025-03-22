import tensorflow as tf
from keras.callbacks import ModelCheckpoint

def run_experiments(model, X_train, y_train, X_test, y_test, batch_size, epochs):
    optimizer = tf.optimizers.AdamW()
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = tf.metrics.SparseCategoricalAccuracy(name='accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


    checkpoint_file = 'weights/checkpoint_experiment1'
    checkpoint_callback = ModelCheckpoint(checkpoint_file, monitor='val_accuracy', save_best_only=True, save_weights_only=True)


    H = model.fit(
        X_train, 
        y_train, 
        batch_size=batch_size, 
        epochs=epochs, 
        validation_data=(X_test, y_test), 
        callbacks=[checkpoint_callback]
        )
    
    return H