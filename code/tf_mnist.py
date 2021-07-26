import tensorflow as tf

def mnist_tf_objective(config):
    tf.random.set_seed(0)
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(config['dropout']),
        tf.keras.layers.Dense(10, activation=None)
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'], epsilon=config['adam_epsilon'])

    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    res = model.fit(x_train, y_train, epochs=config['epochs'], batch_size=int(config['batch_size']), shuffle=False)
    training_loss_history = res.history['loss']
    val_loss_history = res.history['val_loss']
    val_acc_history = res.history['val_accuracy']
    res_test = model.evaluate(x_test, y_test)
    return res_test[1], model, training_loss_history, val_loss_history, val_acc_history

if __name__ == "__main__":
    test_config = {'batch_size': 50, 'learning_rate': .001, 'epochs': 10, 'dropout': 0.5, 'adam_epsilon': 10**-9}
    res = mnist_tf_objective(test_config)