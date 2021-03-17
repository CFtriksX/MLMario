import tensorflow as tf

imgRows , imgCols = 84, 84
numberOfActions = 12

def createModel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Convolution2D(32, 8, 4, input_shape=(imgRows, imgCols, 4)),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Convolution2D(64, 4, 2),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Convolution2D(64, 3, 1),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dense(numberOfActions, activation=tf.nn.softmax),
        #tf.keras.layers.Dense(1, activation="linear")
    ])

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])

    return model