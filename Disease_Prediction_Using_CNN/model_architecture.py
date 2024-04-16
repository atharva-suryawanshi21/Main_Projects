import tensorflow as tf
from keras import layers


image_size = 256
batch_size = 32


# resize_rescale_layer = tf.keras.Sequential([
#     layers.experimental.preprocessing.Resizing(image_size, image_size),
#     layers.experimental.preprocessing.Rescaling(1.0/255)
# ])
resize_rescale_layer = tf.keras.Sequential([
    tf.keras.layers.Resizing(image_size, image_size),
    tf.keras.layers.Rescaling(1.0/255)
])

data_augmentation_layer = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2)
])


def get_model():
    input_shape = (batch_size, image_size, image_size, 3)

    def conv2D_layer(filter):
        return tf.keras.layers.Conv2D(
            filters=filter,
            kernel_size=(3, 3),
            padding="valid",  # no padding
            activation="relu",
            input_shape=input_shape
        )

    model = tf.keras.Sequential([
        # preprocessing layers
        resize_rescale_layer,
        data_augmentation_layer,

        # Convolutional layer
        conv2D_layer(128),
        tf.keras.layers.MaxPool2D((2, 2)),
        conv2D_layer(64),
        tf.keras.layers.MaxPool2D((2, 2)),
        conv2D_layer(64),
        tf.keras.layers.MaxPool2D((2, 2)),
        conv2D_layer(64),
        tf.keras.layers.MaxPool2D((2, 2)),
        conv2D_layer(32),
        tf.keras.layers.MaxPool2D((2, 2)),

        tf.keras.layers.Flatten(),

        # Dense layer
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax")
    ])

    model.build(input_shape=input_shape)

    return model
