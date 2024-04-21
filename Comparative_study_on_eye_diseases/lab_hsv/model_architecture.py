from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten

# Load the pre-trained VGG model without the top (classification) layer


def get_model():
    base_model = VGG16(weights='imagenet', include_top=False,
                       input_shape=(224, 224, 3))

    # Freeze the weights of the pre-trained layers except the last second layer
    for layer in base_model.layers:
        layer.trainable = False

    # Extract the last second layer
    x = base_model.output

    # Add your custom dense layer
    x = Flatten()(x)
    # Add your dense layer with the desired number of neurons
    x = Dense(2048, activation='relu')(x)
    # Add more layers if needed

    # Add the final custom classification layer
    # num_classes is the number of classes in your classification task
    output = Dense(4, activation='softmax')(x)

    # Create the new model
    model = Model(inputs=base_model.input, outputs=output)

    return model
