from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout,
    BatchNormalization,
    Input
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop


class Model:

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def create_baseCNNModel(self):
        """Build a CNN model for image classification"""
        model = Sequential()

        # 2D Convolutional layer
        model.add(
            Conv2D(
                128,  # Number of filters
                (3, 3),  # Padding size
                input_shape=(
                    self.height,
                    self.width,
                    3,
                ),  # Shape of the input images
                activation="relu",  # Output function of the neurons
                padding="same",
            )
        )  # Behaviour of the padding region near the borders
        # 2D Pooling layer to reduce image shape
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Transform 2D input shape into 1D shape
        model.add(Flatten())
        # Dense layer of fully connected neurons
        model.add(Dense(128, activation="relu"))
        # Dropout layer to reduce overfitting, the argument is the proportion of random neurons ignored in the training
        model.add(Dropout(0.2))
        # Output layer
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            loss="binary_crossentropy",  # Loss function for binary classification
            optimizer=RMSprop(
                learning_rate=1e-3
            ),  # Optimizer function to update weights during the training
            metrics=["accuracy", "AUC"],
        )  # Metrics to monitor during training and testing

        # Print model summary
        model.summary()

        return model

    def create_enhancedCNNModel(self):
        model = Sequential()
        model.add(Input(shape=(self.height, self.width, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # Print model summary
        model.summary()

        return model