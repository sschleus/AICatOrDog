from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout,
    BatchNormalization,
    Input
)
import tensorflow as tf
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
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=(self.img_h, self.img_w, 3)))

        self._build_convultional_layers()  # Add Convultional Layer
        self.model.add(tf.keras.layers.Flatten())  # Flatten
        self._fully_connect_network()  # Connect neurons between layers

        self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

        return self.model

    def _fully_connect_network(self):
        self._add_dense_block(512)
        self.model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    def _build_convultional_layers(self):
        filters = 32
        kernel_size = (3, 3)
        for _ in range(4):
            self._add_conv_block(filters, kernel_size)
            filters = 2 * filters

    def _add_conv_block(self, filters, kernel_size):
        self.model.add(
            tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', padding="same"))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    def _add_dense_block(self, units):
        self.model.add(tf.keras.layers.Dense(units=units, activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dropout(rate=0.2))
