import os
import time
from PIL import Image, UnidentifiedImageError

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger

SAVE_DIR = "backup"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


class DogCatClassifier:
    BATCH_SIZE = 128  # We increase the size of batches to improve training speed
    IMG_HEIGHT = 256
    IMG_WIDTH = 256

    def __init__(self, data_dir="data", epochs=1):
        self.epochs = epochs
        self.data_dir = data_dir
        self.X, self.y = self._load_data()

    def _load_data(self):
        """
        Charge les données à partir des sous-dossiers 'cats' et 'dogs'.
        """
        cat_files = [os.path.join("cats", file) for file in os.listdir(os.path.join(self.data_dir, "cats")) if
                     self._is_image_file(os.path.join(self.data_dir, "cats", file))]
        dog_files = [os.path.join("dogs", file) for file in os.listdir(os.path.join(self.data_dir, "dogs")) if
                     self._is_image_file(os.path.join(self.data_dir, "dogs", file))]

        X = cat_files + dog_files
        y = ["c"] * len(cat_files) + ["d"] * len(dog_files)

        return np.array(X), np.array(y)

    def _is_image_file(self, filepath):
        try:
            Image.open(filepath)
            return True
        except (IOError, UnidentifiedImageError):
            return False

    def _get_label(self, filename):
        if filename.startswith("cats"):
            return "c"
        elif filename.startswith("dogs"):
            return "d"
        return "unknown"

    def fit(self, folder, model):
        train_set, val_set, test_set = self._gen_data()

        # Reduce learning rate when a metric has stopped improving.
        # Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates.
        learning_rate_reduction = ReduceLROnPlateau(
            monitor='val_accuracy',
            patience=4,
            verbose=1,
            factor=0.75,
            min_lr=0.00001
        )

        early_stopping = EarlyStopping(patience=10)

        os.makedirs(folder, exist_ok=True)
        checkpoint_acc = ModelCheckpoint(
            filepath=os.path.join(folder, 'trained_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            save_freq='epoch'
        )

        # We create logs who contains the [epoch,accuracy,learning_rate,loss,val_accuracy,val_loss]
        logs_path = os.path.join(folder, 'logs')
        os.makedirs(logs_path, exist_ok=True)
        log_filename = os.path.join(logs_path, 'log.csv')
        history_logger = CSVLogger(log_filename, separator=",", append=True)

        # Liste des callbacks
        callbacks = [learning_rate_reduction, early_stopping, checkpoint_acc, history_logger]

        beg = int(time.time())

        history = model.fit(
            train_set,
            steps_per_epoch=train_set.samples // self.BATCH_SIZE,
            epochs=self.epochs,
            validation_data=val_set,
            validation_steps=val_set.samples // self.BATCH_SIZE,
            callbacks=callbacks
        )

        end = int(time.time())
        t = end - beg
        hrs = t // 3600
        mins = (t - 3600 * hrs) // 60
        secs = t % 60
        print("training took {} hrs -- {} mins -- {} secs".format(hrs, mins, secs))

        result = model.evaluate(test_set, batch_size=self.BATCH_SIZE)
        print("Testing set evaluation:", dict(zip(model.metrics_names, result)))

        model.save_weights(os.path.join(SAVE_DIR, 'model.weights.h5'))
        model.save(os.path.join(SAVE_DIR, 'latest_model.keras'))

        self._plot(history)

    def _gen_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y)
        df_train = pd.DataFrame({"filename": X_train, "class": y_train})
        df_test = pd.DataFrame({"filename": X_test, "class": y_test})

        train_datagen = ImageDataGenerator(
            rescale=1 / 255,
            preprocessing_function=preprocess_input,
            validation_split=0.2,
            horizontal_flip=True,
            shear_range=0.2,
            height_shift_range=0.2,
            width_shift_range=0.2,
            zoom_range=0.2,
            rotation_range=30,
            fill_mode="nearest",
        )
        test_datagen = ImageDataGenerator(
            rescale=1 / 255, preprocessing_function=preprocess_input
        )

        train_data_generator = train_datagen.flow_from_dataframe(
            df_train,
            directory=self.data_dir,
            x_col="filename",
            y_col="class",
            subset="training",
            shuffle=True,
            batch_size=self.BATCH_SIZE,
            class_mode="binary",
            target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
        )
        valid_data_generator = train_datagen.flow_from_dataframe(
            df_train,
            directory=self.data_dir,
            x_col="filename",
            y_col="class",
            subset="validation",
            shuffle=True,
            batch_size=self.BATCH_SIZE,
            class_mode="binary",
            target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
        )
        test_data_generator = test_datagen.flow_from_dataframe(
            df_test,
            directory=self.data_dir,
            x_col="filename",
            y_col="class",
            shuffle=False,
            batch_size=self.BATCH_SIZE,
            class_mode="binary",
            target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
        )

        return train_data_generator, valid_data_generator, test_data_generator

    def _plot(self, history):
        epochs_range = range(len(history.history['accuracy']))

        acc = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label="Training Accuracy")
        plt.plot(epochs_range, val_acc, label="Validation Accuracy")
        plt.legend(loc="lower right")
        plt.title("Training and Validation Accuracy")

        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label="Training Loss")
        plt.plot(epochs_range, val_loss, label="Validation Loss")
        plt.legend(loc="upper right")
        plt.title("Training and Validation Loss")

        plt.savefig(os.path.join(SAVE_DIR, "results.png"))
