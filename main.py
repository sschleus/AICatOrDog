import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
from PIL import Image
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QPushButton,
    QMessageBox,
    QVBoxLayout,
    QApplication,
    QLabel,
    QMessageBox,
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QDialog,
)
from PyQt5.QtChart import QChart, QChartView, QPieSeries, QPieSlice
from PyQt5.QtGui import QPainter

from PyQt5 import QtCore, QtGui
from skimage import io, transform
from Conv_operation import Transformations
import pandas as pd
from tensorflow import keras
import shutil
import matplotlib.pyplot as plt



class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setWindowTitle("dogs & cats classifier")
        self.setFixedSize(600, 500)
        centralwidget = QWidget(self)
        PredictTab(centralwidget)
        self.setCentralWidget(centralwidget)


class ResultsWindow(QDialog):
    def __init__(self, cat_percentage, dog_percentage, parent=None):
        super(ResultsWindow, self).__init__(parent)
        self.setWindowTitle("Prediction Results")
        self.setFixedSize(400, 400)
        layout = QVBoxLayout()

        series = QPieSeries()
        series.append(f"Cats", cat_percentage)
        series.append(f"Dogs", dog_percentage)

        # Set labels to display percentages
        for slice in series.slices():
            label = f"{slice.label()}: {slice.percentage() * 100:.1f}%"
            slice.setLabel(label)

        chart = QChart()
        chart.addSeries(series)
        chart.setTitle("Prediction Results")

        chartView = QChartView(chart)
        chartView.setRenderHint(QPainter.Antialiasing)

        layout.addWidget(chartView)
        self.setLayout(layout)



class PredictTab(QWidget):
    def __init__(self, parent):
        super(PredictTab, self).__init__(parent)
        self.setFixedSize(600, 500)
        self.imgPath = []
        self.imgIndex = 0
        self.predictions = []
        self.cnn = None
        self.kernel_name = None
        self.n_layers = 1
        mainLayout = QVBoxLayout(self)

        self.imgLabel = QLabel()
        self.imgLabel.setStyleSheet("background-color: lightgrey; border: 1px solid gray;")
        self.imgLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.prevButton = QPushButton("<")
        self.prevButton.setMaximumWidth(40)
        self.prevButton.setEnabled(False)
        self.nextButton = QPushButton(">")
        self.nextButton.setMaximumWidth(40)
        self.nextButton.setEnabled(False)
        self.predLabel = QLabel("None")
        self.predLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.predLabel.setFixedWidth(300)
        self.predLabel.setFixedHeight(20)

        hWidget1 = QWidget(self)
        hWidget1.setFixedHeight(20)
        hLayout1 = QHBoxLayout(hWidget1)
        hLayout1.setContentsMargins(0, 0, 0, 0)
        hWidget2 = QWidget(self)
        hWidget2.setFixedHeight(25)
        hLayout2 = QHBoxLayout(hWidget2)
        hLayout2.setContentsMargins(0, 0, 0, 0)
        hWidget3 = QWidget(self)
        hWidget3.setFixedHeight(25)
        hLayout3 = QHBoxLayout(hWidget3)
        hLayout3.setContentsMargins(0, 0, 0, 0)
        hWidget4 = QWidget(self)
        hWidget4.setFixedHeight(25)
        hLayout4 = QHBoxLayout(hWidget4)
        hLayout4.setContentsMargins(0, 0, 0, 0)
        hWidget5 = QWidget(self)
        hWidget5.setFixedHeight(25)
        hLayout5 = QHBoxLayout(hWidget5)
        hLayout5.setContentsMargins(0, 0, 0, 0)
        # hWidget.setStyleSheet("border: 1px solid red; padding: 0 0 0 0; margin: 0px;")

        loadButton = QPushButton("Select picture(s)")
        modelButton = QPushButton("Select model (none)")
        predButton = QPushButton("Predict")
        exportButton = QPushButton("Export")
        convolveButton = QPushButton("Apply test convolution")
        kernelButton = QPushButton("Choose test kernel")
        feedbackButton = QPushButton("Upload to database")
        batchPredictButton = QPushButton("Start Testset")

        loadButton.clicked.connect(self.loadImg)
        self.prevButton.clicked.connect(self.prevImg)
        self.nextButton.clicked.connect(self.nextImg)
        modelButton.clicked.connect(lambda: self.selectedModel(modelButton))
        predButton.clicked.connect(self.predict)
        exportButton.clicked.connect(self.export)
        kernelButton.clicked.connect(lambda: self.choose_kernel(kernelButton))
        convolveButton.clicked.connect(self.convolve)
        feedbackButton.clicked.connect(lambda: self.label_feedback(feedbackButton))
        batchPredictButton.clicked.connect(self.batchPredict)

        mainLayout.addWidget(self.imgLabel)
        hLayout1.addWidget(self.prevButton)
        hLayout1.addWidget(self.predLabel)
        hLayout1.addWidget(self.nextButton)
        hLayout2.addWidget(loadButton)
        hLayout2.addWidget(modelButton)
        hLayout3.addWidget(predButton)
        hLayout3.addWidget(exportButton)
        hLayout4.addWidget(convolveButton)
        hLayout4.addWidget(kernelButton)
        hLayout5.addWidget(feedbackButton)
        mainLayout.addWidget(hWidget1)
        mainLayout.addWidget(hWidget2)
        mainLayout.addWidget(hWidget3)
        mainLayout.addWidget(hWidget4)
        mainLayout.addWidget(hWidget5)
        mainLayout.addWidget(batchPredictButton)

    def loadImg(self):
        dialog = QFileDialog()
        dialog.setWindowTitle("Select an image")
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setViewMode(QFileDialog.Detail)

        if dialog.exec_():
            self.imgPath = [str(i) for i in dialog.selectedFiles()]
            self.predictions = [None for i in range(len(self.imgPath))]
            self.imgIndex = 0
            print("Selection:")
            for i in self.imgPath:
                print(i)
            self.updatePixmap(self.imgPath[self.imgIndex])
            self.prevButton.setEnabled(False)
            if len(self.imgPath) > 1:
                self.nextButton.setEnabled(True)
            elif len(self.imgPath) == 1:
                self.nextButton.setEnabled(False)
            self.updatePixmap(self.imgPath[self.imgIndex])
            # if self.cnn is not None:
            # self.predict()

    def batchPredict(self):
        if self.cnn is None:
            QMessageBox(
                QMessageBox.Warning,
                "Error",
                "Please select a neural network model before making prediction",
            ).exec_()
            return

        cat_dir = 'testset/cats/'
        dog_dir = 'testset/dogs/'

        cat_images = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir)[:100]]
        dog_images = [os.path.join(dog_dir, f) for f in os.listdir(dog_dir)[:100]]

        cat_predictions = []
        dog_predictions = []

        for img_path in cat_images:
            img = transform.resize(io.imread(img_path), (256, 256), anti_aliasing=True)
            img = img.reshape(1, 256, 256, 3)
            pred = self.cnn.predict(img, 1)[0][0]
            cat_predictions.append(pred)

        for img_path in dog_images:
            img = transform.resize(io.imread(img_path), (256, 256), anti_aliasing=True)
            img = img.reshape(1, 256, 256, 3)
            pred = self.cnn.predict(img, 1)[0][0]
            dog_predictions.append(pred)

        cat_detected = sum(1 for p in cat_predictions if p < 0.5)
        dog_detected = sum(1 for p in dog_predictions if p >= 0.5)

        total_cats = len(cat_predictions)
        total_dogs = len(dog_predictions)

        cat_percentage = (cat_detected / total_cats) * 100
        dog_percentage = (dog_detected / total_dogs) * 100

        self.showResults(cat_percentage, dog_percentage)

    def showResults(self, cat_percentage, dog_percentage):
        self.resultsWindow = ResultsWindow(cat_percentage, dog_percentage)
        self.resultsWindow.exec_()

    def updatePixmap(self, path, pred=1000):
        self.imgLabel.setPixmap(QtGui.QPixmap(path).scaled(500, 500))
        # self.imgLabel.setScaledContents(True)
        self.predLabel.setText(str(self.predictions[self.imgIndex]))

        print("Prediction:", pred)
        if pred < 0.5:
            self.predLabel.setText(
                f"I think it's a cat! Confidence: {(1.0 - pred) * 100:.0f}%"
            )
        elif pred > 0.5 and pred != 1000:
            self.predLabel.setText(
                f"I think it's a dog! Confidence: {pred * 100:.0f}%"
            )
        else:
            self.predLabel.setText("I don't know yet ")

    def predict(self):
        if len(self.imgPath) > 0 and self.cnn is not None:
            img = transform.resize(
                io.imread(self.imgPath[self.imgIndex]),
                (256, 256),
                anti_aliasing=True,
            )
            try:
                img = img.reshape(1, 256, 256, 3)

                self.predictions[self.imgIndex] = self.cnn.predict(img, 1)[0][0]
                self.updatePixmap(
                    self.imgPath[self.imgIndex], self.predictions[self.imgIndex]
                )
            except:
                QMessageBox(
                    QMessageBox.Warning,
                    "Error",
                    "Cannot convert image, please select a valid image",
                ).exec_()
        else:
            QMessageBox(
                QMessageBox.Warning,
                "Error",
                "Please select an image and a neural network model before making prediction",
            ).exec_()

    def nextImg(self):
        self.imgIndex += 1
        self.updatePixmap(self.imgPath[self.imgIndex])
        if self.imgIndex == len(self.imgPath) - 1:
            self.nextButton.setEnabled(False)
        else:
            self.nextButton.setEnabled(True)
        self.prevButton.setEnabled(True)

    def prevImg(self):
        self.imgIndex -= 1
        self.updatePixmap(self.imgPath[self.imgIndex])
        if self.imgIndex == 0:
            self.prevButton.setEnabled(False)
        else:
            self.prevButton.setEnabled(True)
        self.nextButton.setEnabled(True)

    def selectedModel(self, btn):
        win = ModelWindow()
        if win.exec_():
            name, self.cnn = win.getModel()
            btn.setText(f"Select model ({name})")

    def export(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fname, _ = QFileDialog.getSaveFileName(
            self, "Save as", "Export.csv", "All (.csv)", options=options
        )

        if fname:
            data_tuples = list(zip(self.imgPath, self.predictions))
            df = pd.DataFrame(data_tuples, columns=["Images", "Predictions"])
            df.to_csv(fname)
            print(fname, "saved")

    def choose_kernel(self, btn):
        win = KernelWindow()
        if win.exec_():
            self.kernel_name = win.getKernel()
            self.n_layers = int(win.getNlayers())
            btn.setText(f"Select kernel ({self.kernel_name})")

    def label_feedback(self, btn):
        win = FeedBackWindow()
        if win.exec_():
            txt = self.upload_database(win.get_selected_label())
            btn.setText(f"{txt}")

    def upload_database(self, label):
        if len(self.imgPath) > 0:
            img = io.imread(self.imgPath[self.imgIndex])
            directory = "data/" + label + "s"
            name = self.get_name_picture(label, directory)
            save_path = os.path.join(directory, name)

            img_pil = Image.fromarray(img)
            img_pil.save(save_path)

            return "Image uploaded"
        return "No image found"

    def get_name_picture(self, label, directory):
        number = 1
        name = label + "." + str(number) + ".jpg"
        file_path = os.path.join(directory, name)
        while (os.path.isfile(file_path)):
            number += 1
            name = label + "." + str(number) + ".jpg"
            file_path = os.path.join(directory, name)
        return name

    def convolve(self):
        if self.kernel_name == None:
            QMessageBox(
                QMessageBox.Warning,
                "Error",
                "Please select a kernel before convoluting",
            ).exec_()
        else:
            for img in self.imgPath:
                # Apply one convolution + pooling operation
                t = Transformations(img)
                t.choose_kernel(self.kernel_name)
                conv_name = t.multilayer(self.n_layers)
                self.imgLabel.setPixmap(
                    QtGui.QPixmap(conv_name).scaled(500, 500)
                )
                self.predLabel.setText(
                    f"{self.kernel_name} convolution with {self.n_layers} layers and maxpooling"
                )


class FeedBackWindow(QDialog):
    def __init__(self):
        super(FeedBackWindow, self).__init__()
        self.setWindowTitle("Upload feedback")
        self.label = None
        mainLayout = QVBoxLayout(self)
        hWidget = QWidget()
        hLayout = QHBoxLayout(hWidget)
        text = QLabel("Label the picture")
        list = QListWidget()

        dir = ["cat", "dog"]
        if len(dir) > 0:
            list.addItems([name for name in dir])
        else:
            self.checkCount(list)

        mainLayout.addWidget(text)
        mainLayout.addWidget(list)

        self.select = QPushButton("Select")
        self.select.clicked.connect(lambda: self.ok_pressed(list.currentItem().text()))
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.cancel_pressed)

        hLayout.addWidget(self.select)
        hLayout.addWidget(cancel)
        hLayout.setContentsMargins(0, 0, 0, 0)
        mainLayout.addWidget(hWidget)

        self.setLayout(mainLayout)

    def checkCount(self, list):
        if list.count() == 0:
            list.addItems(["No label selected"])
            list.setEnabled(False)
            self.select.setEnabled(False)
            self.delete.setEnabled(False)

    def cancel_pressed(self):
        self.reject()

    def ok_pressed(self, selectedLabel):
        self.label = selectedLabel
        self.accept()

    def get_selected_label(self):
        return self.label


class ModelWindow(QDialog):
    def __init__(self):
        super(ModelWindow, self).__init__()
        self.setWindowTitle("Model selection")
        self.model = None
        self.name = None
        mainLayout = QVBoxLayout(self)
        hWidget = QWidget()
        hLayout = QHBoxLayout(hWidget)
        text = QLabel("Select a neural network model: ")
        list = QListWidget()
        self.select = QPushButton("Select")
        self.select.clicked.connect(
            lambda: self.ok_pressed(list.currentItem().text())
        )
        self.delete = QPushButton("Delete")
        self.delete.clicked.connect(lambda: self.delete_pressed(list))
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.cancel_pressed)

        dir = [
            name
            for name in os.listdir(".")
            if os.path.isdir(name) and name.startswith("model_")
        ]

        if len(dir) > 0:
            list.addItems(["_".join(name.split("_")[1:]) for name in dir])
        else:
            self.checkCount(list)

        mainLayout.addWidget(text)
        mainLayout.addWidget(list)
        hLayout.addWidget(self.select)
        hLayout.addWidget(self.delete)
        hLayout.addWidget(cancel)
        hLayout.setContentsMargins(0, 0, 0, 0)
        mainLayout.addWidget(hWidget)
        self.setLayout(mainLayout)

    def checkCount(self, list):
        if list.count() == 0:
            list.addItems(["No models found"])
            list.setEnabled(False)
            self.select.setEnabled(False)
            self.delete.setEnabled(False)

    def getModel(self):
        return (self.name, self.model)

    def ok_pressed(self, selected):
        print(selected, "selected")
        try:
            self.model = keras.models.load_model("model_" + selected + "/trained_model.keras")
            self.name = selected
        except:
            print("Cannot load model")
        self.accept()

    def delete_pressed(self, list):
        shutil.rmtree("model_" + list.currentItem().text())
        list.takeItem(list.currentRow())
        self.checkCount(list)

    def cancel_pressed(self):
        self.reject()


class KernelWindow(QDialog):
    def __init__(self):
        super(KernelWindow, self).__init__()
        self.setWindowTitle("Kernel selection")
        self.model = None
        self.name = None
        mainLayout = QVBoxLayout(self)
        hWidget = QWidget()
        hLayout = QHBoxLayout(hWidget)
        text = QLabel("Select a kernel for the convolution : ")
        list = QListWidget()

        dir = [
            "identity",
            "sharpen",
            "blur",
            "bottom sobel",
            "emboss kernel",
            "left sobel",
            "outline",
            "right sobel",
            "top sobel",
        ]
        if len(dir) > 0:
            list.addItems([name for name in dir])
        else:
            self.checkCount(list)

        mainLayout.addWidget(text)
        mainLayout.addWidget(list)

        hWidgetConv = QWidget()
        hLayoutConv = QHBoxLayout(hWidgetConv)
        textConv = QLabel("Select a number of convolutional layers: ")
        listConv = QListWidget()
        dirConv = [str(i) for i in range(1, 6)]
        if len(dirConv) > 0:
            listConv.addItems([name for name in dirConv])
        else:
            self.checkCount(listConv)

        self.select = QPushButton("Select")
        self.select.clicked.connect(
            lambda: self.ok_pressed(
                list.currentItem().text(), listConv.currentItem().text()
            )
        )
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.cancel_pressed)

        mainLayout.addWidget(textConv)
        mainLayout.addWidget(listConv)
        hLayoutConv.setContentsMargins(0, 0, 0, 0)
        mainLayout.addWidget(hWidgetConv)

        hLayout.addWidget(self.select)
        hLayout.addWidget(cancel)
        hLayout.setContentsMargins(0, 0, 0, 0)
        mainLayout.addWidget(hWidget)

        self.setLayout(mainLayout)

    def checkCount(self, list):
        if list.count() == 0:
            list.addItems(["No kernels found"])
            list.setEnabled(False)
            self.select.setEnabled(False)
            self.delete.setEnabled(False)

    def getKernel(self):
        return self.kernel_name

    def getNlayers(self):
        return self.n_layers

    def ok_pressed(self, selectedKernel, selectedN):
        print(selectedKernel, "kernel selected")
        print(selectedN, "layers selected")
        try:
            self.kernel_name = selectedKernel
            self.n_layers = selectedN
        except:
            print("Cannot choose this kernel")
        self.accept()

    def cancel_pressed(self):
        self.reject()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()