import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtCore import Qt

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


class ImageProcessingApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # Création des widgets
        self.image_label_left = QLabel(self)
        self.image_label_right = QLabel(self)
        self.load_button = QPushButton('Interpolate', self)
        self.load_button_load = QPushButton('Data path', self)
        self.attribute_label = QLabel('Attribute: Male', self)  # Added label for attribute

        # Définir une taille fixe pour le bouton
        self.load_button.setFixedSize(150, 40)
        self.load_button_load.setFixedSize(150, 40)

        # Mettre le titre en gras
        self.attribute_label.setFont(QFont('Arial', 12, QFont.Bold))

        # Mise en place du layout
        layout = QVBoxLayout(self)
        image_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        image_layout.addWidget(self.image_label_left)
        image_layout.addWidget(self.image_label_right)

        button_layout.addWidget(self.load_button_load)
        button_layout.addWidget(self.load_button)

        # Aligner le titre au centre en haut
        self.attribute_label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)

        layout.addWidget(self.attribute_label)  # Added attribute label
        layout.addLayout(image_layout)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Connecter le bouton à la fonction de traitement d'image
        self.load_button.clicked.connect(self.process_image)
        self.load_button_load.clicked.connect(self.set_data_path)

        self.data_path = None

        self.setGeometry(100, 100, 500, 400)
        self.setWindowTitle('Fader Networks App')
        self.show()

    def set_data_path(self):
        # Fonction pour demander le chemin d'un fichier
        options = QFileDialog.Options()
        # options = QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Sélectionner un fichier", "", "Tous les fichiers (*);;Images Files (*.png;*.jpg;*.jpeg)", options=options)

        if file_name:
            print("Fichier sélectionné :", file_name)
            self.data_path = file_name

    def process_image(self):

        if self.data_path is not None:

            input_image, output_image = predict(self.data_path)

            pixmap1 = self.tensor_to_pixmap(input_image[0], True)
            pixmap2 = self.tensor_to_pixmap(output_image[0], False)
            input_pixmap = QPixmap(pixmap1)
            output_pixmap = QPixmap(pixmap2)

            # Afficher les images dans les labels correspondants
            self.image_label_left.setPixmap(input_pixmap.scaledToWidth(200, Qt.SmoothTransformation))
            self.image_label_right.setPixmap(output_pixmap.scaledToWidth(200, Qt.SmoothTransformation))

    def tensor_to_pixmap(self, tensor_image, np):
        # Convert the tensor image to a NumPy array
        if not np:
            numpy_image = tensor_image.numpy()

        # Convert the NumPy array to a QImage
        height, width, channel = numpy_image.shape
        bytes_per_line = 3 * width
        qimage = QImage(numpy_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Convert the QImage to a QPixmap
        pixmap = QPixmap.fromImage(qimage)

        return pixmap


def predict(img_path='data/img/1.jpeg', ae_path='models/_ae.pt', n_interpolations=10, max_attr=2.0, min_attr=2.0, save_fig_path='fig/fig.jpg', use_gpu=False, IMAGE_SIZE=256):

    # Load Model
    ae = torch.load(ae_path, map_location='cuda' if use_gpu else 'cpu')

    # Interpolate Attributes
    alphas = np.linspace(1 - min_attr, max_attr, n_interpolations)
    alphas = [[1 - alpha, alpha] for alpha in alphas]

    with Image.open(img_path) as img:
        img = img.crop((0, 0, 178, 178))
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img = np.array(img, dtype=np.float32) / 127.5-1.0
        img = transforms.ToTensor()(img).unsqueeze(0)

    # Run test
    test_img = img.float()
    test_attr = torch.FloatTensor(alphas[1])
    reconstruct_img = ae.decode(ae.encode(test_img), test_attr)
    
    org_image = np.transpose((1+test_img.squeeze(0).cpu().data.numpy())/2, (1, 2, 0))
    rec_image = np.transpose((1+reconstruct_img.squeeze(0).cpu().data.numpy())/2, (1, 2, 0))
    if save_fig_path:
        plt.imsave(save_fig_path, rec_image)
        print('Result saved to', save_fig_path)
        plt.close()

    # Show result
    return org_image, rec_image


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessingApp()
    sys.exit(app.exec_())
