import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtCore import Qt
import torch

from dataset.dataloader import get_dataloaders
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid

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
        self.attribute_label = QLabel('Attribute: Gender', self)  # Added label for attribute

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

        self.setGeometry(100, 100, 500, 400)
        self.setWindowTitle('Fader Networks App')
        self.show()

    def set_data_path(self):
        # Add fonction to ask for the path data
        options = QFileDialog.Options()
        # options = QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(self, "Sélectionner le dossier de données", options=options)

        if directory:
            print("Chemin sélectionné:", directory)
            self.data_path = directory

    def process_image(self):
        
        input_image, output_image = UI_interpolation(self.data_path)

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


def UI_interpolation(input_data):
    # create logger / load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = torch.load('models/best_autoencoder.pt').to(device)
    
    # restore main parameters
    debug = True
    batch_size = 32
    v_flip = False
    h_flip = False
    img_sz = 256
    attr = ['Young']
    n_attr = 2
    if not (len(attr) == 1 and n_attr == 2):
        raise Exception("The model must use a single boolean attribute only.")
    
    # load dataset
    
    _, _, test_data = get_dataloaders(
        input_data, name_attr=attr[0], batch_size=batch_size)
    
    
    def get_interpolations(ae, images, attributes):
        """
        Reconstruct images / create interpolations
        """
        assert len(images) == len(attributes)
        enc_outputs = ae.encode(images)
    
        # interpolation values
        alphas = np.linspace(1 - 1, 1, 1)
        alphas = [torch.FloatTensor([1 - alpha, alpha]).to(device) for alpha in alphas]
    
        # original image / reconstructed image / interpolations
        outputs = []
        outputs.append(images)
        outputs.append(ae.decode(enc_outputs, attributes))
        for alpha in alphas:
            alpha = Variable(alpha.unsqueeze(0).expand((len(images), 2)))
            outputs.append(ae.decode(enc_outputs, alpha))
    
        # return stacked images
        return torch.cat([x.unsqueeze(1) for x in outputs], 1).data.cpu()
    
    
    interpolations = []
    
    for k in range(0, 1, 100):
        images, attributes = next(iter(test_data))
        images, attributes = images.to(device), attributes.to(device)
        interpolations.append(get_interpolations(ae, images[:1], attributes[:1]))
    
    interpolations = torch.cat(interpolations, 0)
    assert interpolations.size() == (1, 2 + 1,
                                    3, img_sz, img_sz)
    
    
    def get_grid(images, row_wise, plot_size=5):
        """
        Create a grid with all images.
        """
        n_images, n_columns, img_fm, img_sz, _ = images.size()
        if not row_wise:
            images = images.transpose(0, 1).contiguous()
        images = images.view(n_images * n_columns, img_fm, img_sz, img_sz)
        images.add_(1).div_(2.0)
        return make_grid(images, nrow=(n_columns if row_wise else n_images))
    
    
    # generate the grid / save it to a PNG file
    grid = get_grid(interpolations, True, 5)
    normalized_image = (grid.cpu().numpy().transpose((1, 2, 0)) - grid.cpu().numpy().min()) / (grid.cpu().numpy().max() - grid.cpu().numpy().min())
    return images, normalized_image

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessingApp()
    sys.exit(app.exec_())
