from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
from PIL import Image

transform = transforms.Compose([
        transforms.ToTensor(),
        # normalize(mean, std, inplace=False)
        # output = (input - mean) / std
        #transforms.Normalize((0.1307, ), (0.3081, ))
])

class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, height, width, trans=transform):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data = pd.read_csv(csv_path, header=None)
        self.labels = np.asarray(self.data.iloc[:, 0])
        self.height = height
        self.width = width
        self.trans = transform

    def __getitem__(self, index):
        single_image_label = self.labels[index]
        # Read each 784 pixels and reshape the 1D array ([784]) to 2D array ([28,28])
        img_as_np = np.asarray(self.data.iloc[index][1:]).reshape(28,28).astype('uint8')
	# Convert image from numpy array to PIL image, mode 'L' is for grayscale
        img_as_img = Image.fromarray(img_as_np)
        img_as_img = img_as_img.convert('L')
        # Transform image to tensor
        if self.trans is not None:
            img_as_tensor = self.trans(img_as_img)
        # Return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data.index)


