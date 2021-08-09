from tensorflow.keras.datasets import mnist
from torchvision import datasets

# mymnist = datasets.MNIST(root='data', train=True, download=True)

class Data_Loader():
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def __len__(self):
        return len(self.x_train) +len(self.x_test)

    def __getitem__(self, item):
        return self.x_train, self.y_train, self.x_test, self.y_test
