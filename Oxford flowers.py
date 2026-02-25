import os
import tarfile
import urllib.request
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Dataset URL
url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
data_dir = "flowers102"

# Download dataset
os.makedirs(data_dir, exist_ok=True)
file_path = os.path.join(data_dir, "102flowers.tgz")

if not os.path.exists(file_path):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, file_path)

# Extract dataset
print("Extracting dataset...")
with tarfile.open(file_path) as tar:
    tar.extractall(path=data_dir)

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Load dataset
dataset = datasets.ImageFolder(data_dir, transform=transform)

# Data loader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

print("Dataset Loaded:", len(dataset))

import tensorflow as tf
import tensorflow_datasets as tfds

# Load Oxford Flowers 102 dataset
dataset, info = tfds.load(
    'oxford_flowers102',
    with_info=True,
    as_supervised=True
)

train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

print(info)
IMG_SIZE = 224
BATCH_SIZE = 32

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image, label

train_data = train_data.map(preprocess).batch(BATCH_SIZE)
val_data = val_data.map(preprocess).batch(BATCH_SIZE)
import matplotlib.pyplot as plt

for images, labels in train_data.take(1):
    plt.imshow(images[0])
    plt.title(labels[0])
    plt.show()
