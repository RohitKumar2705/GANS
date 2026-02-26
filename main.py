from loader import load_CIFAR10
from models import Discriminator, Generator
from run import run
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
Z_DIM = 100
epochs = 100

# LOAD DATA
x_train_data = load_CIFAR10("dataset", BATCH_SIZE)

# MODELS
G = Generator(Z_DIM).to(device)
D = Discriminator().to(device)

z_val_ = torch.randn(BATCH_SIZE, Z_DIM)

for epoch in range(epochs):
    print(f"epoch: {epoch+1}/{epochs}")
    run(x_train_data, G, D, epoch, z_val_)