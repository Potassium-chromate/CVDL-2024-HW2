# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as utils
import matplotlib.pyplot as plt
import numpy as np

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Define the Generator model
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=3, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Define the GAN class
class GANTrainer:
    def __init__(self, batch_size=64, z_dim=100, num_epochs=50, learning_rate=0.0002):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.num_epochs = num_epochs
        self.lr = learning_rate

        # Initialize the models
        self.generator = Generator(z_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        # Define loss function and optimizers
        self.loss_fn = nn.BCELoss()
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))

        # Track losses and images
        self.g_losses = []
        self.d_losses = []
        self.generated_images = []

    def load_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train_discriminator(self, real_images):
        self.discriminator.zero_grad()

        # Real images
        real_images = real_images.to(self.device)
        real_labels = torch.ones(real_images.size(0), device=self.device)
        real_preds = self.discriminator(real_images)
        real_loss = self.loss_fn(real_preds.view(-1), real_labels)
        real_loss.backward()

        # Fake images
        noise = torch.randn(real_images.size(0), self.z_dim, 1, 1, device=self.device)
        fake_images = self.generator(noise)
        fake_labels = torch.zeros(real_images.size(0), device=self.device)
        fake_preds = self.discriminator(fake_images.detach())
        fake_loss = self.loss_fn(fake_preds.view(-1), fake_labels)
        fake_loss.backward()

        self.d_optimizer.step()

        return real_loss.item() + fake_loss.item()

    def train_generator(self):
        self.generator.zero_grad()

        noise = torch.randn(self.batch_size, self.z_dim, 1, 1, device=self.device)
        fake_images = self.generator(noise)
        target_labels = torch.ones(self.batch_size, device=self.device)
        fake_preds = self.discriminator(fake_images)
        g_loss = self.loss_fn(fake_preds.view(-1), target_labels)
        g_loss.backward()

        self.g_optimizer.step()

        return g_loss.item()

    def plot_training_curves(self):
        plt.figure(figsize=(10, 5))
        plt.title("Loss Trends for Generator and Discriminator")
        plt.plot(self.g_losses, label="Generator Loss")
        plt.plot(self.d_losses, label="Discriminator Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def save_model(self, file_path="generator_model.pth"):
        torch.save(self.generator.state_dict(), file_path)

    def generate_and_save_images(self):
        noise = torch.randn(self.batch_size, self.z_dim, 1, 1, device=self.device)
        with torch.no_grad():
            fake_images = self.generator(noise).detach().cpu()
        image_grid = utils.make_grid(fake_images, padding=2, normalize=True)
        plt.figure(figsize=(8, 8))
        plt.imshow(np.transpose(image_grid.numpy(), (1, 2, 0)))
        plt.axis("off")
        plt.show()

    def train(self):
        print(f"Training on {self.device}...")
        dataloader = self.load_data()

        for epoch in range(self.num_epochs):
            for batch_idx, (real_images, _) in enumerate(dataloader):
                if real_images.size(0) != self.batch_size:
                    continue

                d_loss = self.train_discriminator(real_images)
                g_loss = self.train_generator()

                self.d_losses.append(d_loss)
                self.g_losses.append(g_loss)

                if batch_idx % 50 == 0:
                    print(f"Epoch [{epoch + 1}/{self.num_epochs}] Batch [{batch_idx}/{len(dataloader)}]"
                          f"  D Loss: {d_loss:.4f}  G Loss: {g_loss:.4f}")

        self.generate_and_save_images()
        self.save_model()
        self.plot_training_curves()

if __name__ == "__main__":
    gan_trainer = GANTrainer(batch_size=64, z_dim=100, num_epochs=30, learning_rate=0.0002)
    gan_trainer.train()
