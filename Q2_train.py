# -*- coding: utf-8 -*-
import torch
from torch import nn
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 3, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class GAN:
    def __init__(self, batch_size=64, z_dim=100, epochs=100, lr=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.epochs = epochs
        self.lr = lr
        self.G = Generator(self.z_dim).to(self.device)
        self.D = Discriminator().to(self.device)
        self.criterion = nn.BCELoss()
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr)
        self.G_losses = []
        self.D_losses = []
        self.img_list = []

    def get_dataloader(self):
        """Load the MNIST dataset and apply transformations"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        
        train_set = dset.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
        test_set = dset.MNIST(root='./mnist_data/', train=False, transform=transform, download=False)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=self.batch_size,
            shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=self.batch_size,
            shuffle=False        
        )

        return train_loader, test_loader

    def D_train(self, x):
        """Train the Discriminator with real and fake images"""
        self.D.zero_grad()
        
        # Train on real images
        x_real = x.to(self.device)
        y_real = torch.ones(self.batch_size,).to(self.device)
        x_real_predict = self.D(x_real)
        D_real_loss = self.criterion(x_real_predict.view(-1), y_real)
        D_real_loss.backward()

        # Train on fake images
        noise = torch.randn(self.batch_size, self.z_dim, 1, 1, device=self.device)
        y_fake = torch.zeros(self.batch_size,).to(self.device)
        x_fake = self.G(noise)
        x_fake_predict = self.D(x_fake)
        D_fake_loss = self.criterion(x_fake_predict.view(-1), y_fake)
        D_fake_loss.backward()

        # Total loss and update
        D_total_loss = D_real_loss + D_fake_loss
        self.D_optimizer.step()

        return D_total_loss.item()

    def G_train(self):
        """Train the Generator to fool the Discriminator"""
        self.G.zero_grad()

        noise = torch.randn(self.batch_size, self.z_dim, 1, 1, device=self.device)
        y_target = torch.ones(self.batch_size,).to(self.device)
        x_fake = self.G(noise)
        y_fake = self.D(x_fake)
        G_loss = self.criterion(y_fake.view(-1), y_target)
        G_loss.backward()
        self.G_optimizer.step()

        return G_loss.item()

    def Draw_plot(self):
        """Plot the training loss curves for Generator and Discriminator"""
        plt.figure()
        plt.title("D Loss, G Loss / Iteration")
        plt.plot(self.G_losses, label='G')
        plt.plot(self.D_losses, label='D')
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
    
    def plot_last_image(self, img):
        # Plot the image
        plt.figure(figsize=(8, 8))
        plt.imshow(np.transpose(img.cpu(), (1, 2, 0)))  # Convert from Tensor to NumPy
        plt.axis('off')  # Hide axes
        plt.show()

    def save_generator(self, model_path='generator.pth'):
        """Save the Generator model weights"""
        torch.save(self.G.state_dict(), model_path)

    def load_generator(self, model_path='generator.pth'):
        """Load the saved Generator model weights"""
        self.G.load_state_dict(torch.load(model_path))
        self.G.eval()  # Set the model to evaluation mode

    def generate_fake_image(self):
        """Generate and plot a batch of images using the loaded Generator"""
        noise = torch.randn(self.batch_size, self.z_dim, 1, 1, device=self.device)
        with torch.no_grad():
            fake_images = self.G(noise).detach().cpu()
            
        img_grid = vutils.make_grid(fake_images, padding=2, normalize=True)
        
        return np.transpose(img_grid.numpy(), (1, 2, 0))
        
    def Train(self):
        """Main training loop for GAN"""
        print("\nStarting GAN training...")
        if self.device.type == 'cuda':
            print(f"Using {torch.cuda.get_device_name(0)}\n")
        else:
            print("Using CPU\n")
        
        train_loader, test_loader = self.get_dataloader()

        for epoch in range(self.epochs):
            for id, (train_x, _) in enumerate(train_loader):
                if len(train_x) == 64:  # Ensure correct batch size
                    self.D_losses.append(self.D_train(train_x))
                    self.G_losses.append(self.G_train())

                    if id % 50 == 0:
                        print(f'[{epoch+1}/{self.epochs}]\t[{id}/{len(train_loader)}]\t Loss D: {np.mean(self.D_losses):.4f} \tLoss G: {np.mean(self.G_losses):.4f}')

                    # Save generated image every 10 iterations
                    if id % 10 == 0:
                        with torch.no_grad():
                            noise = torch.randn(self.batch_size, self.z_dim, 1, 1, device=self.device)
                            fake = self.G(noise).detach().cpu()
                        self.img_list.append(vutils.make_grid(fake, padding=0, normalize=True))

        self.save_generator()  # Save the Generator after training
        self.Draw_plot()
        self.plot_last_image(self.img_list[-1])

if __name__ == '__main__':
    # Run GAN training with specified parameters
    train = GAN(batch_size=64, z_dim=100, epochs=30, lr=0.001)
    #train.Train()
    #trained_gan = GAN(batch_size=64, z_dim=100, epochs=30, lr=0.001)
    #trained_gan.load_generator('generator.pth')
    #fake_images = trained_gan.plot_batch()