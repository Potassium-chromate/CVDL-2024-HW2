from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from Q2_train import GANTrainer
import matplotlib.pyplot as plt
import numpy as np

class Q2:
    def __init__(self):
        self.gan = GANTrainer(batch_size=64, z_dim=100, num_epochs=50, learning_rate=0.0002)
    def show_training_original_image(self):
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Load the MNIST dataset
        mnist_train = MNIST(root='mnist_data', train=True, transform=transform, download=True)
        
        # Create DataLoader
        train_loader = DataLoader(mnist_train, batch_size=64, shuffle=False)

        # Get the first batch
        images, labels = next(iter(train_loader))
        
        # Plot the images
        fig = plt.figure(figsize=(8, 8))
        for i in range(64):
            ax = fig.add_subplot(8, 8, i + 1)
            ax.imshow(images[i][0], cmap='gray')  # Convert tensor to image
            ax.axis('off')  # Turn off axis for individual images
        
        # Adjust layout to remove space between subplots
        plt.subplots_adjust(wspace=0, hspace=0)  
        
        # Show the plot
        plt.show()
        
    def show_training_augmented_image(self):
        # Define transforms      
        rotated_transform = transforms.Compose([
            transforms.RandomRotation(60),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Load the MNIST dataset
        mnist_train = MNIST(root='mnist_data', train=True, transform=rotated_transform, download=True)
        
        # Create DataLoader
        train_loader = DataLoader(mnist_train, batch_size=64, shuffle=False)

        # Get the first batch
        images, labels = next(iter(train_loader))
        
        # Plot the images
        fig = plt.figure(figsize=(8, 8))
        for i in range(64):  # 8x8 grid
            ax = fig.add_subplot(8, 8, i + 1)
            ax.imshow(images[i][0], cmap='gray')  # Convert tensor to image
            ax.axis('off')  # Turn off axis for individual images
                
        # Adjust layout to remove space between subplots
        plt.subplots_adjust(wspace=0, hspace=0)  
        
        # Show the plot
        plt.show()
        
    def show_training_image(self):
        original_path = "./pictures/Training_dataset_original.png"
        augmented_path = "./pictures/Training_dataset_augmented.png"

        # Plot the two images side-by-side in a 1x2 layout
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        original_img = plt.imread(original_path)
        augmented_img = plt.imread(augmented_path)

        # Plot the original image
        ax[0].imshow(original_img)
        ax[0].axis('off')
        ax[0].set_title("Training Dataset (Original)", fontsize=20)

        # Plot the augmented image
        ax[1].imshow(augmented_img)
        ax[1].axis('off')
        ax[1].set_title("Training Dataset (Augmented)", fontsize=20)

        # Adjust layout
        plt.subplots_adjust(wspace=0.1, hspace=0)  # Slight gap for clarity between images
        plt.show()

    def show_model_structure(self):
        print(self.gan.generator)
        print("\n\n")
        print(self.gan.discriminator)
        
    def show_training_loss(self):
        loss_path = "./pictures/GD_Loss.png"
        loss_img = plt.imread(loss_path)
        plt.imshow(loss_img)
        plt.axis('off')
        plt.show()
    
    def inference(self):
        self.gan.load_generator('generator_model.pth')
        fake_images = self.gan.generate_fake_image()
    
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
        # Load the MNIST dataset
        mnist_train = MNIST(root='mnist_data', train=True, transform=transform, download=True)
    

        train_loader = DataLoader(mnist_train, batch_size=64, shuffle=False)
        real_images, labels = next(iter(train_loader))
    
        real_images = real_images.cpu().numpy()  # Convert to NumPy array  
        real_images = real_images.reshape(64, 28, 28)
    
        grid_images = np.zeros((224, 224), dtype=np.float32)  # Initialize an empty 224x224 grid
    
        for i in range(8):
            for j in range(8):
                grid_images[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = real_images[i * 8 + j]
    
        real_images_rgb = np.stack([grid_images] * 3, axis=-1)
    
        # Plot the two images side-by-side in a 1x2 layout
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    
        # Plot the real image
        ax[0].imshow(real_images_rgb)
        ax[0].axis('off')  # Turn off the axes
        ax[0].set_title("Real Image", fontsize=20)
    
      
        ax[1].imshow(fake_images)
        ax[1].axis('off')  # Turn off the axes
        ax[1].set_title("Fake Image", fontsize=20)
    
        # Adjust layout
        plt.subplots_adjust(wspace=0.1, hspace=0)  # Slight gap for clarity between images
        plt.show()

        
if __name__ == "__main__":
    # Usage
    q2 = Q2()
    #q2.show_training_original_image()
    #q2.show_training_augmented_image()
    #q2.show_training_image()
    #q2.show_model_structure()
    q2.inference()
    
    
