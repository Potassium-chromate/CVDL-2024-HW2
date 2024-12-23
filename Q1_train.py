import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


class VGG19():
    def __init__(self, batch_size = 64, learning_rate = 0.001, num_epochs = 20): 
        # Check device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Hyperparameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.train_loader = None
        self.test_loader = None
        self.VGG19 = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        # Lists to store loss and accuracy
        self.train_losses, self.val_losses = [], []
        self.train_accuracies, self.val_accuracies = [], []
        
    def data_preprocessing(self):
        # Data preprocessing and augmentation
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load CIFAR-10 dataset
        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
    
    def load_model(self):
        model = models.vgg19_bn(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)  
        self.model = model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        
        # Learning rate scheduler
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.9 ** epoch)
        
    # Training function
    def train(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Wrap the DataLoader with tqdm
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for images, labels in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)
    
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
    
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Accuracy': f"{100. * correct / total:.2f}%"
            })
    
        train_loss = total_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        return train_loss, train_acc

    # Validation function
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
    
        progress_bar = tqdm(self.test_loader, desc="Validating", leave=False)
        
        with torch.no_grad():
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
    
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
    
                # Metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Accuracy': f"{100. * correct / total:.2f}%"
                })
    
        val_loss = total_loss / len(self.test_loader)
        val_acc = 100. * correct / total
        return val_loss, val_acc
    
    def training_loop(self):
        print(f"Using device: {self.device}")
        # Training loop
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self.train()
            val_loss, val_acc = self.validate()
            
            # Append metrics to lists
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
        
            print(f'Epoch {epoch+1}/{self.num_epochs}, '
                  f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')

        # Save the model
        torch.save(self.model.state_dict(), 'vgg19_cifar10.pth')
        print("Model saved as vgg19_cifar10.pth")
    
    def plot_curves(self):
        # Plot loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.show()
        
        # Plot accuracy curves
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_accuracies, label='Training Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.show()
        
if __name__ == "__main__":
    model = VGG19(batch_size = 64, learning_rate = 0.01, num_epochs = 20)
    model.data_preprocessing()
    model.data_preprocessing()
    model.load_model()
    model.training_loop()
    model.plot_curves()
