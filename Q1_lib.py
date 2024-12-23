import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision.models import vgg19_bn
from torchsummary import summary
from torchvision import transforms
from tkinter import Tk, filedialog
import torch.nn as nn
from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Q1:
    def __init__(self):
        self.loaded_image = None
    def load_image(self):
        root = Tk()
        root.withdraw()  
        root.attributes("-topmost", True)  # Bring file dialog to the front

        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
        )

        # Check if a file was selected
        if not file_path:
            print("No file selected.")
            return

        # Open and display the image
        self.loaded_image = Image.open(file_path)
    def show_augmented_image(self):
        # Define the path to the folder
        folder_path = './Q1_image/Q1_1/'
        
        # Get a list of all image file names in the folder
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Define transformations
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),  
            transforms.RandomVerticalFlip(p=1.0),    
            transforms.RandomRotation(30),           
        ])
        
        # Load and transform images
        images = []
        for file in image_files[:9]:  
            img_path = os.path.join(folder_path, file)
            img = Image.open(img_path).convert('RGB')
            augmented_img = transform(img)            
            images.append([file.split(".")[0],augmented_img])

        # Plot original and augmented images side by side
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        for i in range(3):
            for j in range(3):
                idx = 3 * i + j
                axes[i, j].imshow(images[idx][1])
                axes[i, j].set_title(images[idx][0])
                axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.show()

    
    def show_structure(self):

        model = vgg19_bn(num_classes=10)
        summary(model, input_size=(3, 32, 32), device='cpu')
    
    def show_acc_and_loss(self):
        acc_image = Image.open("./acc_curve.png")
        loss_image = Image.open("./loss_curve.png")
        
        # Plot the accuracy curve
        plt.figure(figsize=(6, 6))
        plt.imshow(acc_image, cmap = None)
        plt.title("Accuracy Curve")
        plt.axis('off')
        plt.show()
        
        # Plot the loss curve
        plt.figure(figsize=(6, 6))
        plt.imshow(loss_image, cmap = None)
        plt.title("Loss Curve")
        plt.axis('off')  
        plt.show()
        
    def inference(self):
        if(self.loaded_image == None):
            print("The picture hasn't loaded yet!")
            return
        
        model_path = "./vgg19_cifar10.pth"
        model = models.vgg19_bn(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Set model to evaluation mode
        
        class_labels = ['airplane', 'automobile', 'bird', 'cat', 
                    'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
        # Image preprocessing
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load and preprocess the image
        image_tensor = transform(self.loaded_image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = outputs.max(1)
            class_label = class_labels[predicted.item()]
            probabilities = nn.functional.softmax(outputs, dim=1).squeeze().numpy()
        
        # Show the predict loaded image and predicted class label
        plt.figure(figsize=(6, 6))
        plt.imshow(self.loaded_image, cmap = None)
        plt.title("Predict = " + class_label)
        plt.axis('off')
        plt.show()
        
        plt.figure(figsize=(10, 6))
        plt.bar(class_labels, probabilities, color='blue')
        plt.xlabel('Class Labels')
        plt.ylabel('Probability')
        plt.title('Class Probabilities')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.grid(axis='y')
        plt.show()

        
if __name__ == "__main__":
    Q1 = Q1()
    #Q1.show_argumented_image()
    #Q1.show_structure()
    #Q1.show_acc_and_loss()
    Q1.load_image()
    predicted = Q1.inference()