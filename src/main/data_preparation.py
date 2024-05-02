from collections import Counter

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def prepare_data(training_path, test_path):
    """
        Prepares the training and testing data for the model.

        This function uses torchvision.datasets.ImageFolder to load images from the given directories.
        The images are transformed by resizing them to 224x224 pixels and applying ColorJitter for data augmentation.

        DataLoader is used to create iterable objects for training and testing data with a batch size of 16.
        The training data is shuffled for better results.

        Parameters:
        training_path (str): The path to the training data directory.
        test_path (str): The path to the testing data directory.

        Returns:
        tuple: Tuple containing the DataLoader objects for the training and testing data, and the class names.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder(training_path, transform=transform)
    test_data = datasets.ImageFolder(test_path, transform=transform)

    class_names = train_data.classes

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=True)

    return train_loader, test_loader, class_names


def plot_distribution(loader, title):
    """
        Plots the distribution of classes in the data.

        Parameters:
        loader (DataLoader): The DataLoader object containing the data.
        title (str): The title of the plot.

        Returns:
        None
    """
    class_counts = Counter(label.item() for _, labels in loader for label in labels)
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)
    plt.show()


def show_images(loader, title):
    """
        Displays the images from the data.

        This function uses matplotlib to display images from the DataLoader object.
        The images are displayed in a 10x10 grid, a total of 100 images.
        If the DataLoader object contains fewer than 100 images, all available images will be displayed.

        Parameters:
        loader (DataLoader): The DataLoader object containing the data.
        title (str): The title of the plot.

        Returns:
        None
    """
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(title)

    for i, (images, labels) in enumerate(loader):
        for j in range(len(images)):
            plt.subplot(10, 10, i * 10 + j + 1)
            plt.axis('off')
            plt.imshow(images[j].permute(1, 2, 0))
            if i * 10 + j + 1 >= 100:
                break
        if i >= 9:
            break

    plt.show()
