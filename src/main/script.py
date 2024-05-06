import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from gradcam import compute_gradcam

from data_preparation import prepare_data, plot_distribution, show_images
from main.models import ResNet, ResidualBlock, Bottleneck, SimpleCNN

# Config
EPOCHS = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001


# Training
def train_model(model, train_loader, device, epochs=EPOCHS):
    model.train()
    losses = []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}")
    return losses


# Evaluation
def evaluate_accuracy(model, data_loader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    return correct_predictions / total_predictions


def plot_losses(losses_cnn, losses_18, losses_34, losses_50):
    plt.figure(figsize=(10, 5))
    plt.plot(losses_cnn, color='purple', label='Simple CNN')
    plt.plot(losses_18, color='blue', label='ResNet-18')
    plt.plot(losses_34, color='red', label='ResNet-34')
    plt.plot(losses_50, color='green', label='ResNet-50')
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def show_gradcam_images(model, loader, class_idx, device, title):
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(title)
    image_count = 0
    for i, (images, labels) in enumerate(loader):
        for j in range(len(images)):
            if labels[j] == class_idx:
                image = images[j].unsqueeze(0).to(device)
                cam = compute_gradcam(model, image, class_idx, device)
                plt.subplot(10, 10, image_count + 1)
                plt.axis('off')
                plt.imshow(image[0].cpu().numpy().transpose(1, 2, 0), cmap='gray')
                plt.imshow(cam, cmap='viridis', alpha=0.5)
                image_count += 1
                if image_count >= 100:
                    break
        if image_count >= 100:
            break
    plt.show()


def main():
    train_loader, test_loader, class_names = prepare_data('training', 'test')

    plot_distribution(train_loader, 'Training Data Distribution')
    plot_distribution(test_loader, 'Test Data Distribution')

    show_images(train_loader, 'Training Images')
    show_images(test_loader, 'Test Images')

    model_cnn = SimpleCNN()
    model_18 = ResNet(ResidualBlock, [2, 2, 2, 2])
    model_34 = ResNet(ResidualBlock, [3, 4, 6, 3])
    model_50 = ResNet(Bottleneck, [3, 4, 6, 3], model_name='ResNet-50')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_18 = model_18.to(device)
    model_34 = model_34.to(device)
    model_50 = model_50.to(device)
    model_cnn = model_cnn.to(device)

    print("Training Simple CNN")
    losses_cnn = train_model(model_cnn, train_loader, device)
    print("Training ResNet-18")
    losses_18 = train_model(model_18, train_loader, device)
    print("Training ResNet-34")
    losses_34 = train_model(model_34, train_loader, device)
    print("Training ResNet-50")
    losses_50 = train_model(model_50, train_loader, device)

    plot_losses(losses_cnn, losses_18, losses_34, losses_50)

    test_accuracy_cnn = evaluate_accuracy(model_cnn, test_loader, device)
    test_accuracy_18 = evaluate_accuracy(model_18, test_loader, device)
    test_accuracy_34 = evaluate_accuracy(model_34, test_loader, device)
    test_accuracy_50 = evaluate_accuracy(model_50, test_loader, device)

    print(f'Test Accuracy Simple CNN: {test_accuracy_cnn * 100}%')
    print(f'Test Accuracy ResNet-18: {test_accuracy_18 * 100}%')
    print(f'Test Accuracy ResNet-34: {test_accuracy_34 * 100}%')
    print(f'Test Accuracy ResNet-50: {test_accuracy_50 * 100}%')

    show_gradcam_images(model_18, test_loader, 0, device, 'Grad-CAM Class 0')
    show_gradcam_images(model_34, test_loader, 0, device, 'Grad-CAM Class 0')
    show_gradcam_images(model_50, test_loader, 0, device, 'Grad-CAM Class 0')


if __name__ == '__main__':
    main()
