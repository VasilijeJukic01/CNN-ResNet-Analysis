import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
            Initializes a residual block.

            Parameters:
                in_channels (int):      Number of input channels.
                out_channels (int):     Number of output channels.
                stride (int, optional): Stride value for the convolutional layers. Default is 1.
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
            Defines the forward pass of the residual block.

            Parameters:
                x (torch.Tensor): Input tensor.

            Returns:
                torch.Tensor: Output tensor after passing through the residual block.
        """
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        """
            Initializes a bottleneck block.

            Parameters:
                in_channels (int):      Number of input channels.
                out_channels (int):     Number of output channels.
                stride (int, optional): Stride value for the convolutional layers. Default is 1.
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        """
            Defines the forward pass of the bottleneck block.

            Parameters:
                x (torch.Tensor): Input tensor.

            Returns:
                torch.Tensor: Output tensor after passing through the bottleneck block.
        """
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=5, model_name='ResNet'):
        """
            Initializes a Residual Neural Network.

            Parameters:
                block (nn.Module):              Type of residual block to be used.
                layers (list):                  List specifying the number of blocks in each layer.
                num_classes (int, optional):    Number of output classes. Default is 5.
                model_name (str, optional):     Name of the ResNet model. Default is 'ResNet'.
        """
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0], model_name=model_name)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2, model_name=model_name)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2, model_name=model_name)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2, model_name=model_name)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * getattr(block, 'expansion', 1) if model_name == 'ResNet-50' else 512, num_classes)

        # Grad-CAM Hook
        self.gradients = None
        self.features = None
        self.layer4.register_forward_hook(self.forward_hook)
        self.layer4.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        """
            Hook function to capture feature maps during forward pass.

            Parameters:
                module (nn.Module):     The layer where the hook is registered.
                input (tuple):          Input tensors to the layer.
                output (torch.Tensor):  Output tensor from the layer.
        """
        self.features = output

    def backward_hook(self, module, grad_in, grad_out):
        """
            Hook function to capture gradients during backward pass.

            Parameters:
                module (nn.Module): The layer where the hook is registered.
                grad_in (tuple):    Gradient of the loss with respect to the input tensors.
                grad_out (tuple):   Gradient of the loss with respect to the output tensors.
        """
        self.gradients = grad_out[0]

    def make_layer(self, block, out_channels, blocks, stride=1, model_name='ResNet'):
        """
            Creates a layer of residual blocks.

            Parameters:
                block (nn.Module):          Type of residual block to be used.
                out_channels (int):         Number of output channels.
                blocks (int):               Number of blocks in the layer.
                stride (int, optional):     Stride value for the convolutional layers. Default is 1.
                model_name (str, optional): Name of the ResNet model. Default is 'ResNet'.

            Returns:
                nn.Sequential: A sequential layer of residual blocks.
        """
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * getattr(block, 'expansion', 1) if model_name == 'ResNet-50' else out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
            Defines the forward pass of the ResNet model.

            Parameters:
                x (torch.Tensor): Input tensor.

            Returns:
                torch.Tensor: Output tensor after passing through the ResNet model.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
