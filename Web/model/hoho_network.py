import torch
import torch.nn as nn
import torch.nn.functional as F

class BoardNetwork(nn.Module):

    def __init__(self):
        super(BoardNetwork, self).__init__()
        self.conv1 = nn.Conv2d(8, 14, kernel_size = 5, padding = 2)
        self.conv2 = nn.Conv2d(14, 14, kernel_size = 5, padding = 2)
        self.conv3 = nn.Conv2d(14, 14, kernel_size = 5, padding = 2)
        self.conv4 = nn.Conv2d(14, 10, kernel_size = 5, padding = 2)
        self.conv5 = nn.Conv2d(10, 2, kernel_size = 5, padding = 2)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.softmax(x)

        return x
    


if __name__ == "__main__":

    board_net = BoardNetwork()
    input_tensor = torch.randn(1, 8, 10, 9)
    output_tensor = board_net(input_tensor)

    print("Input Tensor Size:", input_tensor.size())
    print("Output Tensor Size:", output_tensor.size())

    print(f"Output: {output_tensor}")
