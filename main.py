import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rps import datasetRPS

labels = {
            0: "paper",
            1: "rock",
            2: "scissors"
        }
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 75 * 75, 128)
        self.fc2 = nn.Linear(128, 3)  # 3 classes: rock, paper, scissors

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(-1, 32 * 75 * 75)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(networkp, train_loaderp, epoch):
    networkp.train()
    for batch_idx, (data, target) in enumerate(train_loaderp):
        optimizer.zero_grad()
        data = data.float()
        data = data.permute(0, 3, 1, 2)
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loaderp.dataset)} '
                  f'({100. * batch_idx / len(train_loaderp):.0f}%)]\tLoss: {loss.item():.6f}')
            torch.save(network.state_dict(), 'model.pth')
            torch.save(optimizer.state_dict(), 'optim.pth')

def test(networkp, test_loaderp):
    networkp.eval()
    test_loss = 0
    correct = 0
    debug = 0
    with torch.no_grad():
        for data, target in test_loaderp:
            data = data.float()
            data = data.permute(0, 3, 1, 2)
            # print(data.shape)
            output = networkp(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            debug += 1
    test_loss /= len(test_loaderp.dataset)
    print(f'\nTest set: Avg. loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loaderp.dataset)} ({100. * correct / len(test_loaderp.dataset):.0f}%)\n')

    if (100. * correct / len(test_loaderp.dataset)) > 60:
        with open(f'saved_dicts\model-{epoch}-{100. * correct / len(test_loaderp.dataset):.0f}.pth', "xb") as f:
            torch.save(network.state_dict(),f)

def test_frame(networkp, datap):
    networkp.eval()
    data = datap.float()
    data = data.permute(0, 3, 1, 2)
    output = networkp(data)
    pred = output.data.max(1, keepdim=True)[1].item()
    return labels[pred]

if __name__ == '__main__':
    n_epochs = 35
    batch_size_train = 5
    batch_size_test = 10
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    train_data = datasetRPS("custom-rps/custom_labels.csv")
    test_data = datasetRPS("custom-test-set/custom_test_labels.csv")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test, shuffle=True)

    network = Net()
    optimizer = optim.SGD(network.parameters(), lr = learning_rate, momentum = momentum)

    # test(network, test_loader)
    test(network, test_loader)
    for epoch in range(1, n_epochs + 1):
        train(network, train_loader, epoch)
        test(network, test_loader)

