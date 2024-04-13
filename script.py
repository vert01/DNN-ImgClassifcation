import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import Process

# Initialize distributed training environment
def init_process(rank, size, fn, backend='gloo'):
    os.environ['MASTER_ADDR'] = 'master_vm_ip_address' # Replace with IP address of VM that the script is being run on
    os.environ['MASTER_PORT'] = '1234'  # Choose an available port number (higher than 1024)
    os.environ['WORLD_SIZE'] = str(size)  # Number of VMs participating in training (so 2)
    os.environ['RANK'] = str(rank)  # Rank of the current VM (0 or 1)

    dist.init_process_group(backend=backend, init_method='env://', rank=rank, world_size=size)
    fn(rank, size)

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(rank, size):
    # Set up model, loss function, optimizer, and data loaders
    model = Net()
    model = DDP(model) #removed device_ids=[rank] was throwing device type error
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=size, rank=rank)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=False, num_workers=2, sampler=train_sampler)

    # Training
    for epoch in range(5):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f"[{rank}, {epoch + 1:3d}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    print(f"Finished Training on Rank {rank}")

    # Saves the training model to 'trained_model.pth'
    if rank == 0:  # Save only from one process to avoid multiple saves
        torch.save(model.module.state_dict(), 'trained_model.pth')

if __name__ == '__main__':
    # Set up multiprocessing
    import multiprocessing as mp
    mp.set_start_method('spawn')

    # Run the training function on each VM
    size = 2  # Number of VMs
    processes = []
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, train))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
