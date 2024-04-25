import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Initialize distributed training environment
def ddp_setup():
    init_process_group(backend='gloo')

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
    
class Trainer:
  def __init__(
      self,
      model: nn.Module,
      train_data: DataLoader,
      optimizer: optim.Optimizer,
  ) -> None:
      self.local_rank = int(os.environ["LOCAL_RANK"])
      self.global_rank = int(os.environ["RANK"])
      self.train_data = train_data
      self.model = model
      self.optimizer = optimizer

      self.model = DDP(self.model)


  def train(self, max_epochs: int):
      criterion = nn.CrossEntropyLoss()
      
      # Training
      for epoch in range(max_epochs):  # loop over the dataset multiple times
          running_loss = 0.0
         
          for i, data in enumerate(self.train_data, 0):
              # get the inputs; data is a list of [inputs, labels]
              inputs, labels = data

              self.optimizer.zero_grad()

              outputs = self.model(inputs)
              loss = criterion(outputs, labels)
              loss.backward()
              self.optimizer.step()

              # print statistics
              running_loss += loss.item()
              if i % 2000 == 1999:  # print every 2000 mini-batches
                  print(f"[RANK{self.global_rank}, Epoch{epoch + 1:3d}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                  running_loss = 0.0

      print(f"Finished Training on Rank {self.global_rank}")

      # Save the trained model
      if self.global_rank == 0:  # Save only from one process to avoid multiple saves
          torch.save(self.model.module.state_dict(), 'trained_model.pth')

def load_train_objs():
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)  # load your dataset
  model = Net()  # load your model
  optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  return train_set, model, optimizer

def prepare_dataloader(dataset: datasets, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main(total_epochs: int, batch_size: int):
    ddp_setup()
    train_set, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(train_set, batch_size)
    trainer = Trainer(model, train_data, optimizer)
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description='simple distributed training job')
  parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
  args = parser.parse_args()
  
  main(args.total_epochs, 4)