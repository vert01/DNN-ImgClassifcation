# Cloud Computing Project: Distributed DNN Model Training For Image Classification

This repository contains the code and instructions for setting up a distributed DNN model training project on Azure. The project focuses on training a DNN model for image detection using PyTorch and VMs in a distributed fashion.

## Setup Instructions

### 1. Set Up Azure Student Account

### 2. Create VMs

#### Creating VMs of Size B2s (The $30/mo option)

1. Navigate to "Virtual machines" in Azure Portal.
2. Click on "Add" to create a new VM.
3. Choose your subscription, resource group, and region.
4. Enter a name for your VM and select the appropriate size. Choose size **B2s** for this project.
5. Configure networking settings, including virtual network, subnet, and public IP address (if required).
6. Review the configuration and click "Create" to deploy the VM.

Repeat the above steps to create additional VMs if you need a distributed setup. Ensure both VMs created are IDENTICAL in order for the distributed training to work correctly.

### 3. Install Dependencies

#### On Each VM

- SSH or sudo into each VM 
- Sudo recommended: type ```ssh <username>@<VM IP Address>```, followed by the password you set when creating the VM.
- Run the following commands to install dependencies:

```bash
sudo apt-get update
sudo apt-get install -y python3-pip
pip3 install torch torchvision
```

### 4. Download Dataset

#### On Each VM

- Log into each VM and run the following commands to download the CIFAR-10 dataset:

```bash
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzvf cifar-10-python.tar.gz
```
- The dataset will already be broken down into testing and training directories.

### 5. Clone the Repository

- Clone this repository into each VM using:

```bash
git clone https://github.com/your-username/your-repo.git
```

### 6. Run the Training Script

- Navigate to the project directory on each VM and run the training script:

```bash
python3 script.py
```
- Changes can be made to the script using VIM or through the repo directly


