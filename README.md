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
sudo apt-get net-tools
ifconfig
    - Note: This is used to get the VMs IP (not the same IP used to ssh into VM)
    - Note: IP will be under eht0, first line, named inet6 
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
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=IP_ADDR_MASTER_NODE:MASTER_PORT multiVM_model.py NUM_OF_EPOCHS
```
- torchrun is used to run a distributed training script
- nproc_per_node is the number of processes per node (in this case 2 CPUs per VM)
- nnodes is the number of nodeds (in this case it's 2 VMs)
- node_rank specifies the rank of the current node (one VM would have 0 and the other 1)
- rdzv_id is the ID of the rendezvous for coordination
- rdzv_endpoint should contain the IP address and port of the master node where the rdzv service is taken place
    - The IP address is found on the Azure portal under the VM's overview -> Networking -> Private IP address
    - The port an avaiable port
- multiVM_model.py is the distributive training script we are running
- anything after the script are agruments taken in by the script (for us it is the number of epochs that the training script will run)

- Changes can be made to the script using VIM or through the repo directly

### 7. Run the Testing Script
```bash
python3 model.py
```
- This script will calculate the accuracy of the trained model


