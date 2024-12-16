from torch import nn, optim
import torch
from data.dataset import data_loader
import sys
import random
import numpy as np
sys.path.append("./models/")
from mlp import Classifier
import argparse
import os


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description='Train Mapping Networks for encoder blocks outputs')
parser.add_argument('--seed', type=int, help='Seed value', required=False)
parser.add_argument('--dataset', type=str, help='The dataset name', choices=['ChestXRay', 'ISICSkinCancer', 'PathMNIST',
                                                                             'RotatedMNIST'], required=True)
parser.add_argument('--root_dir', type=str, help='The directory to the data images', required=True)
parser.add_argument('--preprocess', type=str, help='The preprocess method', choices=['grayscaled', 'standardized'],
                    required=False, default='grayscaled')
parser.add_argument('--mn_idx', type=int, help='The index of the Mapping Network', choices=[0, 1, 2, 3, 4], required=True)
args = parser.parse_args()


if args.seed is None:
    seed = random.randint(0, 10000)
else:
    seed = args.seed
dataset_name = args.dataset
mn_idx = args.mn_idx
root_dir = args.root_dir
preprocess = args.preprocess
vit = torch.load('models/{}/vit_base_patch16_224_{}.pth'.format(dataset_name, dataset_name))

print('Training Mapping Network {} on {}'.format(mn_idx, root_dir))
set_seed(seed)
if dataset_name in ['ChestXRay', 'ISICSkinCancer']:
    num_classes = 2
    batch_size = 30
elif dataset_name == 'PathMNIST':
    num_classes = 9
    batch_size = 256
elif dataset_name == 'RotatedMNIST':
    num_classes = 10
    batch_size = 128
model = Classifier(num_classes=num_classes)
print(model)
model = model.cuda()
vit = vit.cuda()
loader_, dataset_ = data_loader(root_dir=root_dir, dataset_name=dataset_name, preprocess=preprocess, batch_size=batch_size)
train_loader = loader_['train']
valid_loader = loader_['valid']
train_dataset = dataset_['train']
valid_dataset = dataset_['valid']
# Define loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()
if dataset_name == 'ChestXRay':
    # Train MLPs_plain
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    epochs = 301
elif dataset_name == 'ISICSkinCancer':
    # Train MLPs_plain
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    epochs = 100
elif dataset_name == 'PathMNIST':
    # Train MLPs_plain
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    epochs = 301
elif dataset_name == 'RotatedMNIST':
    # Train MLPs_plain
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    epochs = 301

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training loop
best_acc = 0.0
best_train_acc = 0.0
for epoch in range(epochs):
    vit.eval()
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        x = vit.patch_embed(inputs)
        x_0 = vit.pos_drop(x)
        for i in range(0, mn_idx + 1):
            x_0 = vit.blocks[i](x_0)

        optimizer.zero_grad()

        outputs = model(x_0, dataset=dataset_name)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)

    print(f'Epoch {epoch} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    train_acc = epoch_acc

    # evaluate on test data
    vit.eval()
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in valid_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            x = vit.patch_embed(inputs)
            x_0 = vit.pos_drop(x)
            for i in range(0, mn_idx + 1):
                x_0 = vit.blocks[i](x_0)
            outputs = model(x_0, dataset=dataset_name)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(valid_dataset)
    epoch_acc = running_corrects.double() / len(valid_dataset)

    print(f'Epoch {epoch} Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    if epoch_acc > best_acc:
        best_train_acc = train_acc
        best_acc = epoch_acc
        if not os.path.exists('./models/{}/MLPs/'.format(dataset_name)):
            os.makedirs('./models/{}/MLPs/'.format(dataset_name))
        torch.save(model, './models/{}/MLPs/block_{}.pth'.format(dataset_name, mn_idx))

    if scheduler is not None:
        scheduler.step()

print('MLP {} on {} best train Acc: {:.4f} val Acc: {:.4f}'.format(mn_idx, dataset_name, best_train_acc, best_acc))
