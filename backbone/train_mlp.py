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


parser = argparse.ArgumentParser(description='Train MLPs for ViT blocks outputs')
parser.add_argument('--seed', type=int, help='Seed value, 0, 1000, 4000', required=True)
parser.add_argument('--dataset', type=str, help='The dataset name', choices=['ChestXRay', 'ISICSkinCancer', 'PathMNIST',
                                                                             'RotatedMNIST'], required=True)
parser.add_argument('--root_dir', type=str, help='The directory to the data images', required=True)
parser.add_argument('--preprocess', type=str, help='The preprocess method', choices=['grayscaled', 'standardized'],
                    required=True)
parser.add_argument('--mlp_idx', type=int, help='The index of the output block', required=True)
args = parser.parse_args()


seed = args.seed
dataset_name = args.dataset
mlp_idx = args.mlp_idx
root_dir = args.root_dir
preprocess = args.preprocess
vit = torch.load('models/{}_{}/seed{}/vit_base_patch16_224_in21k_{}.pth'.format(dataset_name, preprocess, seed,
                                                                                dataset_name))

print('Training MLP {} with seed {} on {}'.format(mlp_idx, seed, root_dir))
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
test_loader = loader_['test']
train_dataset = dataset_['train']
test_dataset = dataset_['test']
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
        for i in range(0, mlp_idx + 1):
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

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            x = vit.patch_embed(inputs)
            x_0 = vit.pos_drop(x)
            for i in range(0, mlp_idx + 1):
                x_0 = vit.blocks[i](x_0)
            outputs = model(x_0, dataset=dataset_name)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(test_dataset)
    epoch_acc = running_corrects.double() / len(test_dataset)

    print(f'Epoch {epoch} Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    if epoch_acc > best_acc:
        best_train_acc = train_acc
        best_acc = epoch_acc
        if not os.path.exists('./models/{}_{}/seed{}/MLPs/'.format(dataset_name, preprocess, seed)):
            os.makedirs('./models/{}_{}/seed{}/MLPs/'.format(dataset_name, preprocess, seed))
        torch.save(model, './models/{}_{}/seed{}/MLPs/block_{}.pth'.format(dataset_name, preprocess, seed, mlp_idx))

    if scheduler is not None:
        scheduler.step()

print('MLP {} seed {} on {} {} best train Acc: {:.4f} val Acc: {:.4f}'.format(mlp_idx, seed, preprocess, dataset_name,
                                                                              best_train_acc, best_acc))


