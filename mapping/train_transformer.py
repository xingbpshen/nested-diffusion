from torch import nn, optim
import torch
from data.dataset import data_loader
import sys
import random
import numpy as np
import argparse
import os
import timm

sys.path.append("./models/")


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description='Train Transformer')
parser.add_argument('--seed', type=int, help='Seed value', required=False)
parser.add_argument('--dataset', type=str, help='The dataset name', choices=['ChestXRay', 'ISICSkinCancer', 'PathMNIST',
                                                                             'RotatedMNIST'],
                    required=True)
parser.add_argument('--root_dir', type=str, help='The directory to the data images', required=True)
parser.add_argument('--preprocess', type=str, help='The preprocess method', choices=['grayscaled', 'standardized'],
                    required=False, default='grayscaled')
parser.add_argument('--model_type', type=str, help='The model type', choices=['resnet18',
                                                                              'resnet50',
                                                                              'efficientnetv2',
                                                                              'deit',
                                                                              'vit',
                                                                              'convit'], required=False, default='vit')
args = parser.parse_args()
if args.seed is None:
    seed = random.randint(0, 10000)
else:
    seed = args.seed
dataset_name = args.dataset
root_dir = args.root_dir
model_type = args.model_type
preprocess = args.preprocess
if dataset_name in ['ChestXRay', 'ISICSkinCancer']:
    num_classes = 2
    batch_size = 30
elif dataset_name == 'PathMNIST':
    num_classes = 9
    batch_size = 256
elif dataset_name == 'RotatedMNIST':
    num_classes = 10
    batch_size = 64

print('Training {} on {} with seed {}:'.format(model_type, root_dir, seed))
set_seed(seed)
if model_type == 'resnet18':
    model = torch.load('./models/base/resnet18.pth')
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    save_model_name = 'resnet18'
elif model_type == 'resnet50':
    model = torch.load('./models/base/resnet50.pth')
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    save_model_name = 'resnet50'
elif model_type == 'efficientnetv2':
    model = torch.load('./models/base/efficientnetv2_l.pth')
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    save_model_name = 'efficientnetv2_l'
elif model_type == 'deit':
    model = torch.load('./models/base/deit_base_patch16_224.pth')
    model.head = torch.nn.Linear(model.head.in_features, num_classes)
    save_model_name = 'deit_base_patch16_224'
elif model_type == 'vit':
    model = timm.create_model('vit_base_patch16_224', pretrained=False)
    model.head = torch.nn.Linear(model.head.in_features, num_classes)
    save_model_name = 'vit_base_patch16_224'
elif model_type == 'convit':
    model = torch.load('./models/base/convit_base.pth')
    model.head = torch.nn.Linear(model.head.in_features, num_classes)
    save_model_name = 'convit_base'
else:
    raise NotImplementedError
loader_, dataset_ = data_loader(root_dir=root_dir, dataset_name=dataset_name, preprocess=preprocess,
                                batch_size=batch_size)
train_loader = loader_['train']
valid_loader = loader_['valid']
train_dataset = dataset_['train']
valid_dataset = dataset_['valid']
model = model.cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
if model_type in ['resnet18', 'resnet50', 'efficientnetv2', 'deit', 'vit', 'convit']:
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    epochs = 200
else:
    raise NotImplementedError

# Training loop
best_acc = 0.0
best_train_acc = 0.0
for epoch in range(epochs):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # backward + optimize
        loss.backward()
        # optimizer.step()
        optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)

    print(f'Epoch {epoch} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    train_acc = epoch_acc

    # evaluate on test data
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in valid_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(valid_dataset)
    epoch_acc = running_corrects.double() / len(valid_dataset)

    print(f'Epoch {epoch} Test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_train_acc = train_acc
        if not os.path.exists('./models/{}/'.format(dataset_name)):
            os.makedirs('./models/{}/'.format(dataset_name))
        torch.save(model, './models/{}/{}_{}.pth'.format(dataset_name, save_model_name,
                                                            dataset_name))

    if scheduler is not None:
        scheduler.step()

print('Model {} on {} best train Acc: {:.4f} val Acc: {:.4f}'.format(model_type, dataset_name, best_train_acc, best_acc))
