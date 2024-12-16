import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch


# DataLoader and Dataset (Clean Samples)
def data_loader(root_dir, dataset_name, preprocess, use_precal_mean_std=False, image_size=(224, 224),
                batch_size=30, train_dir='training', test_dir='testing', vald_dir='validation'):
    """
        Class to create Dataset and DataLoader from Image folder. 
        Args: 
            image_size -> size of the image after resize 
            batch_size 
            root_dir -> root directory of the dataset (downloaded dataset) 

        return: 
            dataloader -> dict includes dataloader for train/test and validation 
            dataset -> dict includes dataset for train/test and validation 
        
        """

    dirs = {'train': os.path.join(root_dir, train_dir),
            'valid': os.path.join(root_dir, vald_dir),
            'test': os.path.join(root_dir, test_dir)
            }

    if dataset_name == 'ChestXRay':
        if preprocess == 'grayscaled':
            data_transform = {
                'train': transforms.Compose([
                    transforms.Grayscale(num_output_channels=3),
                    # transforms.RandomRotation(20),
                    transforms.Resize(image_size),
                    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.ToTensor()
                ]),

                'valid': transforms.Compose([
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize(image_size),
                    transforms.ToTensor()
                ]),

                'test': transforms.Compose([
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize(image_size),
                    transforms.ToTensor()
                ])
            }
        elif preprocess == 'standardized':
            if not use_precal_mean_std:
                # Compute the mean and standard deviation
                pre_transform = transforms.Compose([
                    # transforms.RandomRotation(20),
                    transforms.Resize(image_size),
                    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.ToTensor()
                ])
                dataset = ImageFolder(root=dirs['train'], transform=pre_transform)
                mean = torch.zeros(3)
                std = torch.zeros(3)
                for image, _ in dataset:
                    mean += image.mean([1, 2])
                    std += image.std([1, 2])
                mean /= len(dataset)
                std /= len(dataset)
                print('Fresh calculated mean and std of the ChestXRay dataset (training set):')
                print(mean, std)
            else:
                mean = torch.tensor([0.5094, 0.5234, 0.5289])
                std = torch.tensor([0.2189, 0.2225, 0.2244])
                print('Using pre-calculated mean and std of the ChestXRay dataset (training set):')
                print(mean, std)

            data_transform = {
                'train': transforms.Compose([
                    # transforms.RandomRotation(20),
                    transforms.Resize(image_size),
                    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ]),

                'valid': transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ]),

                'test': transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
            }
        else:
            raise ValueError('Invalid preprocess type')

    elif dataset_name == 'ISICSkinCancer':
        if preprocess == 'grayscaled':
            data_transform = {
                'train': transforms.Compose([
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize(image_size),
                    transforms.ToTensor()
                ]),

                'valid': transforms.Compose([
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize(image_size),
                    transforms.ToTensor()
                ]),

                'test': transforms.Compose([
                    transforms.Grayscale(num_output_channels=3),
                    transforms.Resize(image_size),
                    transforms.ToTensor()
                ])
            }
        elif preprocess == 'standardized':
            if not use_precal_mean_std:
                # Compute the mean and standard deviation
                pre_transform = transforms.Compose([
                    # transforms.RandomRotation(20),
                    transforms.Resize(image_size),
                    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                    transforms.ToTensor()
                ])
                dataset = ImageFolder(root=dirs['train'], transform=pre_transform)
                mean = torch.zeros(3)
                std = torch.zeros(3)
                for image, _ in dataset:
                    mean += image.mean([1, 2])
                    std += image.std([1, 2])
                mean /= len(dataset)
                std /= len(dataset)
                print('Fresh calculated mean and std of the ISICSkinCancer dataset (training set):')
                print(mean, std)
            else:
                mean = torch.tensor([0.7187, 0.5684, 0.5464])
                std = torch.tensor([0.1212, 0.1325, 0.1434])
                print('Using pre-calculated mean and std of the ISICSkinCancer dataset (training set):')
                print(mean, std)

            data_transform = {
                'train': transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ]),

                'valid': transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ]),

                'test': transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
            }
        else:
            raise ValueError('Invalid preprocess type')
    else:
        raise ValueError('Dataset name is not valid')

    image_dataset = {x: ImageFolder(dirs[x], transform=data_transform[x])
                     for x in ('train', 'valid', 'test')}

    # data_loaders = {x: DataLoader(image_dataset[x], batch_size=batch_size,
    #                               shuffle=True, num_workers=4) for x in ['train']}
    #
    # data_loaders['test'] = DataLoader(image_dataset['test'], batch_size=batch_size,
    #                                   shuffle=False, num_workers=4, drop_last=True)
    #
    # data_loaders['valid'] = DataLoader(image_dataset['valid'], batch_size=batch_size,
    #                                    shuffle=False, num_workers=4, drop_last=True)

    dataset_size = {x: len(image_dataset[x]) for x in ['train', 'valid', 'test']}

    print([f'number of {i} images is {dataset_size[i]}' for i in (dataset_size)])

    class_idx = image_dataset['test'].class_to_idx
    print(f'Classes with index are: {class_idx}')

    class_names = image_dataset['test'].classes
    print(class_names)
    return None, image_dataset


# Dataloader and Dataset (Adversarial Samples)
def data_loader_attacks(root_dir, attack_name, image_size=(224, 224), batch_size=30):
    """
        Class to create Dataset and DataLoader from Image folder for adversarial samples generated. 
        Args: 
            root _dir: root directory of generated adversarial samples.
            attack_name: attack name that has folder in root_dir.
            image_size : size of the image after resize (224,224)
            batch_size

        return: 
            dataloader : dataloader for the attack
            dataset :  dataset for attack 
        
        """

    dirs = os.path.join(root_dir, f'Test_attacks_{attack_name}')
    data_transform = transforms.Compose([transforms.Resize(image_size),
                                         transforms.ToTensor()]
                                        )

    image_dataset = ImageFolder(dirs, transform=data_transform)

    # data_loaders = DataLoader(image_dataset, batch_size=batch_size,
    #                           shuffle=False, num_workers=4, drop_last=True)

    print(f'number of images is {len(image_dataset)}')

    class_idx = image_dataset.class_to_idx

    print(f'Classes with index are: {class_idx}')

    return None, image_dataset
