import torch
import os
import random
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from collections import Counter
from src.utils.ntk_util import get_full_data
from src.core.datasets import Cifar10, Cifar100, ImbalancedCifar10, MNIST, ImbalancedMNIST, FashionMNIST, SVHN


DATASETS = {
    'cifar10': Cifar10,
    'imb_cifar10': ImbalancedCifar10,
    'cifar100': Cifar100,
    'mnist': MNIST,
    'imb_mnist': ImbalancedMNIST,
    'fashionmnist': FashionMNIST,
    'svhn': SVHN
}

MODES = ['train', 'unlabeled', 'val']


class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def build(data_config, logger):
    data_name = data_config['name']
    root = data_config['root']
    batch_size = data_config['batch_size']
    test_batch_size = data_config.get('test_batch_size', batch_size)
    num_workers = data_config['num_workers']
    transform_config = data_config['transform']
    al_params = data_config['al_params']

    dataloaders = {}
    if data_name == 'cifar10' or data_name == 'cifar100':

        indices = list(range(50000))
        random.shuffle(indices)
        begin_index = al_params.get('init_num', al_params['add_num'])
        labeled_set = indices[:begin_index]
        unlabeled_set = indices[begin_index:]
        random.shuffle(unlabeled_set)
        subset = unlabeled_set[:al_params['num_subset']]

        dataloaders['subset'] = subset
        dataloaders['labeled_set'] = labeled_set
        dataloaders['unlabeled_set'] = unlabeled_set

        for mode in MODES:
            transform = compose_transforms(data_name, transform_config, mode, logger)
            if mode == 'train':
                dataset = DATASETS[data_name](
                    root, mode, download=True, logger=logger, transform=transform)
                dataloader = DataLoader(dataset, batch_size=batch_size,
                                        sampler=SubsetRandomSampler(labeled_set),
                                        num_workers=num_workers)
            elif mode == 'unlabeled':
                dataset = DATASETS[data_name](
                    root, mode, download=True, logger=logger, transform=transform)
                dataloader = DataLoader(dataset, batch_size=1,
                                        sampler=SubsetSequentialSampler(subset),
                                        num_workers=num_workers)
            elif mode == 'val':
                dataset = DATASETS[data_name](
                    root, mode, download=True, logger=logger, transform=transform)
                dataloader = DataLoader(dataset, batch_size=test_batch_size)

            dataloaders[mode] = dataloader

    elif data_name == 'imb_cifar10':
        indices = list(range(27500))
        random.shuffle(indices)
        begin_index = al_params.get('init_num', al_params['add_num'])
        labeled_set = indices[:begin_index]
        unlabeled_set = indices[begin_index:]
        random.shuffle(unlabeled_set)
        subset = unlabeled_set[:al_params['num_subset']]

        dataloaders['subset'] = subset
        dataloaders['labeled_set'] = labeled_set
        dataloaders['unlabeled_set'] = unlabeled_set

        for mode in MODES:
            transform = compose_transforms(data_name, transform_config, mode, logger)
            if mode == 'train':
                dataset = DATASETS[data_name](
                    root, mode, download=True, logger=logger, transform=transform)
                dataloader = DataLoader(dataset, batch_size=batch_size,
                                        sampler=SubsetRandomSampler(labeled_set),
                                        num_workers=num_workers)
            elif mode == 'unlabeled':
                dataset = DATASETS[data_name](
                    root, mode, download=True, logger=logger, transform=transform)
                dataloader = DataLoader(dataset, batch_size=1,
                                        sampler=SubsetSequentialSampler(subset),
                                        num_workers=num_workers)
            elif mode == 'val':
                dataset = DATASETS[data_name](
                    root, mode, download=True, logger=logger, transform=transform)
                dataloader = DataLoader(dataset, batch_size=test_batch_size)

            dataloaders[mode] = dataloader

    elif data_name == 'mnist':
        indices = list(range(60000))
        random.shuffle(indices)
        begin_index = al_params.get('init_num', al_params['add_num'])
        labeled_set = indices[:begin_index]
        unlabeled_set = indices[begin_index:]
        random.shuffle(unlabeled_set)
        subset = unlabeled_set[:al_params['num_subset']]

        dataloaders['subset'] = subset
        dataloaders['labeled_set'] = labeled_set
        dataloaders['unlabeled_set'] = unlabeled_set
        for mode in MODES:
            transform = compose_transforms(data_name, transform_config, mode, logger)
            if mode == 'train':
                dataset = DATASETS[data_name](
                    root, mode, download=True, logger=logger, transform=transform)
                dataloader = DataLoader(dataset, batch_size=batch_size,
                                        sampler=SubsetRandomSampler(labeled_set),
                                        num_workers=num_workers)
            elif mode == 'unlabeled':
                dataset = DATASETS[data_name](
                    root, mode, download=True, logger=logger, transform=transform)
                dataloader = DataLoader(dataset, batch_size=1,
                                        sampler=SubsetSequentialSampler(subset),
                                        num_workers=num_workers)
            elif mode == 'val':
                dataset = DATASETS[data_name](
                    root, mode, download=True, logger=logger, transform=transform)
                val_subset = list(range(len(dataset)))
                random.shuffle(val_subset)
                dataloader = DataLoader(dataset, batch_size=test_batch_size,
                                        num_workers=num_workers)

            dataloaders[mode] = dataloader

    elif data_name == 'imb_mnist':
        indices = list(range(32462))
        random.shuffle(indices)
        begin_index = al_params.get('init_num', al_params['add_num'])
        labeled_set = indices[:begin_index]
        unlabeled_set = indices[begin_index:]
        random.shuffle(unlabeled_set)
        subset = unlabeled_set[:al_params['num_subset']]

        dataloaders['subset'] = subset
        dataloaders['labeled_set'] = labeled_set
        dataloaders['unlabeled_set'] = unlabeled_set

        for mode in MODES:
            transform = compose_transforms(data_name, transform_config, mode, logger)
            if mode == 'train':
                dataset = DATASETS[data_name](
                    root, mode, download=True, logger=logger, transform=transform)
                dataloader = DataLoader(dataset, batch_size=batch_size,
                                        sampler=SubsetRandomSampler(labeled_set),
                                        num_workers=num_workers)
            elif mode == 'unlabeled':
                dataset = DATASETS[data_name](
                    root, mode, download=True, logger=logger, transform=transform)
                dataloader = DataLoader(dataset, batch_size=1,
                                        sampler=SubsetSequentialSampler(subset),
                                        num_workers=num_workers)
            elif mode == 'val':
                dataset = DATASETS[data_name](
                    root, mode, download=True, logger=logger, transform=transform)
                dataloader = DataLoader(dataset, batch_size=test_batch_size)

            dataloaders[mode] = dataloader

    elif data_name == 'fashionmnist':
        indices = list(range(60000))
        random.shuffle(indices)
        begin_index = al_params.get('init_num', al_params['add_num'])
        labeled_set = indices[:begin_index]
        unlabeled_set = indices[begin_index:]
        random.shuffle(unlabeled_set)
        subset = unlabeled_set[:al_params['num_subset']]

        dataloaders['subset'] = subset
        dataloaders['labeled_set'] = labeled_set
        dataloaders['unlabeled_set'] = unlabeled_set

        for mode in MODES:
            transform = compose_transforms(data_name, transform_config, mode, logger)
            if mode == 'train':
                dataset = DATASETS[data_name](
                    root, mode, download=True, logger=logger, transform=transform)
                dataloader = DataLoader(dataset, batch_size=batch_size,
                                        sampler=SubsetRandomSampler(labeled_set),
                                        num_workers=num_workers)
            elif mode == 'unlabeled':
                dataset = DATASETS[data_name](
                    root, mode, download=True, logger=logger, transform=transform)
                dataloader = DataLoader(dataset, batch_size=1,
                                        sampler=SubsetSequentialSampler(subset),
                                        num_workers=num_workers)
            elif mode == 'val':
                dataset = DATASETS[data_name](
                    root, mode, download=True, logger=logger, transform=transform)
                dataloader = DataLoader(dataset, batch_size=test_batch_size)

            dataloaders[mode] = dataloader

    elif data_name == 'svhn':
        indices = list(range(73257))
        random.shuffle(indices)
        begin_index = al_params.get('init_num', al_params['add_num'])
        labeled_set = indices[:begin_index]
        unlabeled_set = indices[begin_index:]
        random.shuffle(unlabeled_set)
        subset = unlabeled_set[:al_params['num_subset']]

        dataloaders['subset'] = subset
        dataloaders['labeled_set'] = labeled_set
        dataloaders['unlabeled_set'] = unlabeled_set

        for mode in MODES:
            transform = compose_transforms(data_name, transform_config, mode, logger)
            if mode == 'train':
                dataset = DATASETS[data_name](
                    root, mode, download=True, logger=logger, transform=transform)
                dataloader = DataLoader(dataset, batch_size=batch_size,
                                        sampler=SubsetRandomSampler(labeled_set),
                                        num_workers=num_workers)
            elif mode == 'unlabeled':
                dataset = DATASETS[data_name](
                    root, mode, download=True, logger=logger, transform=transform)
                dataloader = DataLoader(dataset, batch_size=1,
                                        sampler=SubsetSequentialSampler(subset),
                                        num_workers=num_workers)
            elif mode == 'val':
                dataset = DATASETS[data_name](
                    root, mode, download=True, logger=logger, transform=transform)
                dataloader = DataLoader(dataset, batch_size=test_batch_size,
                                        num_workers=num_workers)

            dataloaders[mode] = dataloader
    else:
        logger.error(
            'Specify valid data name'.format(DATASETS.keys()))

    assert len(set(labeled_set) & set(subset)) == 0
    assert len(set(subset) & set(unlabeled_set[len(subset):])) == 0
    X_train, y_train = get_full_data(dataloaders['train'].dataset, labeled_set)

    count_train = Counter(y_train.numpy())
    ordered_count = sorted(count_train.items(), key=lambda i: i[0])
    print('y train: {}'.format(str(ordered_count)), flush=True)

    return dataloaders


def update(cycle, dataloaders, arg, data_config, model_config, writer, save_dir):
    al_params = data_config['al_params']
    add_num, num_subset = al_params['add_num'], al_params['num_subset']
    batch_size, num_workers = data_config['batch_size'], data_config['num_workers']
    labeled_set, unlabeled_set, subset =\
        dataloaders['labeled_set'], dataloaders['unlabeled_set'], dataloaders['subset']

    # Update the labeled dataset, the unlabeled dataset, and subset
    queried_labeled_set = list(torch.tensor(subset)[arg][-add_num:].numpy())
    worst_queries = list(torch.tensor(subset)[arg][:add_num].numpy())
    new_labeled_set = list(labeled_set) + queried_labeled_set
    new_unlabeled_set = list(torch.tensor(subset)[arg][:-add_num].numpy()) + list(unlabeled_set[num_subset:])
    random.shuffle(new_unlabeled_set)
    new_subset = new_unlabeled_set[:num_subset]

    save_dir = Path(save_dir).joinpath('query_info')
    save_dir.mkdir(parents=True, exist_ok=True)
    if cycle == 0:
        path = save_dir.joinpath('train_cycle_0')
        np.save(str(path.absolute()), labeled_set)
    path = save_dir.joinpath('train_cycle_{}'.format(cycle+1))
    np.save(str(path.absolute()), new_labeled_set)
    path = save_dir.joinpath(('worst_cycle_{}'.format(cycle+1)))
    np.save(str(path.absolute()), worst_queries)

    dataloaders['labeled_set'] = new_labeled_set
    dataloaders['unlabeled_set'] = new_unlabeled_set
    dataloaders['subset'] = new_subset

    # check if label_set and subset are overlapping
    if len(set(new_labeled_set) & set(new_subset)) > 0:
        print('Problematic new_labeled_set and new_subset, they have overlapping points!')
        import IPython; IPython.embed()
    assert len(set(new_labeled_set) & set(new_subset)) == 0

    # Update train and unlabeled dataloaders
    train_dataset = dataloaders['train'].dataset
    unlabeled_dataset = dataloaders['unlabeled'].dataset
    dataloaders['train'] = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=SubsetRandomSampler(new_labeled_set), num_workers=num_workers)
    dataloaders['unlabeled'] = DataLoader(
        unlabeled_dataset, batch_size=1,
        sampler=SubsetSequentialSampler(new_subset), num_workers=num_workers)

    return dataloaders


def normalization_params(data_name, logger):
    if 'cifar10' in data_name or 'svhn' in data_name:
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif data_name == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    elif 'mnist' in data_name:
        mean = (0.0,)
        std = (1.0,)
    elif data_name == 'fashionmnist':
        mean = (0.0,)
        std = (1.0,)
    else:
        logger.error(
            'Specify valid data name'.format(DATASETS.keys()))
    return mean, std


def compose_transforms(data_name, transform_config, mode, logger):
    mean, std = normalization_params(data_name, logger)
    image_size = transform_config['image_size']
    crop_size = transform_config['crop_size']
    augment = transform_config.get('augment', False)

    if 'cifar10' in data_name  or 'mnist' in data_name \
            or data_name == 'fashionmnist' or data_name == 'svhn' or 'cifar100' in data_name:
        if mode == 'train' or mode == 'unlabeled':
            if augment:
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(size=crop_size, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
    else:
        logger.error(
            'Specify valid data name'.format(DATASETS.keys()))
    return transform


