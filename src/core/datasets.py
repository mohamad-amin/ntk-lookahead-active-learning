import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets


class CustomDataset(Dataset):
    def __init__(self, root, logger, mode, transform=None):
        self.root = root
        self.logger = logger
        self.mode = mode
        self.transform =transform
        self.data = self._load_data()

    def _load_data(self):
        data = []
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = sample.path
        label = sample.label
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return (image, label)


class MNIST(datasets.MNIST):
    def __init__(self, root, mode, download, logger, transform=None):
        train = True if mode == 'train' or mode == 'unlabeled' else False
        super(MNIST, self).__init__(root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.array(Image.fromarray(img.numpy(), mode='L'))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        input_dict = {
            'inputs': img,
            'labels': target,
            'indices': index
        }
        return input_dict


class ImbalancedMNIST(Dataset):
    def __init__(self, root, mode, download, logger, transform=None):
        self.dataset = MNIST(
            root=root, mode=mode, download=download, logger=logger, transform=transform)
        self.train = True if mode == 'train' or mode == 'unlabeled' else False
        self.imbal_class_prop = [0.1] * 5 + [1.0] * 5
        self.num_classes = 10
        self.idxs = self.resample()

    def get_labels_and_class_counts(self, label_list):
        labels = np.array(label_list)
        _, class_counts = np.unique(labels, return_counts=True)
        return labels, class_counts

    def resample(self):
        '''
        Resample the indices to create an artificially imbalanced dataset.
        '''
        if self.train:
            targets, class_counts = self.get_labels_and_class_counts(
                self.dataset.train_labels)
        else:
            targets, class_counts = self.get_labels_and_class_counts(
                self.dataset.test_labels)
        # Get class indices for resampling
        class_indices = [np.where(targets == i)[0] for i in range(self.num_classes)]
        # Reduce class count by proportion
        self.imbal_class_counts = [
            int(count * prop)
            for count, prop in zip(class_counts, self.imbal_class_prop)
        ]
        # Get class indices for reduced class count
        idxs = []
        for c in range(self.num_classes):
            imbal_class_count = self.imbal_class_counts[c]
            idxs.append(class_indices[c][:imbal_class_count])
        idxs = np.hstack(idxs)
        self.labels = targets[idxs]
        return idxs

    def __getitem__(self, index):
        input_dict = self.dataset[self.idxs[index]]
        return input_dict

    def __len__(self):
        return len(self.idxs)


class Cifar10(datasets.CIFAR10):
    def __init__(self, root, mode, download, logger, transform=None):
        train = True if mode == 'train' or mode == 'unlabeled' else False
        super(Cifar10, self).__init__(root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.array(Image.fromarray(img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        input_dict = {
            'inputs': img,
            'labels': target,
            'indices': index
        }
        return input_dict


class Cifar100(datasets.CIFAR100):
    def __init__(self, root, mode, download, logger, transform=None):
        train = True if mode == 'train' or mode == 'unlabeled' else False
        super(Cifar100, self).__init__(root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.array(Image.fromarray(img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        input_dict = {
            'inputs': img,
            'labels': target,
            'indices': index
        }
        return input_dict


class ImbalancedCifar10(Dataset):
    def __init__(self, root, mode, download, logger, transform=None):
        self.dataset = Cifar10(
            root=root, mode=mode, download=download, logger=logger, transform=transform)
        self.train = True if mode == 'train' or mode == 'unlabeled' else False
        self.imbal_class_prop = [0.1] * 5 + [1.0] * 5
        self.num_classes = 10
        self.idxs = self.resample()

    def get_labels_and_class_counts(self, label_list):
        labels = np.array(label_list)
        _, class_counts = np.unique(labels, return_counts=True)
        return labels, class_counts

    def resample(self):
        '''
        Resample the indices to create an artificially imbalanced dataset.
        '''
        if self.train:
            targets, class_counts = self.get_labels_and_class_counts(
                self.dataset.targets)
        else:
            targets, class_counts = self.get_labels_and_class_counts(
                self.dataset.targets)
        # Get class indices for resampling
        class_indices = [np.where(targets == i)[0] for i in range(self.num_classes)]
        # Reduce class count by proportion
        self.imbal_class_counts = [
            int(count * prop)
            for count, prop in zip(class_counts, self.imbal_class_prop)
        ]
        # Get class indices for reduced class count
        idxs = []
        for c in range(self.num_classes):
            imbal_class_count = self.imbal_class_counts[c]
            idxs.append(class_indices[c][:imbal_class_count])
        idxs = np.hstack(idxs)
        self.labels = targets[idxs]
        return idxs

    def __getitem__(self, index):
        input_dict = self.dataset[index]
        return input_dict

    def __len__(self):
        return len(self.idxs)


class FashionMNIST(datasets.FashionMNIST):
    def __init__(self, root, mode, download, logger, transform=None):
        train = True if mode == 'train' or mode == 'unlabeled' else False
        super(FashionMNIST, self).__init__(root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.array(Image.fromarray(img.numpy(), mode='L'))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        input_dict = {
            'inputs': img,
            'labels': target,
            'indices': index
        }
        return input_dict


class SVHN(datasets.SVHN):
    def __init__(self, root, mode, download, logger, transform=None):
        split = 'train' if mode == 'train' or mode == 'unlabeled' else 'test'
        super(SVHN, self).__init__(os.path.join(root, 'SVHN'), split=split, download=download, transform=transform)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.array(Image.fromarray(np.transpose(img, (1, 2, 0))))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        input_dict = {
            'inputs': img,
            'labels': target,
            'indices': index
        }
        return input_dict


