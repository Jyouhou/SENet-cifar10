import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

class DataLoader(object):
    def __init__(self):

        transform_scheme = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        # training set
        self.train_set = CIFAR10(
            root=f'./data/',
            train=True,
            download=True,
            transform=transform_scheme)
        self.training_set_size = self.train_set.data.shape[0]

        # test set
        self.test_set = CIFAR10(
            root=f'./data/',
            train=False,
            download=True,
            transform=transform_scheme)

    def generator(self, train=True, batch_size=128, GPU_num=4, num_worker=8, CUDA=True):

        def _generator(data_iter):
            for image, label in data_iter:
                if CUDA:
                    image = image.cuda()
                    label = label.cuda()
                yield image, label

        return _generator(torch.utils.data.DataLoader(
            self.train_set if train else self.test_set,
            batch_size=batch_size if train else batch_size // 100 * 100 // GPU_num * GPU_num,
            shuffle=True,
            drop_last=train,
            num_workers=num_worker
        ))