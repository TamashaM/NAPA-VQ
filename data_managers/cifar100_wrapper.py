from torchvision import transforms
import numpy as np

from data_managers.iCIFAR100 import iCIFAR100


def map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def setup_data(test_targets, shuffle, seed):
    order = [i for i in range(len(np.unique(test_targets)))]
    if shuffle:
        np.random.seed(seed)
        order = np.random.permutation(len(order)).tolist()
    else:
        order = range(len(order))
    class_order = order
    print(100 * '#')
    print(class_order)
    return map_new_class_index(test_targets, class_order)


class CifarWrapper:
    def __init__(self, dataset, shuffle=True, seed=1993):
        self.dataset = dataset
        self.train_transform = transforms.Compose([transforms.RandomCrop((32, 32), padding=4),
                                                   transforms.RandomHorizontalFlip(p=0.5),
                                                   transforms.ColorJitter(brightness=0.24705882352941178),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                        (0.2675, 0.2565, 0.2761))])
        self.test_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                       (0.2675, 0.2565, 0.2761))])
        self.seed = seed
        self.shuffle = shuffle

    def get_train_transform(self):
        return self.train_transform

    def get_test_transform(self):
        return self.test_transform

    def get_train_dataset(self):
        train_dataset = iCIFAR100('../dataset',
                                  transform=self.train_transform,
                                  download=True)
        train_dataset.targets = setup_data(train_dataset.targets, shuffle=self.shuffle, seed=self.seed)

        return train_dataset

    def get_test_dataset(self):
        test_dataset = iCIFAR100('../dataset',
                                 test_transform=self.test_transform,
                                 train=False,
                                 download=True)
        test_dataset.targets = setup_data(test_dataset.targets, shuffle=self.shuffle, seed=self.seed)

        return test_dataset
