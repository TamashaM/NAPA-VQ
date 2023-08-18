from torchvision import transforms
import numpy as np

from data_managers.cifar100_wrapper import CifarWrapper
from data_managers.data_manager_imagenet import DataManagerImagenet

wrapper_dict = {
    "cifar100": CifarWrapper,
    "imagenet": DataManagerImagenet
}


class DataComposer:
    def __init__(self, dataset, shuffle=True, seed=1993):
        self.dataset = dataset
        self.shuffle = shuffle
        print("shuffling is", shuffle)
        self.wrapper = wrapper_dict[dataset](dataset, shuffle=shuffle, seed=seed)
        self.seed = seed

    def get_train_transform(self):
        return self.wrapper.get_train_transform()

    def get_test_transform(self):
        return self.wrapper.get_test_transform()

    def get_train_dataset(self):
        return self.wrapper.get_train_dataset()

    def get_test_dataset(self):
        return self.wrapper.get_test_dataset()
