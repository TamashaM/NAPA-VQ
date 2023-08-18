import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np

import os
import copy

from incrementalNetwork import network
from DataComposer import DataComposer


class Finetune:
    def __init__(self, args, file_name, feature_extractor, task_size, device):
        self.file_name = file_name
        self.args = args
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.model = network(args.fg_nc, feature_extractor)
        self.class_label = None
        self.num_class = args.fg_nc
        self.task_size = task_size
        self.device = device
        self.old_model = None
        self.data_composer = DataComposer(args.data_name, args.shuffle, args.seed)
        self.train_dataset = self.data_composer.get_train_dataset()
        self.test_dataset = self.data_composer.get_test_dataset()
        self.train_loader = None
        self.test_loader = None

    def before_train(self, current_task):
        self.model.eval()
        if current_task == 0:
            classes = [0, self.num_class]
        else:
            classes = [self.num_class - self.task_size, self.num_class]
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)

        if current_task > 0:
            self.model.Incremental_learning(self.num_class)

        self.model.train()
        self.model.to(self.device)

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.args.batch_size)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.args.batch_size)

        return train_loader, test_loader

    def _get_test_dataloader(self, classes):
        self.test_dataset.getTestData_up2now(classes)
        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.args.batch_size)
        return test_loader

    def train(self, current_task, old_class=0, tb_writer=None):
        if current_task == 0:
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=2e-4)
        else:
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate * 0.1, weight_decay=2e-4)

        scheduler = StepLR(self.opt, step_size=45, gamma=0.1)

        for epoch in range(self.epochs):
            total_loss = 0
            for step, (indices, images, target) in enumerate(self.train_loader):
                images, target = images.to(self.device), target.to(self.device)

                self.opt.zero_grad()

                loss = self._compute_loss_cce(images, target, old_class)

                total_loss += loss.item()

                loss.backward()
                self.opt.step()
            scheduler.step()
            if epoch % self.args.print_freq == 0:
                accuracy = self._test(self.test_loader)
                print('epoch:%d, accuracy:%.5f' % (epoch, accuracy))
            accuracy_train = self._test(self.train_loader)
            overall_epoch = self.epochs * current_task + epoch
            tb_writer.add_scalar('Accuracy_softmax/train', accuracy_train, overall_epoch)

            tb_writer.add_scalar('Loss_total/train', total_loss / len(self.train_loader),
                                 overall_epoch)

    def _test(self, testloader, mode=0):
        self.model.eval()
        correct, total = 0.0, 0.0

        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            with torch.no_grad():
                outputs = self.model(imgs)

            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)

        accuracy = correct.item() / total

        self.model.train()

        return accuracy

    def _compute_loss_cce(self, imgs, target, old_class=0):

        output = self.model(imgs)
        output, target = output.to(self.device), target.to(self.device)

        loss_cce = nn.CrossEntropyLoss()(output / self.args.temp, target)

        return loss_cce

    def after_train(self, current_task, old_class):

        path = self.args.save_path + self.file_name + '/'
        if not os.path.isdir(path):
            os.makedirs(path)
        filename = path + '%d_model.pkl' % (self.num_class)
        accuracy = self._test(self.test_loader, mode=1)
        print('Final, accuracy:%.5f' % (
            accuracy))
        self.num_class += self.task_size

        torch.save(self.model, filename)
        self.old_model = torch.load(filename)
        self.old_model.to(self.device)
        self.old_model.eval()

