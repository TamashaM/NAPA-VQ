import torch
import torch.utils.data
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import argparse
import random

import numpy as np

from Finetune import Finetune
from backbone.ResNet import resnet18_cbam
from data_managers.iCIFAR100 import iCIFAR100
from DataComposer import DataComposer

parser = argparse.ArgumentParser(description='Prototype Augmentation and Self-Supervision for Incremental Learning')
parser.add_argument('--epochs', default=101, type=int, help='Total number of epochs to run')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--data_name', default='cifar100', type=str, help='Dataset name to use')
parser.add_argument('--total_nc', default=100, type=int, help='class number for the dataset')
parser.add_argument('--fg_nc', default=50, type=int, help='the number of classes in first task')
parser.add_argument('--task_num', default=10, type=int, help='the number of incremental steps')
parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--temp', default=0.1, type=float, help='training time temperature')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
parser.add_argument('--save_path', default='model_saved_check/', type=str, help='save files directory')
parser.add_argument('--custom_name', default='new-code', type=str,
                    help='custom name for each experiment')
parser.add_argument('--base_model', default='new-code', type=str, help='base_model for task 0')
parser.add_argument('--seed', default=1993, type=int, help='random seed for each run')
parser.add_argument('--emb_size', default=512, type=int, help='embedding size')
parser.add_argument('--shuffle', action='store_true', help='shuffle class order')

args = parser.parse_args()
print(args)

# torch.manual_seed(2222)
# torch.cuda.manual_seed(2222)
# np.random.seed(2222)
# random.seed(2222)
# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.deterministic = True

def main():
    cuda_index = 'cuda:' + args.gpu
    device = torch.device(cuda_index if torch.cuda.is_available() else "cpu")
    task_size = int((args.total_nc - args.fg_nc) / args.task_num)  # number of classes in each incremental step
    file_name = args.data_name + '_' + str(args.fg_nc) + '_' + str(args.task_num) + '*' + str(task_size) + '_' + \
                args.custom_name
    feature_extractor = resnet18_cbam()

    model = Finetune(args, file_name, feature_extractor, task_size, device)
    class_set = list(range(args.total_nc))
    tb_writer = SummaryWriter("./runs/{}".format(args.custom_name))

    for i in range(args.task_num + 1):
        if i == 0:
            old_class = 0
        else:
            old_class = len(class_set[:args.fg_nc + (i - 1) * task_size])

        if i == 0:
            file_name_base = args.data_name + '_' + str(args.fg_nc) + '_' + str(args.task_num) + '*' + str(
                task_size) + '_' + \
                             args.base_model
            path = args.save_path + file_name_base + '/'
        else:
            file_name_custom = args.data_name + '_' + str(args.fg_nc) + '_' + str(args.task_num) + '*' + str(
                task_size) + '_' + \
                               args.custom_name
            path = args.save_path + file_name_custom + '/'

        model_name = path + '%d_model.pkl' % (args.fg_nc + i * task_size)

        model.before_train(i)

        if os.path.exists(model_name):  # checking if there is a file with this name
            model.model = torch.load(model_name)
            print("model exists for task {}".format(i))
        else:
            print("moving to train")
            model.train(i, old_class=old_class, tb_writer=tb_writer)

        model.after_train(i, old_class=old_class)

    ####### Test ######
    print("############# Test for each Task #############")
    data_composer = DataComposer(args.data_name, seed=args.seed)
    test_dataset = data_composer.get_test_dataset()

    acc_all = []

    for current_task in range(args.task_num + 1):
        class_index = args.fg_nc + current_task * task_size
        filename = args.save_path + file_name + '/' + '%d_model.pkl' % (class_index)
        model = torch.load(filename)
        model.eval()
        acc_up2now = []
        for i in range(current_task + 1):
            if i == 0:
                classes = [0, args.fg_nc]
            else:
                classes = [args.fg_nc + (i - 1) * task_size, args.fg_nc + i * task_size]
            test_dataset.getTestData_up2now(classes)
            test_loader = DataLoader(dataset=test_dataset,
                                     shuffle=True,
                                     batch_size=args.batch_size)
            correct, total = 0.0, 0.0
            for setp, (indexs, imgs, labels) in enumerate(test_loader):
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.no_grad():
                    outputs = model(imgs)
                predicts = torch.max(outputs, dim=1)[1]
                correct += (predicts.cpu() == labels.cpu()).sum()
                total += len(labels)
            accuracy = correct.item() / total
            acc_up2now.append(accuracy)
        if current_task < args.task_num:
            acc_up2now.extend((args.task_num-current_task)*[0])
        acc_all.append(acc_up2now)
        print(acc_up2now)
    print(acc_all)

    print("############# Test for up2now Task #############")
    data_composer = DataComposer(args.data_name, shuffle=args.shuffle, seed=args.seed)
    test_dataset = data_composer.get_test_dataset()

    for current_task in range(args.task_num + 1):
        class_index = args.fg_nc + current_task * task_size
        filename = args.save_path + file_name + '/' + '%d_model.pkl' % (class_index)
        model = torch.load(filename)
        model.to(device)
        model.eval()

        classes = [0, args.fg_nc + current_task * task_size]
        test_dataset.getTestData_up2now(classes)
        test_loader = DataLoader(dataset=test_dataset,
                                 shuffle=True,
                                 batch_size=args.batch_size)
        correct, total = 0.0, 0.0
        for setp, (indexs, imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(imgs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        print(accuracy)


if __name__ == "__main__":
    main()
