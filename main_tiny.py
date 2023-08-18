import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse

from NAPAVQ_tiny import NAPAVQ
from backbone.ResNet import resnet18_cbam
from data_managers.data_manager_tiny import *


parser = argparse.ArgumentParser(description='Neighborhood Aware Prototype Augmentation with Vector Quantization')
parser.add_argument('--epochs', default=50, type=int, help='Total number of epochs to run')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency (default: 10)')
parser.add_argument('--data_name', default='tiny', type=str, help='Dataset name to use')
parser.add_argument('--total_nc', default=200, type=int, help='class number for the dataset')
parser.add_argument('--fg_nc', default=100, type=int, help='the number of classes in first task')
parser.add_argument('--task_num', default=10, type=int, help='the number of incremental steps')
parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--sng_learning_rate', default=5, type=float, help='initial learning rate for sng module')
parser.add_argument('--protoAug_weight', default=10.0, type=float, help='protoAug loss weight')
parser.add_argument('--kd_weight', default=10.0, type=float, help='knowledge distillation loss weight')
parser.add_argument('--temp', default=0.1, type=float, help='training time temperature')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
parser.add_argument('--save_path', default='model_saved_check/', type=str, help='save files directory')
parser.add_argument('--custom_name', default='new-code-tiny', type=str,
                    help='custom name for each experiment')
parser.add_argument('--base_model', default='new-code-tiny', type=str, help='base_model for task 0')
parser.add_argument('--seed', default=1993, type=int, help='random seed for each run')
parser.add_argument('--emb_size', default=512, type=int, help='embedding size')
parser.add_argument('--shuffle', action='store_true', help='shuffle class order')

args = parser.parse_args()
print(args)


def _get_dist_each_class(feature, navq):
    features = feature.unsqueeze(1)
    cvs = navq.cvs.unsqueeze(0).repeat(feature.size(0), 1, 1)
    dist = torch.cdist(features, cvs).squeeze(1)

    return -dist


def main():
    cuda_index = 'cuda:' + args.gpu
    device = torch.device(cuda_index if torch.cuda.is_available() else "cpu")
    task_size = int((args.total_nc - args.fg_nc) / args.task_num)  # number of classes in each incremental step
    file_name = args.data_name + '_' + str(args.fg_nc) + '_' + str(args.task_num) + '*' + str(task_size) + '_' + \
                args.custom_name
    feature_extractor = resnet18_cbam()

    model = NAPAVQ(args, file_name, feature_extractor, task_size, device)
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
        model_name_cl = path + '%d_cl.pkl' % (args.fg_nc + i * task_size)

        model.before_train(i)

        if os.path.exists(model_name):  # checking if there is a file with this name
            model.model = torch.load(model_name)
            model.navq = torch.load(model_name_cl)
            print("model exists for task {}".format(i))
        else:
            print("moving to train")
            model.train(i, old_class=old_class, tb_writer=tb_writer)

        model.after_train(i, old_class=old_class)

    data_manager = DataManagerTiny(shuffle=args.shuffle, seed=args.seed)

    ####### Test ######

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    print("############# Test for each Task #############")
    acc_all_ncm = []

    for current_task in range(args.task_num + 1):
        class_index = args.fg_nc + current_task * task_size
        filename = args.save_path + file_name + '/' + '%d_model.pkl' % (class_index)
        filename_cl = args.save_path + file_name + '/' + '%d_cl.pkl' % (class_index)
        model = torch.load(filename)
        NAVQ = torch.load(filename_cl)

        model.eval()
        NAVQ.eval()

        acc_up2now_ncm = []
        for i in range(current_task + 1):
            if i == 0:
                classes = class_set[:args.fg_nc]
            else:
                classes = class_set[(args.fg_nc + (i - 1) * task_size):(args.fg_nc + i * task_size)]

            test_dataset = data_manager.get_dataset(test_transform, index=classes, train=False)
            test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)
            correct, total, correct_ncm = 0.0, 0.0, 0.0
            for setp, (imgs, labels) in enumerate(test_loader):
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.no_grad():
                    features = model.feature(imgs)
                    features_norm = (features.T / torch.norm(features.T, dim=0)).T
                total += len(labels)

                cvs_copy = NAVQ.cvs.detach().clone()[::4, :]
                cvs_norm = (cvs_copy.T / torch.norm(cvs_copy.T, dim=0)).T.to(device)

                sqd = torch.cdist(cvs_norm, features_norm)
                predicts_ncm = torch.argmax((-sqd).T, dim=1)
                correct_ncm += (predicts_ncm.cpu() == labels.cpu()).sum()

            accuracy_ncm = correct_ncm.item() / total
            acc_up2now_ncm.append(accuracy_ncm)

        if current_task < args.task_num:
            acc_up2now_ncm.extend((args.task_num - current_task) * [0])
        acc_all_ncm.append(acc_up2now_ncm)
        print(acc_up2now_ncm)
    print( acc_all_ncm)

    print("############# Test for up2now Task #############")
    average_acc_ncm = 0
    for current_task in range(args.task_num + 1):
        class_index = args.fg_nc + current_task * task_size
        filename = args.save_path + file_name + '/' + '%d_model.pkl' % (class_index)
        filename_cl = args.save_path + file_name + '/' + '%d_cl.pkl' % (class_index)
        model = torch.load(filename)
        NAVQ = torch.load(filename_cl)
        NAVQ.to(device)
        model.to(device)

        model.eval()
        NAVQ.eval()

        classes = class_set[:args.fg_nc + current_task * task_size]
        test_dataset = data_manager.get_dataset(test_transform, index=classes, train=False)
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)
        total, correct_ncm = 0.0, 0.0
        for setp, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                features = model.feature(imgs)
                features_norm = (features.T / torch.norm(features.T, dim=0)).T
            total += len(labels)

            cvs_copy = NAVQ.cvs.detach().clone()[::4, :]
            cvs_norm = (cvs_copy.T / torch.norm(cvs_copy.T, dim=0)).T.to(device)
            sqd = torch.cdist(cvs_norm, features_norm)
            predicts_ncm = torch.argmax((-sqd).T, dim=1)
            correct_ncm += (predicts_ncm.cpu() == labels.cpu()).sum()

        accuracy_ncm = correct_ncm.item() / total

        print(accuracy_ncm)
        average_acc_ncm += accuracy_ncm
    print('average acc: ')
    print(average_acc_ncm / (args.task_num + 1))


if __name__ == "__main__":
    main()
