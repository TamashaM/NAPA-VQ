import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import copy

from incrementalNetwork import network
from data_managers.data_manager_imagenet import *
from losses.na_loss import NAVQ


def _get_dist_each_class(feature, navq):
    features = feature.unsqueeze(1)
    cvs = navq.cvs.unsqueeze(0).repeat(feature.size(0), 1, 1)
    dist = torch.cdist(features, cvs).squeeze(1)

    return -dist


class NAPAVQ:
    def __init__(self, args, file_name, feature_extractor, task_size, device):
        self.file_name = file_name
        self.args = args
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.model = nn.DataParallel(network(args.fg_nc * 4, feature_extractor))
        self.radius = 0
        self.size = 224
        self.prototype = None
        self.class_label = None
        self.num_class = args.fg_nc
        self.task_size = task_size
        self.device = device
        self.old_model = None
        self.data_manager = DataManagerImagenet(shuffle=args.shuffle, seed=args.seed)
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.test_transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.train_loader = None
        self.test_loader = None

        self.navq = NAVQ(
            num_classes=self.num_class * 4,
            feat_dim=self.args.emb_size,
            device=device,
        )

        self.prototype_dict = {}

    def _before_update(self, current_task):
        # setting the gradients of old coding vectors to be 0
        if current_task > 0:
            classes_old = range((self.num_class - self.task_size) * 4)
            self.navq.cvs.grad[classes_old, :] *= 0

    def before_train(self, current_task):
        self.model.eval()
        class_set = list(range(self.args.total_nc))
        if current_task == 0:
            classes = class_set[:self.num_class]
        else:
            classes = class_set[self.num_class - self.task_size: self.num_class]
        print(classes)

        trainfolder = self.data_manager.get_dataset(self.train_transform, index=classes, train=True)
        testfolder = self.data_manager.get_dataset(self.test_transform, index=class_set[:self.num_class], train=False)

        self.train_loader = torch.utils.data.DataLoader(trainfolder, batch_size=self.args.batch_size,
                                                        shuffle=True, drop_last=True, num_workers=8)
        self.test_loader = torch.utils.data.DataLoader(testfolder, batch_size=self.args.batch_size,
                                                       shuffle=False, drop_last=False, num_workers=8)
        if current_task > 0:
            self.model.module.Incremental_learning(4 * self.num_class)

            self.model.to(self.device)
            self.proto_save(self.model, self.train_loader)

            self.navq.add_cvs(self.task_size * 4)

        self.model.train()
        self.model.to(self.device)

    def train(self, current_task, old_class=0, tb_writer=None):
        if current_task == 0:
            base_lr = 0.1  # Initial learning rate
            lr_strat = [80, 120, 150]  # Epochs where learning rate gets decreased
            lr_factor = 0.1  # Learning rate decrease factor
            custom_weight_decay = 5e-4  # Weight Decay
            custom_momentum = 0.9  # Momentum
            self.opt = torch.optim.SGD(
                self.model.parameters(),
                lr=base_lr,
                momentum=custom_momentum,
                weight_decay=custom_weight_decay)
            scheduler = MultiStepLR(
                self.opt,
                milestones=lr_strat,
                gamma=lr_factor
            )
            self.args.epochs = 160

            self.optimizer_cvs = optim.SGD(self.navq.parameters(), lr=self.args.sng_learning_rate)

        else:
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate * 0.1, weight_decay=2e-4)
            scheduler = StepLR(self.opt, step_size=45, gamma=0.1)
            self.args.epochs = 100

            self.optimizer_cvs = optim.SGD(self.navq.parameters(), lr=self.args.sng_learning_rate * 0.1)

        self.navq.optimizer = self.optimizer_cvs
        scheduler_sng = StepLR(self.optimizer_cvs, step_size=20, gamma=0.1)

        for epoch in range(self.args.epochs):

            total_dce, total_na = 0, 0
            for step, data in enumerate(self.train_loader):
                images, target = data
                images, target = images.to(self.device), target.to(self.device)

                # # self-supervised learning based label augmentation
                images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
                images = images.view(-1, 3, self.size, self.size)
                target = torch.stack([target * 4 + k for k in range(4)], 1).view(-1)

                self.opt.zero_grad()
                self.optimizer_cvs.zero_grad()

                loss_dce, loss_other, loss_na = self._compute_loss(images, target, old_class)
                loss = loss_dce + loss_other + loss_na

                total_dce += loss_dce.item()
                total_na += loss_na.item()
                total_loss = total_dce + total_na

                loss.backward()
                self._before_update(current_task)
                self.opt.step()
                self.optimizer_cvs.step()
            scheduler.step()
            scheduler_sng.step()
            if (epoch) % self.args.print_freq == 0 or epoch == self.epochs - 1:
                accuracy_ncm = self._test(self.test_loader)
                print('epoch:%d, accuracy_ncm:%.5f' % (epoch, accuracy_ncm))
            accuracy_ncm_train = self._test(self.train_loader)
            overall_epoch = self.epochs * current_task + epoch
            tb_writer.add_scalar('Accuracy_ncm/train', accuracy_ncm_train, overall_epoch)

            tb_writer.add_scalar('Loss_dce/train', total_dce / len(self.train_loader),
                                 overall_epoch)
            tb_writer.add_scalar('Loss_na/train', total_na / len(self.train_loader),
                                 overall_epoch)
            tb_writer.add_scalar('Loss_total/train', total_loss / len(self.train_loader),
                                 overall_epoch)

    def _test(self, testloader, mode=0):
        self.model.eval()
        total = 0.0
        correct_ncm = 0.0

        for setp, data in enumerate(testloader):
            imgs, labels = data
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            with torch.no_grad():
                features = self.model.module.feature(imgs)
                features_norm = (features.T / torch.norm(features.T, dim=0)).T

            total += len(labels)

            cvs_copy = self.navq.cvs.detach().clone()
            cvs_norm = (cvs_copy.T / torch.norm(cvs_copy.T, dim=0)).T.to(self.device)

            selected_class_indices = [self.navq.class_indices[i * 4] for i in range(0, self.num_class)]
            selected_class_indices_flat = [item for sublist in selected_class_indices for item in sublist]

            filtered_cvs = torch.index_select(cvs_norm, 0,
                                                  torch.tensor(selected_class_indices_flat).to(self.device))
            filtered_targets = [self.navq.cv_class[i] for i in selected_class_indices_flat]

            result = []
            for target in features_norm.cpu().numpy():
                x = target - filtered_cvs.cpu().numpy()
                x = np.linalg.norm(x, ord=2, axis=1)
                x = np.argmin(x)
                result.append(filtered_targets[x] // 4)

            predicts_ncm = torch.tensor(result)
            correct_ncm += (predicts_ncm.cpu() == labels.cpu()).sum()

        accuracy_ncm = correct_ncm.item() / total
        self.model.train()

        return accuracy_ncm

    def _compute_loss(self, imgs, target, old_class=0):

        feature = self.model.module.feature(imgs)
        output = _get_dist_each_class(feature, self.navq)
        output, target = output.to(self.device), target.to(self.device)

        loss_dce = nn.CrossEntropyLoss()(output / self.args.temp, target)
        loss_na = self.navq(feature, target)

        if self.old_model is None:
            return loss_dce, 0, loss_na
        else:
            feature_old = self.old_model.module.feature(imgs)
            loss_kd = torch.dist(feature, feature_old, 2)

            index = np.arange(old_class)

            # code for NA-PA
            random_indices = index[np.random.choice(len(index), size=self.args.batch_size, replace=True)] * 4
            proto_list = [self.prototype_dict[i] for i in random_indices]
            proto_array = np.array(proto_list)
            proto_neighbours = self.navq.edges.cpu().numpy()[[random_indices]]
            picked_neighbour_indices = np.array([np.random.choice(r.nonzero()[0]) for r in proto_neighbours])
            picked_neighbours = np.array([self.prototype_dict[i] for i in picked_neighbour_indices])
            gammas = np.random.uniform(0.5, 1, self.args.batch_size)
            proto_aug = proto_array * gammas[:, None] + picked_neighbours * (1 - gammas)[:, None]

            proto_aug = torch.tensor(proto_aug, dtype=torch.float).to(self.device)
            proto_aug_label = torch.from_numpy(random_indices).to(self.device)

            loss_na += self.navq(proto_aug, proto_aug_label)

            soft_feat_aug = _get_dist_each_class(proto_aug, self.navq)
            loss_protoAug = nn.CrossEntropyLoss()(soft_feat_aug / self.args.temp, proto_aug_label)

            return loss_dce, self.args.protoAug_weight * loss_protoAug + self.args.kd_weight * loss_kd, loss_na

    def after_train(self, current_task, old_class):

        self.proto_save(self.model, self.train_loader)

        path = self.args.save_path + self.file_name + '/'
        if not os.path.isdir(path):
            os.makedirs(path)
        filename = path + '%d_model.pkl' % (self.num_class)
        filename_cl = path + '%d_cl.pkl' % (self.num_class)
        accuracy_ncm = self._test(self.test_loader, mode=1)
        print('Final, accuracy_ncm:%.5f' % (
            accuracy_ncm,))
        self.num_class += self.task_size

        self.navq.old_class_indices = copy.deepcopy(self.navq.class_indices)

        torch.save(self.model, filename)
        torch.save(self.navq, filename_cl)
        self.old_model = torch.load(filename)
        self.old_model = nn.DataParallel(self.old_model.module)
        self.old_model.to(self.device)
        self.old_model.eval()

        self.old_navq = torch.load(filename_cl)
        self.old_navq.to(self.device)
        self.old_navq.eval()

    def proto_save(self, model, loader):
        features = []
        labels = []
        model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(loader):
                images, target = images.to(self.device), target.to(self.device)
                images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
                images = images.view(-1, 3, self.size, self.size)
                target = torch.stack([target * 4 + k for k in range(4)], 1).view(-1)

                feature = model.module.feature(images)
                if feature.shape[0] == self.args.batch_size * 4:
                    labels.append(target.cpu().numpy())
                    features.append(feature.cpu().numpy())
        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))

        for item in labels_set:
            index = np.where(item == labels)[0]
            feature_classwise = features[index]
            self.prototype_dict[item] = np.mean(feature_classwise, axis=0)
