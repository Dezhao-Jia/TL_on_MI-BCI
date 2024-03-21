import copy

import torch
import random
import numpy as np
import torch.nn as nn
import torch.backends.cudnn

from Datasets import LoadData
from torch.utils.data import DataLoader
from neural_nets.SB_FCN import SB_FCN


class Process:
    def __init__(self, args):
        self.args = args
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.max_index = None
        self.loss_fn = nn.NLLLoss()
        self.eeg_fold_list = LoadData(self.args.sub_id, self.args.windows, self.args.k_fold).load_data()

    def torch_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def training(self):
        net_list = []
        test_corr_list = []

        for fold_num, eeg_list in enumerate(self.eeg_fold_list):
            net = SB_FCN(in_chans=22, n_classes=4, drop_prob=self.args.drop_prob).to(self.device)
            optim = torch.optim.Adam(net.parameters(), lr=self.args.lr)

            train_iter = DataLoader(eeg_list[0], batch_size=self.args.batch_size, shuffle=True)
            test_iter = DataLoader(eeg_list[-1], batch_size=self.args.batch_size, shuffle=False)

            net_dict, test_corr = self.do_train(fold_num, net, optim, train_iter, test_iter)

            net_list.append(net_dict)
            test_corr_list.append(test_corr)

            fold_mess = 'fold {:2d}, test_corr {:.4f}'.format(fold_num + 1, test_corr)
            print(fold_mess)

        save_path = 'check_points/SB_FCN/sub' + str(self.args.sub_id) + '.pth'
        print("save_path :", save_path)
        torch.save({'lr': self.args.lr, 'seed': self.args.seed, 'net_list':net_list,'test_corr': test_corr_list}, save_path)

        test_corr_max = 100.0 * max(test_corr_list)
        sub_mess = 'sub {:2d}, test_corr_max {:.4f}'.format(int(self.args.sub_id), test_corr_max)
        
        print('=' * 50)
        print(sub_mess)
        print('=' * 50)

    def do_train(self, fold_num, net, optim, train_iter, eval_iter):
        best_net = None
        early_stop_epoch = 120
        remain_epoch = early_stop_epoch
        corr_max = 0.0
        test_corr_max_list = []
        for epoch in range(self.args.max_epochs):
            net.train()
            for t_data, t_labels in train_iter:
                t_data, t_labels = t_data.to(self.device), t_labels.to(self.device)
                _, t_outs = net(t_data)
                t_loss = self.loss_fn(t_outs, t_labels)
                optim.zero_grad()
                t_loss.backward()
                optim.step()

            train_corr_sum, train_loss_sum = self.evaluate_corr(net, train_iter)
            test_corr_sum, test_loss_sum = self.evaluate_corr(net, eval_iter)

            test_corr_max_list.append(test_corr_sum)

            remain_epoch -= 1

            if corr_max < test_corr_sum:
                corr_max = test_corr_sum
                best_net = copy.deepcopy(net.state_dict())
                remain_epoch = early_stop_epoch

            if epoch % 10 == 0:
                mess = "fold {:2d}, epoch {:3d}," \
                       " train_loss {:.5f}, train_corr {:.4f}, test_loss {:.5f}, test_corr {:.4f}" \
                    .format(fold_num + 1, epoch, train_loss_sum, train_corr_sum, test_loss_sum, test_corr_sum)
                print(mess)

            if remain_epoch <= 0:
                break

        max_index = test_corr_max_list.index(max(test_corr_max_list))
        test_corr = test_corr_max_list[max_index]

        return best_net, test_corr

    def evaluate_corr(self, net, data_iter):
        corr_sum, loss_sum = 0.0, 0.0
        size = 0
        with torch.no_grad():
            net.eval()
            for data, labels in data_iter:
                data, labels = data.to(self.device), labels.to(self.device)
                _, outs = net(data)
                loss = self.loss_fn(outs, labels)
                _, pred = torch.max(outs.data, dim=1)
                corr = pred.eq(labels.data).cpu().sum()
                corr_sum += corr
                loss_sum += loss
                k = labels.data.shape[0]
                size += k
            corr_sum = corr_sum / size
            loss_sum = loss_sum / size

        return corr_sum, loss_sum

    def running(self):
        self.torch_seed()
        self.training()
