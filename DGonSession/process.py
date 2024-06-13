import copy
import torch
import random
import torch.backends.cudnn

import numpy as np
import torch.nn as nn

from Datasets import LoadData
from torch.utils.data import DataLoader
from neural_nets.DGSEB_FCN import DGSEB_FCN
from loss_funcs.matrix_orth import orth
from loss_funcs.dynamic_center import CenterLoss, CenterDisc


class Process:
    def __init__(self, args):
        self.args = args
        self.loss_fn = nn.NLLLoss()
        self.center_loss = CenterLoss(num_classes=4)
        self.center_disc = CenterDisc(num_classes=4)
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.eeg_list = LoadData(args.sub_id, args.windows).load_data()

    def torch_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def training(self):
        self.torch_seed()
        net = DGSEB_FCN(in_chans=22, n_classes=4).to(self.device)
        optim = torch.optim.Adam(net.parameters(), lr=self.args.lr)
        src_iter = DataLoader(self.eeg_list[0], batch_size=self.args.batch_size, shuffle=True)
        tag_iter = DataLoader(self.eeg_list[-1], batch_size=self.args.batch_size, shuffle=False)

        net_dict, tag_corr = self.do_train(net, optim, src_iter, tag_iter)
        sub_mess = 'sub_id{:2d}, tag_corr{:.4f}'.format(self.args.sub_id, tag_corr)
        print('=' * 80)
        print(sub_mess)
        print('=' * 80)

        path = 'check_points/sub' + str(self.args.sub_id) + '.pth'
        torch.save({'lr': self.args.lr, 'seed': self.args.seed, 'best_net': net_dict, 'tag_corr': tag_corr}, path)

    def do_train(self, net, optim, src_iter, tag_iter):
        best_net = None
        corr_max = 0.0
        early_stop_epoch = 80
        remain_epoch = early_stop_epoch
        tag_corr_max_list = []

        for epoch in range(self.args.max_epochs):
            net.train()
            for data, labels in src_iter:
                data, labels = data.to(self.device), labels.to(self.device)
                feat, out_b, out_g = net(data)
                weight_b = net.b_mode.weight
                weight_g = net.g_mode.weight

                L_bias = self.loss_fn(out_b, labels)
                L_orth = orth(weight_b, weight_g)
                L_cls = self.loss_fn(out_g, labels)
                L_center = self.center_loss(feat, labels) + self.center_disc(feat, labels)
                loss = L_cls + self.args.alpha * L_center + self.args.beta * L_orth + self.args.gamma * L_bias

                optim.zero_grad()
                loss.backward()
                optim.step()

            src_corr, src_loss = self.evaluate_corr(net, src_iter)
            tag_corr, tag_loss = self.evaluate_corr(net, tag_iter)
            tag_corr_max_list.append(tag_corr)

            remain_epoch -= 1
            if tag_corr > corr_max:
                corr_max = tag_corr
                best_net = copy.deepcopy(net.state_dict())
                remain_epoch = early_stop_epoch

            if remain_epoch <= 0:
                break

            if epoch % 10 == 0:
                mes = 'epoch {:3d}, src_loss {:.5f}, src_corr {:.4f}, tag_loss {:.5f}, tag_corr {:.4f}'\
                    .format(epoch, src_loss, src_corr, tag_loss, tag_corr)
                print(mes)

        max_index = tag_corr_max_list.index(max(tag_corr_max_list))
        tag_corr = tag_corr_max_list[max_index]

        return best_net, tag_corr

    def evaluate_corr(self, net, data_iter):
        corr_sum, loss_sum = 0.0, 0.0
        size = 0
        with torch.no_grad():
            net.eval()
            for data, labels in data_iter:
                data, labels = data.to(self.device), labels.to(self.device)
                _, out_b, out_g = net(data)
                weight_b = net.b_mode.weight
                weight_g = net.g_mode.weight
                loss_bias = self.loss_fn(out_b, labels)
                loss_reg = orth(weight_b, weight_g)
                loss_gen = self.loss_fn(out_g, labels)
                loss = loss_bias + loss_reg + loss_gen
                _, pred = torch.max(out_g.data, dim=1)
                corr = pred.eq(labels.data).cpu().sum()
                corr_sum += corr
                loss_sum += loss
                k = labels.data.shape[0]
                size += k
            corr_sum = corr_sum / size
            loss_sum = loss_sum / len(data_iter)

        return corr_sum, loss_sum

    def running(self):
        self.torch_seed()
        self.training()
