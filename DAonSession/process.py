import copy

import torch
import random
import torch.nn as nn
import numpy as np
import torch.backends.cudnn

from DASB_Models import Net
from Datasets import LoadData
from loss_funcs.coral import CORAL
from torch.utils.data import DataLoader


class Process:
    def __init__(self, args):
        self.args = args
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.eeg_fold_list = LoadData(args.sub_id, args.windows).load_data()
        self.loss_fn = nn.CrossEntropyLoss()

    def torch_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def training(self):
        net_list = []
        tag_corr_list = []

        for fold_num, eeg_list in enumerate(self.eeg_fold_list):
            net = Net(drop_prob=self.args.drop_prob, reverse=self.args.if_reverse).to(self.device)
            optim = torch.optim.Adam(net.parameters(), lr=self.args.lr)

            src_iter = DataLoader(eeg_list[0], batch_size=self.args.batch_size, shuffle=True)
            aux_iter = DataLoader(eeg_list[1], batch_size=self.args.batch_size, shuffle=True)
            tag_iter = DataLoader(eeg_list[2], batch_size=self.args.batch_size, shuffle=False)

            net_dict, tag_corr = self.do_train(net, optim, src_iter, aux_iter, tag_iter)
            net_list.append(net_dict)
            tag_corr_list.append(tag_corr)

            fold_mess = 'fold {:2d}, tag_corr {:.4f}'.format(fold_num + 1, tag_corr)
            print(fold_mess)

        path = 'check_points/sub' + str(self.args.sub_id) + '.pth'
        torch.save({'lr': self.args.lr, 'seed': self.args.seed, 'net_list': net_list, 'tag_corr': tag_corr_list}, path)

        tag_corr_mean = 100.0 * sum(tag_corr_list) / len(tag_corr_list)
        tag_corr_max = 100.0 * max(tag_corr_list)
        sub_mess = 'sub {:2d}, tag_corr_max {:.4f}, tag_corr_mean {:.4f}' \
            .format(self.args.sub_id, tag_corr_max, tag_corr_mean)
        print('^' * 50)
        print('sub_mess :')
        print(sub_mess)
        print('^' * 50)

    def do_train(self, net, optim, src_iter, aux_iter, tag_iter):
        best_net = None
        early_stop_epoch = 80
        remain_epoch = early_stop_epoch
        corr_max = 0.0
        tag_corr_list = []
        for epoch in range(self.args.max_epochs):
            net.train()
            feat_src = []
            feat_aux = []
            L_cls, L_dom = 0.0, 0.0
            for src_data, src_labels in src_iter:
                src_data, src_labels = src_data.to(self.device), src_labels.to(self.device)
                feat_s, pred_src_c, pred_src_d = net(src_data)
                feat_src.append(feat_s)
                L_cls_s = self.loss_fn(pred_src_c, src_labels)
                L_dom_s = self.loss_fn(pred_src_d, torch.ones(src_data.shape[0], device=self.device).long())
                L_cls += L_cls_s
                L_dom += L_dom_s

            for aux_data, aux_labels in aux_iter:
                aux_data, aux_labels = aux_data.to(self.device), aux_labels.to(self.device)
                feat_a, pred_aux_c, pred_aux_d = net(aux_data)
                feat_aux.append(feat_a)
                L_cls_a = self.loss_fn(pred_aux_c, aux_labels)
                L_dom_a = self.loss_fn(pred_aux_d, torch.ones(aux_data.shape[0], device=self.device).long())
                L_cls += L_cls_a
                L_dom += L_dom_a

            feat_src = torch.cat(feat_src, dim=0)
            feat_aux = torch.cat(feat_aux, dim=0)
            L_dis = CORAL(feat_src, feat_aux)
            if epoch % 2 == 0:
                loss = L_cls + self.args.alpha * L_dis
            else:
                loss = L_cls + self.args.beta * L_dom
            optim.zero_grad()
            loss.backward()
            optim.step()

            tag_corr_sum, tag_loss_sum = self.evaluate_corr(net, tag_iter)
            tag_corr_list.append(tag_corr_sum)

            remain_epoch -= 1
            if tag_corr_sum > corr_max:
                corr_max = tag_corr_sum
                best_net = copy.deepcopy(net.state_dict())
                remain_epoch = early_stop_epoch

            if epoch % 10 == 0:
                mess = "epoch {:3d}, tag_loss {:.5f}, tag_corr {:.4f}".format(epoch, tag_loss_sum, tag_corr_sum)
                print(mess)

            if remain_epoch <= 0:
                break

        max_index = tag_corr_list.index(max(tag_corr_list))
        tag_corr = tag_corr_list[max_index]

        return best_net, tag_corr

    def evaluate_corr(self, net, data_iter):
        corr_sum, loss_sum = 0.0, 0.0
        size = 0
        with torch.no_grad():
            net.eval()
            for data, labels in data_iter:
                data, labels = data.to(self.device), labels.to(self.device)
                _, outs, _ = net(data)
                loss = self.loss_fn(outs, labels)
                _, pred = torch.max(outs.data, dim=1)
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
