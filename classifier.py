import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
from copy import deepcopy


def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias, 0.0)


class CLASSIFIER:
    def __init__(
        self,
        opt,
        _train_X,
        _train_Y,
        data_loader,
        test_seen_feature,
        test_unseen_feature,
        _nclass,
        _cuda,
        _lr=0.001,
        _beta1=0.5,
        _nepoch=30,
        _batch_size=100,
        generalized=True,
        MCA=True
    ):
        self.train_X =  _train_X
        self.train_Y = _train_Y
        self.test_seen_feature = test_seen_feature
        self.test_seen_label = data_loader.test_seen_label

        self.test_unseen_feature = test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label

        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.ntrain_class = data_loader.ntrain_class
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _train_X.shape[1]
        self.cuda = _cuda
        self.MCA = MCA
        self.model =  LINEAR_LOGSOFTMAX(self.input_dim, self.nclass)
        self.model.apply(weights_init)
        self.criterion = nn.NLLLoss()
        self.opt = opt
        self.input = torch.FloatTensor(_batch_size, self.input_dim)
        self.label = torch.LongTensor(_batch_size)

        self.lr = _lr
        self.beta1 = _beta1
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))

        if self.cuda:
            self.model.to(opt.gpu)
            self.criterion.to(opt.gpu)
            self.input = self.input.to(opt.gpu)
            self.label = self.label.to(opt.gpu)

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.shape[0]

        self.best_model_state_dict = None
        self.best_optimizer_state_dict = None

        if generalized:
            self.acc_seen, self.acc_unseen, self.H = self.fit()
        else:
            self.acc = self.fit_zsl()

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]

            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]

    # ZSL
    def fit_zsl(self):
        best_acc = 0
        mean_loss = 0
        for _ in range(self.nepoch):
            for _ in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                mean_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            acc = self.val(self.test_unseen_feature, self.test_unseen_label)

            if acc > best_acc:
                best_acc = acc
                self.best_model_state_dict = deepcopy(self.model.state_dict())
                self.best_optimizer_state_dict = deepcopy(self.optimizer.state_dict())
        return best_acc * 100

    # GZSL
    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        for _ in range(self.nepoch):
            for _ in range(0, self.ntrain, self.batch_size):
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                inputv = Variable(self.input)
                labelv = Variable(self.label)

                output = self.model(inputv)

                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()

            acc_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label)
            acc_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label+self.ntrain_class)
            H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)

            if H > best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
                self.best_model_state_dict = deepcopy(self.model.state_dict())
                self.best_optimizer_state_dict = deepcopy(self.optimizer.state_dict())

        return best_seen * 100, best_unseen * 100, best_H * 100

    # ZSL
    def val(self, test_X, test_label):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        with torch.no_grad():
            for i in range(0, ntest, self.batch_size):
                end = min(ntest, start+self.batch_size)
                if self.cuda:
                    output = self.model(test_X[start:end].to(self.opt.gpu))
                else:
                    output = self.model(test_X[start:end])
                _, predicted_label[start:end] = torch.max(output.data, 1)
                start = end
        if self.MCA:
            acc = self.eval_MCA(predicted_label.numpy(), test_label.numpy())
        else:
            acc = (predicted_label.numpy() == test_label.numpy()).mean()

        return acc

    # GZSL
    def val_gzsl(self, test_X, test_label):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        with torch.no_grad():
            for i in range(0, ntest, self.batch_size):
                end = min(ntest, start+self.batch_size)
                if self.cuda:
                    output = self.model(test_X[start:end].to(self.opt.gpu))
                else:
                    output = self.model(test_X[start:end])

                _, predicted_label[start:end] = torch.max(output.data, 1)
                start = end
        if self.MCA:
            acc = self.eval_MCA(predicted_label.numpy(), test_label.numpy())
        else:
            acc = (predicted_label.numpy() == test_label.numpy()).mean()
        return acc

    def eval_MCA(self, preds, y):
        cls_label = np.unique(y)
        acc = list()
        for i in cls_label:
            acc.append((preds[y == i] == i).mean())
        return np.asarray(acc).mean()


class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o
