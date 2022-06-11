import torch.nn.init as init
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import torch.nn.functional as F
import torch.nn as nn


def class_scores_in_matrix(opt, mapNet, data, embed, input_label, relation_net):
    expand_embed = embed.unsqueeze(dim=1).repeat(1, data.ntrain_class, 1).reshape(embed.shape[0] * data.ntrain_class, -1)
    attribute_seen = torch.from_numpy(data.train_att).to(opt.gpu)
    attribute_seen = mapNet(attribute_seen).cpu()
    expand_att = attribute_seen.unsqueeze(dim=0).repeat(embed.shape[0], 1, 1).reshape(
        embed.shape[0] * data.ntrain_class, -1).to(opt.gpu)
    all_scores = torch.div(relation_net(torch.cat((expand_embed, expand_att), dim=1)), 0.1).reshape(embed.shape[0],
                                                                                                    data.ntrain_class)
    score_max, _ = torch.max(all_scores, dim=1, keepdim=True)
    scores_norm = all_scores - score_max.detach()
    mask = F.one_hot(input_label, num_classes=data.ntrain_class).float().to(opt.gpu)
    exp_scores = torch.exp(scores_norm)
    log_scores = scores_norm - torch.log(exp_scores.sum(1, keepdim=True))
    cls_loss = -((mask * log_scores).sum(1) / mask.sum(1)).mean()
    return cls_loss

def permute_dims(hs,hn):
    assert hs.dim() == 2
    assert hn.dim() == 2

    B, _ = hs.size()

    perm = torch.randperm(B).to(hs.device)
    perm_hs= hs[perm]
    perm = torch.randperm(B).to(hs.device)
    perm_hn= hn[perm]

    return perm_hs, perm_hn

def multinomial_loss_function(x_logit, x, z_mu, z_var, z, beta=1.):

    """
    Computes the cross entropy loss function while summing over batch dimension, not averaged!
    :param x_logit: shape: (batch_size, num_classes * num_channels, pixel_width, pixel_height), real valued logits
    :param x: shape (batchsize, num_channels, pixel_width, pixel_height), pixel values rescaled between [0, 1].
    :param z_mu: mean of z_0
    :param z_var: variance of z_0
    :param z_0: first stochastic latent variable
    :param z_k: last stochastic latent variable
    :param ldj: log det jacobian
    :param args: global parameter settings
    :param beta: beta for kl loss
    :return: loss, ce, kl
    """

    batch_size = x.size(0)
    target = x
    ce = nn.MSELoss(reduction='sum')(x_logit, target)
    kl = - (0.5 * torch.sum(1 + z_var.log() - z_mu.pow(2) - z_var.log().exp()))
    loss = ce + beta * kl
    loss = loss / float(batch_size)
    ce = ce / float(batch_size)
    kl = kl / float(batch_size)

    return loss, kl


class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_iter = 0.0
        self.best_acc_S_T = 0.0
        self.best_acc_U_T = 0.0
        self.acc_list = []
        self.iter_list = []
        self.save_model = False

    def update(self, it, acc):
        self.acc_list += [acc]
        self.iter_list += [it]
        self.save_model = False
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_iter = it
            self.save_model = True

    def update_gzsl(self, it, acc_u, acc_s, H):
        self.acc_list += [H]
        self.iter_list += [it]
        self.save_model = False
        if H > self.best_acc:
            self.best_acc = H
            self.best_iter = it
            self.best_acc_U_T = acc_u
            self.best_acc_S_T = acc_s
            self.save_model = True


def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.normal_(m.weight.data,  mean=0, std=0.02)
        init.constant_(m.bias, 0.0)

def log_print(s, log):
    print(s)
    with open(log, 'a') as f:
        f.write(s + '\n')

def synthesize_feature_test_ori(vaeNet, dataset, opt):
    gen_feat = torch.FloatTensor(dataset.ntest_class * opt.nSample, opt.X_dim)
    gen_label = np.zeros([0])
    with torch.no_grad():
        for i in range(dataset.ntest_class):
            text_feat = np.tile(dataset.test_att[i].astype('float32'), (opt.nSample, 1))
            text_feat = torch.from_numpy(text_feat).to(opt.gpu)
            z = torch.randn(opt.nSample, opt.Z_dim).to(opt.gpu)
            G_sample = vaeNet.decode(z, text_feat)
            gen_feat[i*opt.nSample:(i+1)*opt.nSample] = G_sample
            gen_label = np.hstack((gen_label, np.ones([opt.nSample])*i))
    return gen_feat, torch.from_numpy(gen_label.astype(int))

def synthesize_feature_test(vaeNet, att_enc, dataset, opt):
    gen_feat = torch.FloatTensor(dataset.ntest_class * opt.nSample, opt.A_dim)
    gen_label = np.zeros([0])
    with torch.no_grad():
        for i in range(dataset.ntest_class):
            test_feat = np.tile(dataset.test_att[i].astype('float32'), (opt.nSample, 1))
            test_feat = torch.from_numpy(test_feat).to(opt.gpu)
            z = torch.randn(opt.nSample, opt.Z_dim).to(opt.gpu)

            G_sample = att_enc(vaeNet.decode(z, test_feat))
            gen_feat[i*opt.nSample:(i+1)*opt.nSample] = G_sample
            gen_label = np.hstack((gen_label, np.ones([opt.nSample])*i))
    return gen_feat, torch.from_numpy(gen_label.astype(int))

def save_model(it, netG, netD, random_seed, log, fout):
    torch.save({
        'it': it + 1,
        'state_dict_G': netG.state_dict(),
        'state_dict_D': netD.state_dict(),
        'random_seed': random_seed,
        'log': log,
    }, fout)

def save_model(it, model, random_seed, log, fout):
    torch.save({
        'it': it + 1,
        'state_dict': model.state_dict(),
        'random_seed': random_seed,
        'log': log,
    }, fout)

def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes):
    acc_per_class = 0
    for i in test_label.unique():
        idx = (test_label == i)
        acc_per_class += float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
    acc_per_class /= float(target_classes.size(0))
    return acc_per_class

def eval_MCA(preds, y):
    cls_label = np.unique(y)
    acc = list()
    for i in cls_label:
        acc.append((preds[y == i] == i).mean())
    return np.asarray(acc).mean()


# clear those instances that have no positive instances to avoid training error
class SupConLoss_clear(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss_clear, self).__init__()
        self.temperature = temperature

    def forward(self, opt, features, labels):

        device = opt.gpu

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        # normalize the logits for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        single_samples = (mask.sum(1) == 0).float()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # invoid to devide the zero
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+single_samples)

        # loss
        # filter those single sample
        loss = - mean_log_prob_pos*(1-single_samples)
        loss = loss.sum()/(loss.shape[0]-single_samples.sum())

        return loss
