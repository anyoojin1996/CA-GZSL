from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.z_size = args.Z_dim
        self.input_size = args.S_dim
        self.args = args
        self.q_z_nn_output_dim = args.q_z_nn_output_dim
        self.q_z_nn, self.q_z_mean, self.q_z_var = self.create_encoder()
        self.p_x_nn, self.p_x_mean = self.create_decoder()
        self.FloatTensor = torch.FloatTensor

    def create_encoder(self):
        q_z_nn = nn.Sequential(
            nn.Linear(self.args.X_dim + self.args.A_dim, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 2048),
            nn.Dropout(self.args.vae_enc_drop),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, self.q_z_nn_output_dim)
        )
        q_z_mean = nn.Linear(self.q_z_nn_output_dim, self.z_size)

        q_z_var = nn.Sequential(
            nn.Linear(self.q_z_nn_output_dim, self.z_size),
            nn.Dropout(0.2),
            nn.Softplus(),
        )
        return q_z_nn, q_z_mean, q_z_var

    def create_decoder(self):
        p_x_nn = nn.Sequential(
            nn.Linear(self.z_size + self.args.A_dim, 2048),
            nn.Dropout(self.args.vae_dec_drop),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048, 0.8),
            nn.Dropout(self.args.vae_dec_drop),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048, 0.8),
            nn.Dropout(self.args.vae_dec_drop),
            nn.LeakyReLU(0.2, inplace=True)
        )
        p_x_mean = nn.Sequential(
            nn.Linear(2048, self.args.X_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return p_x_nn, p_x_mean

    def reparameterize(self, mu, var):
        std = var.sqrt()
        eps = self.FloatTensor(std.size()).normal_().to(self.args.gpu)
        eps = Variable(eps)
        z = eps.mul(std).add_(mu)
        return z

    def encode(self, x, c):
        input = torch.cat((x,c),1)
        h = self.q_z_nn(input)
        h = h.view(h.size(0), -1)
        mean = self.q_z_mean(h)
        var = self.q_z_var(h)
        return mean, var

    def decode(self, z, c):
        input = torch.cat((z, c), 1)
        h = self.p_x_nn(input)
        x_mean = self.p_x_mean(h)
        return x_mean

    def forward(self, x, c, weights=None):
        z_mu, z_var = self.encode(x, c)
        z = self.reparameterize(z_mu, z_var)
        x_mean = self.decode(z, c)
        return x_mean, z_mu, z_var, z

class AutoEncoder(nn.Module):
    def __init__(self, args):
        super(AutoEncoder, self).__init__()
        self.args = args
        self.encoder = nn.Sequential(
            nn.Linear(args.X_dim, args.A_dim + args.A_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(args.A_dim + args.A_dim, args.A_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(args.A_dim, args.A_dim + args.A_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(args.A_dim + args.A_dim, args.X_dim)
        )
        
    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 2)
        N, C = size
        feat_var = feat.unsqueeze(dim=2).var(dim=2, unbiased=False) + eps
        feat_std = feat_var.sqrt()
        feat_mean = feat.unsqueeze(dim=2).mean(dim=2)
        return feat_mean, feat_std

    def apply_adain(self, content, attribute):
        assert (content.size()[:2] == attribute.size()[:2])
        size = content.size()
        attribute_mean, attribute_std = self.calc_mean_std(attribute)
        content_mean, content_std = self.calc_mean_std(content)
        normalized_feat = (content - content_mean.expand(size)) / content_std.expand(size)
        return normalized_feat * attribute_std.expand(size) + attribute_mean.expand(size)

    def forward(self, x, a):
        z = self.encoder(x)
        z = self.apply_adain(z, a)
        z = self.decoder(z)
        return z

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.encoder = nn.Sequential(
            nn.Linear(args.X_dim, args.A_dim + args.A_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(args.A_dim + args.A_dim, args.A_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return z

class RelationNet(nn.Module):
    def __init__(self, args):
        super(RelationNet, self).__init__()
        self.fc1 = nn.Linear(args.A_dim + args.A_dim, 2048)
        self.fc2 = nn.Linear(2048, 1)

    def forward(self, s, c):
        c_ext = c.unsqueeze(0).repeat(s.shape[0], 1, 1)
        cls_num = c_ext.shape[1] #18

        s_ext = torch.transpose(s.unsqueeze(0).repeat(cls_num, 1, 1), 0, 1)
        relation_pairs = torch.cat((s_ext, c_ext), 2).view(-1, c.shape[1] + s.shape[1])

        relation = nn.ReLU()(self.fc1(relation_pairs))
        relation = nn.Sigmoid()(self.fc2(relation))
        return relation
